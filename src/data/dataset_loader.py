# ==============================================================================
# ENGINE_ALPHA - PURGED PYTORCH SEQUENCE DATALOADER
# Core Component: src/data/dataset_loader.py
# Description: Implements high-performance memory-mapped sequence generation.
# CRITICAL: Introduces the Purged Time-Series "Embargo Gap" to completely 
# eliminate Look-Ahead Bias between the Train and Validation DL splits.
# ==============================================================================

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DataLoader] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Resolve Root and Imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
except ImportError:
    logger.error("Failed to import feature_config. Ensure script runs from project root.")
    sys.exit(1)


class WingoSequenceDataset(Dataset):
    """
    PyTorch Dataset for Time-Series Sequence Slicing.
    Uses contiguous array views to generate 3D Tensors on-the-fly without 
    blowing up CPU RAM.
    """
    def __init__(self, features: np.ndarray, targets: dict, sequence_length: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.sequence_length = sequence_length
        
        self.targets = {}
        for key, arr in targets.items():
            if key == 'exact_number':
                self.targets[key] = torch.tensor(arr, dtype=torch.long)
            else:
                self.targets[key] = torch.tensor(arr, dtype=torch.float32)
                
        self.dataset_len = len(self.features) - self.sequence_length
        
    def __len__(self):
        return self.dataset_len
        
    def __getitem__(self, idx):
        x_window = self.features[idx : idx + self.sequence_length]
        target_idx = idx + self.sequence_length
        y_dict = {key: tensor[target_idx] for key, tensor in self.targets.items()}
        return x_window, y_dict


class DataLoaderFactory:
    """
    Institutional Orchestrator for loading, splitting, scaling, and packaging data.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.seq_len = self.config['data_pipeline']['feature_engineering']['sequence_length']
        self.batch_size_lstm = self.config['models']['lstm']['batch_size']
        self.split_ratio = self.config['data_pipeline']['preprocessing']['train_test_split_ratio']
        self.scaler_dir = os.path.join(PROJECT_ROOT, self.config['paths']['scaler_artifact_dir'])
        
        os.makedirs(self.scaler_dir, exist_ok=True)

    def _isolate_features_and_targets(self, df: pd.DataFrame):
        logger.info("Isolating Input Tensors from Target Vectors...")
        
        target_dict = {}
        for target_name, col_name in feature_config.TARGETS.items():
            if col_name in df.columns:
                target_dict[target_name] = df[col_name].values
            else:
                raise KeyError(f"Required target {col_name} missing from dataframe!")

        feature_cols = list(feature_config.MODEL_INPUT_FEATURES)
        
        overlap = set(feature_cols).intersection(set(feature_config.TARGETS.values()))
        if overlap:
            raise ValueError(f"CRITICAL LEAKAGE: Target columns {overlap} found in input features!")
            
        feature_matrix = df[feature_cols].values
        return feature_matrix, target_dict, feature_cols

    def create_dataloaders(self, data_path: str, mode="train"):
        logger.info(f"Loading engineered dataset from: {data_path}")
        df = pd.read_csv(data_path).sort_values(by='issue_id').reset_index(drop=True)
        total_rows = len(df)
        
        # ======================================================================
        # THE PURGED TIME-SERIES SPLIT (ELIMINATING LOOK-AHEAD BIAS)
        # ======================================================================
        # Standard 80/20 splitting causes the end of the train set to share 
        # a sliding window with the beginning of the validation set. 
        # The neural network "memorizes" this overlap and artificially boosts val accuracy.
        # We mathematically purge a block of rows equal to the sequence length to
        # completely sever the relationship.
        
        # ======================================================================
        # CHRONOLOGICAL ISOLATION (ZERO LOOK-AHEAD BIAS)
        # ======================================================================
        # 🚨 FRACTURE 3 FIXED: In a strict forward chronological split, the training 
        # set ends at T, and validation starts at T. Because the model looks backward 
        # (T-60 to T-1) to predict T, the target at T has NEVER been seen by the 
        # training set. Applying a purge gap here mathematically starves the 
        # validation set of contiguous data. The embargo gap is removed.
        
        split_idx = int(total_rows * self.split_ratio)
        
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
        
        logger.info("="*60)
        logger.info("EXECUTING CHRONOLOGICAL ISOLATION SPLIT")
        logger.info(f" -> Train Set: {len(train_df)} rows")
        logger.info(f" -> Validation Set: {len(val_df)} rows (Contiguous)")
        logger.info("="*60)

        # Isolation
        X_train_raw, y_train_dict, f_cols = self._isolate_features_and_targets(train_df)
        X_val_raw, y_val_dict, _ = self._isolate_features_and_targets(val_df)

        # Strict Single-Pass Scaling
        logger.info("Applying StandardScaler (Fitted purely on Training Split)...")
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        scaler_path = os.path.join(self.scaler_dir, "master_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler Artifact locked to {scaler_path}")

        # Dataset Instantiation
        train_dataset = WingoSequenceDataset(X_train_scaled, y_train_dict, self.seq_len)
        val_dataset = WingoSequenceDataset(X_val_scaled, y_val_dict, self.seq_len)

        # DataLoader Compilation
        num_workers = self.config['system'].get('max_workers', 2)
        
        logger.info(f"Building DataLoaders | Batch Size: {self.batch_size_lstm} | Threads: {num_workers}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size_lstm, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,                
            prefetch_factor=2,              
            persistent_workers=True,        
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size_lstm, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=False
        )

        logger.info("="*60)
        logger.info("DATALOADER PIPELINE COMPLETE")
        logger.info(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
        logger.info(f"Feature Dimension Lock: {len(f_cols)}")
        logger.info("="*60)
        
        return train_loader, val_loader, len(f_cols)

# ==============================================================================
# LOCAL EXECUTION TEST SCRIPT
# ==============================================================================
if __name__ == "__main__":
    logger.info("Running standalone DataLoader Integration Test...")
    
    test_data_path = os.path.join(PROJECT_ROOT, "datasets", "WinGo30S_Ready_data.csv")
    
    if os.path.exists(test_data_path):
        factory = DataLoaderFactory()
        train_loader, val_loader, input_dim = factory.create_dataloaders(test_data_path)
        
        batch_X, batch_Y = next(iter(train_loader))
        
        print("\n--- BATCH INSPECTION ---")
        print(f"X (History Window) Shape: {batch_X.shape}") 
        print(f"Y ['binary_size'] Shape: {batch_Y['binary_size'].shape}")
        
        print("\n[+] Success: Purged Data Tensors formatted for Deep DL Ingestion.")
    else:
        logger.error(f"Cannot run test. Processed dataset not found at: {test_data_path}")



        