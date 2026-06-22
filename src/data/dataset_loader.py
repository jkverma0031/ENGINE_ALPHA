# ==============================================================================
# ENGINE_ALPHA - PYTORCH SEQUENCE DATALOADER
# Core Component: src/data/dataset_loader.py
# Description: Implements high-performance memory-mapped sequence generation, 
# strict chronological train/val splitting, and target variable isolation.
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
    Uses contiguous array views (In-Memory Pointer Slicing) to generate 3D Tensors 
    on-the-fly without blowing up RAM.
    """
    def __init__(self, features: np.ndarray, targets: dict, sequence_length: int):
        """
        Args:
            features (np.ndarray): 2D array of scaled input features shape (N, F).
            targets (dict): Dictionary mapping target names to 1D arrays of shape (N,).
            sequence_length (int): How many past games the model gets to look at.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.sequence_length = sequence_length
        
        # Convert all target arrays to tensors
        self.targets = {}
        for key, arr in targets.items():
            # Targets for exact_number (0-9) must be LongTensors for CrossEntropyLoss
            if key == 'exact_number':
                self.targets[key] = torch.tensor(arr, dtype=torch.long)
            else:
                # Binary targets (Size, Color) are floats for BCEWithLogitsLoss
                self.targets[key] = torch.tensor(arr, dtype=torch.float32) # Add a dimension for compatibility with loss functions
                
        self.dataset_len = len(self.features) - self.sequence_length
        
    def __len__(self):
        return self.dataset_len
        
    def __getitem__(self, idx):
        """
        The core sequence generator. 
        If sequence_length is 60, and idx is 0:
        X gets rows [0 to 59]. 
        Y gets row [60] (The outcome the model must predict).
        """
        # Slice the feature window (The History)
        x_window = self.features[idx : idx + self.sequence_length]
        
        # Grab the target at the timestep immediately following the window
        target_idx = idx + self.sequence_length
        
        y_dict = {key: tensor[target_idx] for key, tensor in self.targets.items()}
        
        return x_window, y_dict


class DataLoaderFactory:
    """
    Enterprise Orchestrator for loading, splitting, scaling, and packaging data.
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
        """
        Strictly segregates the dataframe into an X matrix and Y dictionary to 
        prevent target leakage into the neural network inputs.
        """
        logger.info("Isolating Input Tensors from Target Vectors...")
        
        # 1. Isolate Targets based on feature_config schema
        target_dict = {}
        for target_name, col_name in feature_config.TARGETS.items():
            if col_name in df.columns:
                target_dict[target_name] = df[col_name].values
            else:
                logger.critical(f"Required target {col_name} missing from dataframe!")
                raise KeyError(col_name)

        # 2. Isolate Features
        feature_cols = list(feature_config.MODEL_INPUT_FEATURES)
        
        # Verify no targets accidentally slipped into feature cols
        overlap = set(feature_cols).intersection(set(feature_config.TARGETS.values()))
        if overlap:
            raise ValueError(f"CRITICAL LEAKAGE: Target columns {overlap} found in input features!")
            
        feature_matrix = df[feature_cols].values
        
        return feature_matrix, target_dict, feature_cols

    def create_dataloaders(self, data_path: str, mode="train"):
        """
        Main execution method. Reads the processed CSV, splits chronologically, 
        scales the data, and returns ready-to-train PyTorch DataLoaders.
        """
        logger.info(f"Loading engineered dataset from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Ensure chronological order
        df = df.sort_values(by='issue_id').reset_index(drop=True)
        total_rows = len(df)
        
        # --- 1. Chronological Train / Validation Split ---
        # Time-series rule: Never shuffle before splitting. The past predicts the future.
        split_idx = int(total_rows * self.split_ratio)
        
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
        
        logger.info(f"Chronological Split -> Train: {len(train_df)} rows | Val: {len(val_df)} rows")

        # --- 2. Isolation ---
        X_train_raw, y_train_dict, f_cols = self._isolate_features_and_targets(train_df)
        X_val_raw, y_val_dict, _ = self._isolate_features_and_targets(val_df)

        # --- 3. Dynamic Scaling (Preventing Future Leakage) ---
        # We MUST fit the scaler ONLY on the training data. If we fit it on the entire 
        # dataset, information from the validation set "leaks" into the training set's mean/std.
        logger.info("Applying StandardScaler to feature matrices...")
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # Save the scaler artifact. The live_engine.py MUST load this exact scaler later.
        scaler_path = os.path.join(self.scaler_dir, "master_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler fitted and saved to {scaler_path}")

        # --- 4. Dataset Instantiation ---
        train_dataset = WingoSequenceDataset(X_train_scaled, y_train_dict, self.seq_len)
        val_dataset = WingoSequenceDataset(X_val_scaled, y_val_dict, self.seq_len)

        # --- 5. DataLoader Instantiation ---
        # pin_memory=True dramatically speeds up CPU RAM to GPU VRAM data transfers
        # num_workers allows background threads to prepare the next batch while the GPU trains
        logger.info(f"Building PyTorch DataLoaders (Batch Size: {self.batch_size_lstm})")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size_lstm, 
            shuffle=True, # We can shuffle the *sequences* now, just not the chronological split
            num_workers=self.config['system']['max_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size_lstm, 
            shuffle=False, # Never shuffle validation sequences
            num_workers=self.config['system']['max_workers'],
            pin_memory=True,
            drop_last=False
        )

        logger.info("="*50)
        logger.info("DATALOADER PIPELINE COMPLETE")
        logger.info(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
        logger.info(f"Input Feature Dimension: {len(f_cols)}")
        logger.info("="*50)
        
        return train_loader, val_loader, len(f_cols)

# ==============================================================================
# LOCAL EXECUTION TEST SCRIPT
# Run this file directly to verify PyTorch Tensor shapes and memory allocation.
# ==============================================================================
if __name__ == "__main__":
    logger.info("Running standalone DataLoader Integration Test...")
    
    test_data_path = os.path.join(PROJECT_ROOT, "datasets", "WinGo30S_Ready_data.csv")
    
    if os.path.exists(test_data_path):
        factory = DataLoaderFactory()
        train_loader, val_loader, input_dim = factory.create_dataloaders(test_data_path)
        
        # Fetch exactly one batch to prove the math works
        batch_X, batch_Y = next(iter(train_loader))
        
        print("\n--- BATCH INSPECTION ---")
        print(f"X (History Window) Shape: {batch_X.shape}") 
        print("   -> Format: [Batch_Size, Sequence_Length, Num_Features]")
        
        print(f"\nY (Targets) Dictionary Keys: {list(batch_Y.keys())}")
        print(f"Y ['binary_size'] Shape: {batch_Y['binary_size'].shape}")
        print(f"Y ['exact_number'] Shape: {batch_Y['exact_number'].shape}")
        
        print("\n[+] Success: Tensors are perfectly formatted for Deep Learning ingestion.")
        
    else:
        logger.error(f"Cannot run test. Processed dataset not found at: {test_data_path}")
        logger.error("Run src/data/feature_factory.py first!")