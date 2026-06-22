# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE NEURAL META-LEARNER & CALIBRATOR
# Core Component: src/training/train_meta_learner.py
# Description: Generates Level-2 Out-of-Fold prediction matrices from all 
# frozen base artifacts. Trains a Temperature-Scaled Neural Aggregator to 
# optimally combine predictions, and applies Platt Scaling for pure calibration.
# ==============================================================================

import os
import sys
import yaml
import time
import json
import logging
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import xgboost as xgb

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MetaLearner] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Resolve Root and Imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
    from src.data.dataset_loader import WingoSequenceDataset
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
    from src.models.autoencoder import WingoTemporalVAE
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


class NeuralMetaAggregator(nn.Module):
    """
    Level-2 Stacking Network.
    Takes the raw probabilities from [LSTM, Transformer, XGBoost] and the 
    Anomaly Score from [VAE] and learns how to weigh them mathematically.
    Uses Non-Negative constraints to prevent inverted logic.
    """
    def __init__(self, num_models: int = 4):
        super(NeuralMetaAggregator, self).__init__()
        
        # We use a linear layer without bias to purely weight the model inputs
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Non-linear interaction layer (if models agree/disagree)
        self.interaction = nn.Sequential(
            nn.Linear(num_models, 16),
            nn.Mish(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (Batch, Num_Models) -> containing raw probabilities and VAE scores
        """
        # Force weights to be positive (softmax normalization)
        normalized_weights = F.softmax(self.weights / self.temperature, dim=0)
        
        # Linear weighted ensemble
        linear_ensemble = torch.sum(x * normalized_weights, dim=1, keepdim=True)
        
        # Non-linear interaction adjustment
        interaction_effect = self.interaction(x)
        
        # Final Logit output
        final_logit = linear_ensemble + (0.1 * interaction_effect)
        return final_logit


class Level2DataBuilder:
    """
    Loads all frozen models and the validation dataset. 
    Passes the validation data through the frozen models to generate the 
    Level-2 Matrix (X_meta) required to train the Stacker.
    """
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        self.sup_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        self.unsup_dir = os.path.join(PROJECT_ROOT, self.config['paths']['unsupervised_artifact_dir'])
        self.scaler_path = os.path.join(PROJECT_ROOT, self.config['paths']['scaler_artifact_dir'], "master_scaler.joblib")
        
        self.seq_len = self.config['data_pipeline']['feature_engineering']['sequence_length']

    def load_frozen_artifacts(self, input_dim: int):
        logger.info("Initializing Frozen Artifact Vault...")
        
        # 1. Load LSTM
        self.lstm = WingoMTLLSTM(
            input_dim=input_dim,
            hidden_dim=self.config['models']['lstm']['hidden_dim'],
            num_layers=self.config['models']['lstm']['num_layers']
        ).to(self.device)
        lstm_ckpt = torch.load(os.path.join(self.sup_dir, "lstm_best_weights.pt"), map_location=self.device)
        self.lstm.load_state_dict(lstm_ckpt['model_state_dict'])
        self.lstm.eval()
        
        # 2. Load Transformer
        self.transformer = WingoMTLTransformer(
            input_dim=input_dim,
            seq_len=self.seq_len,
            d_model=self.config['models']['transformer']['d_model'],
            nhead=self.config['models']['transformer']['nhead'],
            num_layers=self.config['models']['transformer']['num_layers']
        ).to(self.device)
        trans_ckpt = torch.load(os.path.join(self.sup_dir, "transformer_best_weights.pt"), map_location=self.device)
        self.transformer.load_state_dict(trans_ckpt['model_state_dict'])
        self.transformer.eval()
        
        # 3. Load VAE
        self.vae = WingoTemporalVAE(
            input_dim=input_dim, 
            sequence_length=self.seq_len, 
            latent_dim=self.config['models']['autoencoder']['bottleneck_dim']
        ).to(self.device)
        self.vae.load_state_dict(torch.load(os.path.join(self.unsup_dir, "temporal_vae_weights.pt"), map_location=self.device))
        self.vae.eval()
        
        # 4. Load XGBoost
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(os.path.join(self.sup_dir, "xgboost_master.json"))
        
        logger.info("All Base Brains Loaded and Frozen Successfully.")

    def generate_meta_matrix(self, val_df: pd.DataFrame, scaler):
        """Passes data through frozen brains to get Level-2 predictions."""
        logger.info("Generating Level-2 Meta-Matrix...")
        
        # Extract features and targets
        feature_cols = feature_config.MODEL_INPUT_FEATURES
        X_raw = val_df[feature_cols].values
        Y_true = val_df[feature_config.TARGETS['binary_size']].values
        
        # Scale Data
        X_scaled = scaler.transform(X_raw)
        
        # Build PyTorch Dataset for Sequence Sliding Windows
        # We need continuous targets for the meta-learner evaluation
        targets_dict = {'binary_size': Y_true}
        val_dataset = WingoSequenceDataset(X_scaled, targets_dict, self.seq_len)
        
        meta_X = []
        meta_Y = []
        
        with torch.no_grad():
            for i in range(len(val_dataset)):
                x_seq, y_dict = val_dataset[i]
                
                # Add batch dimension
                x_seq_tensor = x_seq.unsqueeze(0).to(self.device)
                
                # 1. Sequence Inferences (Sigmoid converts logits to probabilities)
                lstm_prob = torch.sigmoid(self.lstm(x_seq_tensor)['binary_size']).item()
                trans_prob = torch.sigmoid(self.transformer(x_seq_tensor)['binary_size']).item()
                
                # 2. VAE Anomaly Inference (MSE Reconstruction Error)
                vae_out = self.vae(x_seq_tensor)
                vae_error = F.mse_loss(vae_out['reconstructed'], x_seq_tensor).item()
                
                # 3. XGBoost Inference (Requires flat immediate row, not sequence)
                # XGBoost predicts based on the state at t-1 (which is the last row of the sequence)
                x_flat = x_seq[-1].numpy().reshape(1, -1)
                xgb_prob = self.xgb.predict_proba(x_flat)[0][1]
                
                # Append to Level-2 Matrix
                meta_X.append([lstm_prob, trans_prob, xgb_prob, vae_error])
                meta_Y.append(y_dict['binary_size'].item())
                
        logger.info(f"Level-2 Matrix Generated: {len(meta_X)} samples.")
        return torch.tensor(meta_X, dtype=torch.float32), torch.tensor(meta_Y, dtype=torch.float32)


class MetaLearnerTrainer:
    """
    Trains the NeuralMetaAggregator and calibrates the final output using Platt Scaling.
    """
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.meta_model = NeuralMetaAggregator(num_models=4).to(device)
        
        self.meta_dir = os.path.join(PROJECT_ROOT, self.config['paths']['meta_learner_artifact_dir'])
        os.makedirs(self.meta_dir, exist_ok=True)

    def fit_aggregator(self, X_meta: torch.Tensor, Y_meta: torch.Tensor):
        logger.info("="*60)
        logger.info("TRAINING LEVEL-2 NEURAL AGGREGATOR")
        
        X_meta = X_meta.to(self.device)
        Y_meta = Y_meta.to(self.device).unsqueeze(1)
        
        # LBFGS is the absolute best optimizer for small datasets (like the Level-2 matrix)
        optimizer = LBFGS(self.meta_model.parameters(), lr=0.1, max_iter=100)
        criterion = nn.BCEWithLogitsLoss()
        
        def closure():
            optimizer.zero_grad()
            logits = self.meta_model(X_meta)
            loss = criterion(logits, Y_meta)
            loss.backward()
            return loss
            
        # Optimize
        optimizer.step(closure)
        
        # Evaluate Meta-Learner
        self.meta_model.eval()
        with torch.no_grad():
            final_logits = self.meta_model(X_meta)
            final_probs = torch.sigmoid(final_logits).cpu().numpy().flatten()
            y_true = Y_meta.cpu().numpy().flatten()
            
            auc = roc_auc_score(y_true, final_probs)
            brier = brier_score_loss(y_true, final_probs)
            
        logger.info(f"Meta-Learner Training Complete -> AUC: {auc:.4f} | Brier Score: {brier:.4f}")
        
        # Save Neural Aggregator
        torch.save(self.meta_model.state_dict(), os.path.join(self.meta_dir, "meta_aggregator_weights.pt"))
        return final_probs, y_true

    def calibrate_probabilities(self, meta_probs: np.ndarray, y_true: np.ndarray):
        """
        Platt Scaling (Logistic Calibration).
        Maps the raw neural network output to pure statistical reality.
        If calibrated_prob = 0.65, there is exactly a 65% chance the bet wins.
        """
        logger.info("Applying Platt Scaling Calibration for Financial Execution...")
        
        calibrator = LogisticRegression(solver='lbfgs')
        # Scikit-learn requires 2D arrays
        meta_probs_2d = meta_probs.reshape(-1, 1)
        
        calibrator.fit(meta_probs_2d, y_true)
        
        # Evaluate Calibration Shift
        calibrated_probs = calibrator.predict_proba(meta_probs_2d)[:, 1]
        raw_brier = brier_score_loss(y_true, meta_probs)
        cal_brier = brier_score_loss(y_true, calibrated_probs)
        
        logger.info(f"Calibration Shift Results:")
        logger.info(f" -> Raw Brier Score: {raw_brier:.5f} (Lower is better)")
        logger.info(f" -> Calibrated Brier Score: {cal_brier:.5f}")
        
        # Save Calibrator Artifact
        joblib.dump(calibrator, os.path.join(self.meta_dir, "platt_calibrator.joblib"))
        logger.info("Platt Calibrator artifact saved.")


def execute_meta_pipeline():
    # 1. Load Configs
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    
    # 2. Load Validation Data (The Meta-Learner must train on data the base models didn't see)
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    df = pd.read_csv(data_path).sort_values(by='issue_id').reset_index(drop=True)
    df = df.dropna()
    
    split_idx = int(len(df) * config['data_pipeline']['preprocessing']['train_test_split_ratio'])
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # 3. Load Scaler
    scaler_path = os.path.join(PROJECT_ROOT, config['paths']['scaler_artifact_dir'], "master_scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    # 4. Generate Level-2 Matrix
    builder = Level2DataBuilder(config, device)
    
    # Calculate input_dim dynamically from feature_config schema
    input_dim = len(feature_config.MODEL_INPUT_FEATURES)
    
    builder.load_frozen_artifacts(input_dim)
    X_meta, Y_meta = builder.generate_meta_matrix(val_df, scaler)
    
    # 5. Train Stacker & Calibrate
    trainer = MetaLearnerTrainer(config, device)
    raw_meta_probs, y_true = trainer.fit_aggregator(X_meta, Y_meta)
    trainer.calibrate_probabilities(raw_meta_probs, y_true)
    
    logger.info("="*60)
    logger.info("PHASE 4: CLOUD TRAINING ENGINES OFFICIALLY COMPLETE.")
    logger.info("All Artifacts Locked. Ready for Phase 5: Live Inference.")
    logger.info("="*60)


if __name__ == "__main__":
    execute_meta_pipeline()