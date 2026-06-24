# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE NEURAL META-LEARNER & CALIBRATOR
# Core Component: src/training/train_meta_learner.py
# Description: Generates Level-2 Out-of-Fold prediction matrices via Multi-GPU
# batched inference. Trains a Context-Aware Attention Aggregator to dynamically 
# shift trust between models based on real-time casino features (Volatility, 
# Latency), and applies Platt Scaling for absolute risk calibration.
# ==============================================================================

import os
import sys
import yaml
import time
import json
import logging
import gc
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import xgboost as xgb
from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MetaLearner] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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
    Context-Aware Squeeze-and-Excitation Aggregator.
    Takes the Base Probabilities AND the raw physical Casino Features (Context).
    Uses the Context to dynamically route trust and attention.
    """
    def __init__(self, num_models: int, context_dim: int):
        super(NeuralMetaAggregator, self).__init__()
        self.num_models = num_models
        self.context_dim = context_dim
        
        # Context Processing Network (Reads the environment)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.Mish()
        )
        
        # Attention Gate (Decides which model to trust based on the context)
        self.attention_gate = nn.Sequential(
            nn.Linear(32 + num_models, 32),
            nn.Mish(),
            nn.Linear(32, num_models)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, probs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        probs shape: (Batch, Num_Models)
        context shape: (Batch, Context_Dim)
        """
        # 1. Process the Casino Environment
        env_state = self.context_net(context)
        
        # 2. Combine Environment State with current Model Confidences
        gate_input = torch.cat([env_state, probs], dim=1)
        
        # 3. Generate dynamic voting weights
        gate_logits = self.attention_gate(gate_input)
        dynamic_weights = F.softmax(gate_logits / self.temperature, dim=1)
        
        # 4. Execute weighted consensus
        combined_ensemble = torch.sum(probs * dynamic_weights, dim=1, keepdim=True) + self.bias
        return combined_ensemble


class Level2DataBuilder:
    """
    Generates the massive Level-2 Matrix. 
    Handles deep sequential models and dynamic Tabular inference.
    (Unsupervised Autoencoder branch permanently bypassed).
    """
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        self.sup_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        self.tab_dir = os.path.join(PROJECT_ROOT, self.config['paths'].get('tabular_artifact_dir', 'artifacts/tabular'))
        self.meta_dir = os.path.join(PROJECT_ROOT, self.config['paths']['meta_learner_artifact_dir'])
        
        self.seq_len = self.config['data_pipeline']['feature_engineering']['sequence_length']
        self.gpu_count = torch.cuda.device_count()

    def load_frozen_artifacts(self, input_dim: int):
        logger.info("Initializing Frozen Artifact Vault (BYPASSING AUTOENCODER)...")
        
        # 1. Load Deep Sequence Models (The 52% AUC Base)
        self.lstm = WingoMTLLSTM(input_dim, self.config['models']['lstm']['hidden_dim'], self.config['models']['lstm']['num_layers']).to(self.device)
        self.lstm.load_state_dict(torch.load(os.path.join(self.sup_dir, "lstm_SWA_master.pt"), map_location=self.device)['model_state_dict'])
        self.lstm.eval()
        
        self.transformer = WingoMTLTransformer(input_dim, self.seq_len, self.config['models']['transformer']['d_model'], self.config['models']['transformer']['nhead'], self.config['models']['transformer']['num_layers']).to(self.device)
        self.transformer.load_state_dict(torch.load(os.path.join(self.sup_dir, "transformer_SWA_master.pt"), map_location=self.device)['model_state_dict'])
        self.transformer.eval()
        
        # 2. Load Tabular Ensembles
        logger.info("Loading Tabular Ensembles from Supervised Vault...")
        
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(os.path.join(self.sup_dir, "xgboost_master.json"))
        
        import lightgbm as lgb
        self.lgb = lgb.Booster(model_file=os.path.join(self.sup_dir, "lightgbm_master.txt"))
        
        self.use_cat = False
        try:
            import catboost as cb
            self.cat = cb.CatBoostClassifier()
            self.cat.load_model(os.path.join(self.sup_dir, "catboost_master.cbm"))
            self.use_cat = True
        except Exception:
            logger.warning("CatBoost master artifact not found. Proceeding with XGBoost and LightGBM.")
            pass
        
        # 3. Reconstruct SHAP Pruning Indices (Ensures 39-dim match)
        dropped_cols = [
            'prev_1_is_red', 'freq_size_big_last_20', 'time_second_sin', 
            'prev_1_is_green', 'prev_2_size_target', 'time_second_cos', 
            'lockout_ms', 'duration_ms', 'prev_1_size_target', 'latency_rolling_std_10'
        ]
        all_cols = list(feature_config.MODEL_INPUT_FEATURES)
        self.xgb_indices = [i for i, col in enumerate(all_cols) if col not in dropped_cols]
        
        if self.gpu_count > 1 and self.device.type == 'cuda':
            logger.info(f"Distributing Deep Models across {self.gpu_count} GPUs for Precomputation!")
            self.lstm = nn.DataParallel(self.lstm)
            self.transformer = nn.DataParallel(self.transformer)
            
        logger.info("All Available Brains Loaded Successfully.")

    def generate_meta_matrix(self, val_df: pd.DataFrame, master_scaler, tabular_oof_val=None):
        logger.info("Generating Level-2 Meta-Matrix (Probabilities + Raw Context)...")
        
        X_raw = val_df[list(feature_config.MODEL_INPUT_FEATURES)].values
        Y_true = val_df[feature_config.TARGETS['binary_size']].values
        X_scaled = master_scaler.transform(X_raw)
        
        val_dataset = WingoSequenceDataset(X_scaled, {'binary_size': Y_true}, self.seq_len)
        num_workers = self.config['system'].get('max_workers', 2)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=2048, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        probs_blocks = []
        context_blocks = []
        Y_blocks = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                
                # 1. Deep Learning Inference
                with autocast():
                    p_lstm = torch.sigmoid(self.lstm(batch_X)['binary_size']).view(-1, 1)
                    p_trans = torch.sigmoid(self.transformer(batch_X)['binary_size']).view(-1, 1)
                    
                # 2. Tabular Inference (Inverse scaling + SHAP subsetting)
                X_last_step_scaled = batch_X[:, -1, :].cpu().numpy()
                X_last_step_raw = master_scaler.inverse_transform(X_last_step_scaled)
                X_tabular_ready = X_last_step_raw[:, self.xgb_indices]
                
                p_xgb = self.xgb.predict_proba(X_tabular_ready)[:, 1]
                p_xgb_tensor = torch.tensor(p_xgb, dtype=torch.float32, device=self.device).view(-1, 1)
                
                # Handling LightGBM standard output
                p_lgb = self.lgb.predict(X_tabular_ready)
                p_lgb_tensor = torch.tensor(p_lgb, dtype=torch.float32, device=self.device).view(-1, 1)
                
                p_cat = self.cat.predict_proba(X_tabular_ready)[:, 1]
                p_cat_tensor = torch.tensor(p_cat, dtype=torch.float32, device=self.device).view(-1, 1)
                
                valid_tensors = [p_lstm, p_trans, p_xgb_tensor, p_lgb_tensor, p_cat_tensor]
                
                # 3. Combine Tensors 
                batch_probs = torch.cat(valid_tensors, dim=1)
                
                # Context is purely raw physical environment features (Autoencoder GMM bypassed)
                batch_context = batch_X[:, -1, :]
                
                probs_blocks.append(batch_probs.cpu())
                context_blocks.append(batch_context.cpu())
                Y_blocks.append(batch_Y['binary_size'].cpu())
                
        meta_Probs = torch.cat(probs_blocks, dim=0)
        meta_Context = torch.cat(context_blocks, dim=0)
        meta_Y = torch.cat(Y_blocks, dim=0)
        
        logger.info(f"Level-2 Matrix Generated! Probs Shape: {meta_Probs.shape} | Context Shape: {meta_Context.shape}")
        return meta_Probs, meta_Context, meta_Y
class MetaLearnerTrainer:
    def __init__(self, config: dict, device: torch.device, num_models: int, context_dim: int):
        self.config = config
        self.device = device
        self.meta_model = NeuralMetaAggregator(num_models=num_models, context_dim=context_dim).to(device)
        self.meta_dir = os.path.join(PROJECT_ROOT, self.config['paths']['meta_learner_artifact_dir'])
        os.makedirs(self.meta_dir, exist_ok=True)

    def fit_aggregator(self, Probs: torch.Tensor, Context: torch.Tensor, Y: torch.Tensor):
        logger.info("="*60)
        logger.info("TRAINING CONTEXT-AWARE ATTENTION AGGREGATOR")
        
        dataset = torch.utils.data.TensorDataset(Probs, Context, Y.unsqueeze(1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        
        # Deep network optimization
        optimizer = AdamW(self.meta_model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.BCEWithLogitsLoss()
        
        self.meta_model.train()
        for epoch in range(1, 51):
            epoch_loss = 0
            for b_probs, b_ctx, b_y in loader:
                b_probs, b_ctx, b_y = b_probs.to(self.device), b_ctx.to(self.device), b_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.meta_model(b_probs, b_ctx)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            scheduler.step()
            if epoch % 10 == 0:
                logger.info(f"Context-Aggregator Epoch [{epoch:02d}/50] | Loss: {epoch_loss/len(loader):.4f}")
        
        # Evaluate
        self.meta_model.eval()
        with torch.no_grad():
            final_logits = self.meta_model(Probs.to(self.device), Context.to(self.device))
            final_probs = torch.sigmoid(final_logits).cpu().numpy().flatten()
            y_true = Y.cpu().numpy().flatten()
            
            auc = roc_auc_score(y_true, final_probs)
            brier = brier_score_loss(y_true, final_probs)
            
        logger.info(f"Attention Training Complete -> Contextual AUC: {auc:.4f} | Brier: {brier:.4f}")
        torch.save(self.meta_model.state_dict(), os.path.join(self.meta_dir, "meta_aggregator_weights.pt"))
        
        return final_probs, y_true

    def calibrate_probabilities(self, meta_probs: np.ndarray, y_true: np.ndarray):
        logger.info("Applying Platt Scaling Calibration for Financial Execution...")
        
        calibrator = LogisticRegression(solver='lbfgs')
        meta_probs_2d = meta_probs.reshape(-1, 1)
        calibrator.fit(meta_probs_2d, y_true)
        
        calibrated_probs = calibrator.predict_proba(meta_probs_2d)[:, 1]
        raw_brier = brier_score_loss(y_true, meta_probs)
        cal_brier = brier_score_loss(y_true, calibrated_probs)
        
        logger.info(f"Calibration Shift Results: Raw Brier {raw_brier:.5f} -> Calibrated {cal_brier:.5f}")
        joblib.dump(calibrator, os.path.join(self.meta_dir, "platt_calibrator.joblib"))


def execute_meta_pipeline():
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    df = pd.read_csv(data_path).sort_values(by='issue_id').reset_index(drop=True)
    
    # Base Models trained on 0% -> 80%.
    # Meta-Learner trains on 80% -> 90%.
    base_split_idx = int(len(df) * config['data_pipeline']['preprocessing']['train_test_split_ratio'])
    meta_split_idx = int(len(df) * 0.90)
    
    val_df = df.iloc[base_split_idx:meta_split_idx].reset_index(drop=True)
    logger.info(f"Meta-Learner isolated to Temporal Slice: {base_split_idx} -> {meta_split_idx}")
    
    scaler_path = os.path.join(PROJECT_ROOT, config['paths']['scaler_artifact_dir'], "master_scaler.joblib")
    master_scaler = joblib.load(scaler_path)
    
    builder = Level2DataBuilder(config, device)
    input_dim = len(feature_config.MODEL_INPUT_FEATURES)
    
    builder.load_frozen_artifacts(input_dim)
    
    # 🚨 FRACTURE FIXED: Removed all references to stale tabular_oof_features.npy
    # Live inference prevents out-of-bounds slicing crashes.
    Probs, Context, Y_meta = builder.generate_meta_matrix(val_df, master_scaler, tabular_oof_val=None)
    
    actual_num_models = Probs.shape[1]
    context_dim = Context.shape[1]
    
    trainer = MetaLearnerTrainer(config, device, num_models=actual_num_models, context_dim=context_dim)
    raw_meta_probs, y_true = trainer.fit_aggregator(Probs, Context, Y_meta)
    trainer.calibrate_probabilities(raw_meta_probs, y_true)
    
    logger.info("="*60)
    logger.info("PHASE 4: CLOUD TRAINING ENGINES OFFICIALLY COMPLETE.")
    logger.info("="*60)

if __name__ == "__main__":
    execute_meta_pipeline()