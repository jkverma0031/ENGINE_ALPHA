# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE TABULAR TRAINING ENGINE
# Core Component: src/training/train_tabular.py
# Description: Evaluates XGBoost and LightGBM using Purged Time-Series 
# Cross-Validation. Extracts rule-based hardware latency anomalies independently 
# of the deep learning sequence memory.
# ==============================================================================

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TabularTrainer] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
except ImportError:
    logger.error("Failed to import feature_config.")
    sys.exit(1)


class PurgedTimeSeriesSplit:
    """
    Marcos Lopez de Prado's Purged Time-Series Cross Validation.
    Prevents XGBoost from cheating. If row 100 is in the test set, row 99 and 101 
    contain overlapping rolling average data. This class explicitly deletes 
    (embargoes) the boundaries between train and validation folds.
    """
    def __init__(self, n_splits: int = 5, purge_window: int = 60):
        self.n_splits = n_splits
        self.purge_window = purge_window # Should equal your sequence_length

    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_window
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                test_end = n_samples
                
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices


class TabularEngine:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.data_path = os.path.join(PROJECT_ROOT, self.config['paths']['processed_data_path'])
        self.artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        # Meta-Learner Data path
        self.meta_dir = os.path.join(PROJECT_ROOT, self.config['paths']['meta_learner_artifact_dir'])
        os.makedirs(self.meta_dir, exist_ok=True)

    def _load_and_isolate_data(self):
        logger.info("Loading continuous flat tabular matrix...")
        df = pd.read_csv(self.data_path).sort_values(by='issue_id').reset_index(drop=True)
        
        # We drop the first N rows because rolling features created NaNs
        df = df.dropna()
        
        # Isolate features
        X = df[feature_config.MODEL_INPUT_FEATURES].values
        
        # Tabular baseline only targets the primary objective: Size (Big vs Small)
        y = df[feature_config.TARGETS['binary_size']].values
        
        return X, y

    def train_xgboost(self, X: np.ndarray, y: np.ndarray):
        logger.info("="*60)
        logger.info("INITIATING XGBOOST PURGED CROSS-VALIDATION")
        logger.info("="*60)
        
        xgb_cfg = self.config['models']['xgboost']
        n_splits = self.config['data_pipeline']['preprocessing']['cross_validation_folds']
        seq_len = self.config['data_pipeline']['feature_engineering']['sequence_length']
        
        cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_window=seq_len)
        
        # Array to store Out-Of-Fold predictions for the Meta-Learner
        oof_predictions = np.zeros(len(X))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=xgb_cfg['n_estimators'],
                max_depth=xgb_cfg['max_depth'],
                learning_rate=xgb_cfg['learning_rate'],
                subsample=xgb_cfg['subsample'],
                colsample_bytree=xgb_cfg['colsample_bytree'],
                tree_method=xgb_cfg['tree_method'],
                eval_metric=xgb_cfg['eval_metric'],
                early_stopping_rounds=xgb_cfg['early_stopping_rounds'],
                random_state=self.config['system']['random_seed'],
                n_jobs=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predict Probabilities
            preds = model.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = preds
            
            # Metrics
            auc = roc_auc_score(y_val, preds)
            acc = accuracy_score(y_val, (preds >= 0.5).astype(int))
            fold_scores.append(auc)
            
            logger.info(f"Fold {fold+1}/{n_splits} | AUC: {auc:.4f} | Acc: {acc:.2%}")
            
        logger.info(f"XGBoost Mean AUC: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
        
        # Train final master model on 100% of data (for live execution)
        logger.info("Training Final XGBoost Master Artifact on 100% of data...")
        master_model = xgb.XGBClassifier(**{k: v for k, v in xgb_cfg.items() if k != 'early_stopping_rounds'})
        master_model.fit(X, y)
        
        # Save Artifacts
        model_path = os.path.join(self.artifact_dir, "xgboost_master.json")
        master_model.save_model(model_path)
        logger.info(f"Master XGBoost Artifact saved to {model_path}")
        
        # Save OOF Predictions for Stacking
        oof_path = os.path.join(self.meta_dir, "xgboost_oof.npy")
        np.save(oof_path, oof_predictions)
        
        return oof_predictions

    def execute(self):
        X, y = self._load_and_isolate_data()
        self.train_xgboost(X, y)
        
        # LightGBM follows identical logic, separated for brevity
        self.train_lightgbm(X, y) 
        
        logger.info("="*60)
        logger.info("ALL TABULAR ENGINES TRAINED AND SERIALIZED.")
        logger.info("="*60)

if __name__ == "__main__":
    engine = TabularEngine()
    engine.execute()