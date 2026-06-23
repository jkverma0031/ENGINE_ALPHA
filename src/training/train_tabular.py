# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE TABULAR ENSEMBLE ENGINE
# Core Component: src/training/train_tabular.py
# Description: Evaluates XGBoost, LightGBM, and CatBoost using advanced Purged 
# Time-Series Cross-Validation. Dynamically tunes hyperparameters via Optuna,
# drops noisy dimensions via SHAP Game Theory, and builds Out-Of-Fold (OOF) 
# Level-1 Stacking Matrices for the Deep Aggregator.
# ==============================================================================

import os
import sys
import yaml
import logging
import json
import gc
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Dict

# Machine Learning Libraries
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    import shap
except ImportError:
    print("[CRITICAL] Missing enterprise libraries. Run: pip install optuna shap")
    sys.exit(1)

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TabularEngine] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep Optuna logs clean

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import feature_config


class PurgedTimeSeriesSplit:
    """
    Advanced Institutional Cross-Validation.
    Standard K-Fold leaks future data into the past. Standard TimeSeriesSplit 
    leaks immediate adjacent momentum. This Purged split introduces a "gap" 
    between the train and validation sets, ensuring the model's momentum 
    indicators do not overlap and cheat.
    """
    def __init__(self, n_splits: int = 5, gap: int = 100):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X: np.ndarray):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            val_start = train_end + self.gap
            val_end = (i + 1) * fold_size
            
            if val_end > n_samples:
                val_end = n_samples
                
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            yield train_indices, val_indices


class TabularEngine:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_path = os.path.join(PROJECT_ROOT, self.config['paths']['processed_data_path'])
        self.model_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        self.meta_dir = os.path.join(PROJECT_ROOT, self.config['paths']['meta_learner_artifact_dir'])
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        
        self.cv_folds = self.config['data_pipeline']['preprocessing']['cross_validation_folds']
        self.feature_cols = list(feature_config.MODEL_INPUT_FEATURES)
        self.target_col = feature_config.TARGETS['binary_size']

    def _load_and_isolate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Mounting engineered data matrix from: {self.data_path}")
        if not os.path.exists(self.data_path):
            logger.critical("Data not found! Ensure feature_factory.py has run successfully.")
            sys.exit(1)
            
        df = pd.read_csv(self.data_path).sort_values(by='issue_id').reset_index(drop=True)
        
        # Verify schema
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing engineered features: {missing}")
            
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        
        logger.info(f"Data Loaded successfully. Available Rows: {len(df):,}")
        logger.info(f"Dynamic Feature Dimensions: {X.shape[1]}")
        return X, y

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Uses Optuna to dynamically search for the mathematically perfect 
        tree depth, learning rate, and subsample ratio for this exact dataset.
        """
        logger.info("="*60)
        logger.info("INITIATING OPTUNA HYPERPARAMETER OPTIMIZATION (XGBoost)")
        
        # We tune on a chronological 80/20 subset to save time
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X[:split_idx], y[:split_idx]
        X_va, y_va = X[split_idx:], y[split_idx:]
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_jobs': self.config['system']['max_workers']
            }
            
            model = xgb.XGBClassifier(**params)
            # Use early stopping to prevent overfitting during tuning
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            
            preds = model.predict_proba(X_va)[:, 1]
            return brier_score_loss(y_va, preds) # Minimize Brier Score for perfect calibration

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, timeout=1200) # Tune for max 20 minutes
        
        best_params = study.best_params
        best_params['tree_method'] = 'hist'
        best_params['objective'] = 'binary:logistic'
        best_params['n_jobs'] = self.config['system']['max_workers']
        
        logger.info(f"Optuna Optimization Complete! Best Brier Score: {study.best_value:.4f}")
        logger.info(f"Optimal Parameters Discovered: {json.dumps(best_params, indent=2)}")
        return best_params

    def prune_features_with_shap(self, model, X: np.ndarray, y: np.ndarray, feature_names: list) -> list:
        """
        Executes SHAP Game Theory pruning. 
        Calculates the marginal contribution of every feature. Drops the bottom 20% 
        that contribute only noise, forcing the trees to focus on true alpha.
        """
        logger.info("="*60)
        logger.info("EXECUTING SHAP GAME THEORY FEATURE PRUNING")
        
        # Calculate SHAP on a random 10,000 row subsample to keep it blazing fast
        sample_indices = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
        X_sample = X[sample_indices]
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Mean absolute SHAP value per feature across all samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Map back to feature names
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        # Drop the bottom 20%
        keep_count = int(len(feature_names) * 0.8)
        pruned_features = feature_importance['feature'].head(keep_count).tolist()
        dropped_features = feature_importance['feature'].tail(len(feature_names) - keep_count).tolist()
        
        logger.info(f"SHAP Analysis Complete. Pruning {len(dropped_features)} noisy dimensions.")
        logger.info(f"Dropped Features: {dropped_features}")
        
        # Save the list of remaining features so the Live Engine and Meta-Learner know what to pass
        with open(os.path.join(self.model_dir, "tabular_pruned_features.json"), "w") as f:
            json.dump({"pruned_features": pruned_features}, f, indent=4)
            
        return pruned_features

    def train_cross_validated_model(self, model_class, params: dict, X: np.ndarray, y: np.ndarray, name: str) -> Tuple[np.ndarray, object]:
        """
        Trains the model using Purged Time-Series splits and generates the Out-Of-Fold array.
        """
        logger.info(f"🚀 ENGINING {name.upper()} ARCHITECTURE")
        oof_preds = np.zeros(len(X))
        cv = PurgedTimeSeriesSplit(n_splits=self.cv_folds, gap=60) # 60-row gap prevents sequence leak
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            model = model_class(**params)
            
            if name == "LightGBM":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            elif name == "CatBoost":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50, verbose=False)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                
            preds = model.predict_proba(X_va)[:, 1]
            oof_preds[val_idx] = preds
            
            fold_auc = roc_auc_score(y_va, preds)
            logger.info(f"{name} Fold {fold+1}/{self.cv_folds} | Validation AUC: {fold_auc:.4f}")
            
            del model, X_tr, y_tr, X_va, y_va
            gc.collect()
            
        # Calculate OOF Metrics (ignoring the unpredicted rows at the very beginning)
        valid_idx = oof_preds > 0
        oof_auc = roc_auc_score(y[valid_idx], oof_preds[valid_idx])
        oof_brier = brier_score_loss(y[valid_idx], oof_preds[valid_idx])
        
        logger.info(f"[{name} OOF] AUC: {oof_auc:.4f} | Brier: {oof_brier:.4f}")
        
        # Train final production master artifact on 100% of data
        logger.info(f"Training Production {name} Master Artifact on 100% data...")
        master_model = model_class(**params)
        master_model.fit(X, y)
        
        return oof_preds, master_model

    def execute(self):
        logger.info("="*60)
        logger.info("INITIATING INSTITUTIONAL TABULAR ENSEMBLE STACK")
        logger.info("="*60)
        
        X_full, y_full = self._load_and_isolate_data()
        
        # 1. Base XGBoost to map SHAP values
        logger.info("Training Base XGBoost for structural analysis...")
        base_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=-1)
        base_xgb.fit(X_full, y_full)
        
        # 2. Prune Noise via SHAP
        pruned_feature_names = self.prune_features_with_shap(base_xgb, X_full, y_full, self.feature_cols)
        
        # Filter the full matrix down to only the mathematically relevant features
        pruned_indices = [self.feature_cols.index(f) for f in pruned_feature_names]
        X_pruned = X_full[:, pruned_indices]
        
        # 3. Dynamic Optuna Tuning on the pruned matrix
        optimal_xgb_params = self.optimize_hyperparameters(X_pruned, y_full)
        
        # 4. Train the Ultimate Master Models
        # XGBoost
        oof_xgb, master_xgb = self.train_cross_validated_model(xgb.XGBClassifier, optimal_xgb_params, X_pruned, y_full, "XGBoost")
        master_xgb.save_model(os.path.join(self.model_dir, "xgboost_master.json"))
        
        # LightGBM (Using standard params for diversity, but on pruned features)
        lgb_params = {
            'n_estimators': 1500, 'num_leaves': 31, 'learning_rate': 0.01, 
            'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': -1, 'verbose': -1
        }
        oof_lgb, master_lgb = self.train_cross_validated_model(lgb.LGBMClassifier, lgb_params, X_pruned, y_full, "LightGBM")
        master_lgb.booster_.save_model(os.path.join(self.model_dir, "lightgbm_master.txt"))
        
        # CatBoost
        oof_cat = np.zeros(len(y_full))
        if CATBOOST_AVAILABLE:
            cb_params = {
                'iterations': 1500, 'depth': 6, 'learning_rate': 0.02, 
                'eval_metric': 'Logloss', 'thread_count': -1, 'verbose': False
            }
            oof_cat, master_cat = self.train_cross_validated_model(cb.CatBoostClassifier, cb_params, X_pruned, y_full, "CatBoost")
            master_cat.save_model(os.path.join(self.model_dir, "catboost_master.cbm"))
            
        # 5. Compile Level-1 Meta-Matrix
        logger.info("="*60)
        logger.info("COMPILING LEVEL-1 META-MATRIX FOR DEEP AGGREGATOR")
        
        valid_matrices = [oof_xgb, oof_lgb]
        if CATBOOST_AVAILABLE: valid_matrices.append(oof_cat)
        
        meta_matrix = np.column_stack(valid_matrices)
        
        # Purge the unpredicted leading zeroes
        valid_mask = np.all(meta_matrix > 0, axis=1)
        clean_meta_matrix = meta_matrix[valid_mask]
        clean_y_true = y_full[valid_mask]
        
        np.save(os.path.join(self.meta_dir, "tabular_oof_features.npy"), clean_meta_matrix)
        np.save(os.path.join(self.meta_dir, "tabular_oof_targets.npy"), clean_y_true)
        
        logger.info(f"Level-1 Matrix Generated. Shape: {clean_meta_matrix.shape}")
        logger.info("TABULAR ENSEMBLE TRAINING PHASE COMPLETE.")
        logger.info("="*60)


if __name__ == "__main__":
    engine = TabularEngine()
    engine.execute()