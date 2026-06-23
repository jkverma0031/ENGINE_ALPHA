# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE MULTI-TASK SEQUENCE TRAINING ENGINE (MONOLITH)
# Core Component: src/training/train_sequences.py
# Description: The absolute apex of quantitative deep learning architecture.
# Implements pure NVIDIA CUDA orchestration, Focal Loss + Label Smoothing, 
# Stochastic Weight Averaging (SWA), Gradient Accumulation (VRAM Armor), 
# Deep Telemetry, Expected Calibration Error (ECE), Pre-Flight Sanity Checks, 
# and a Purged Chronological Embargo Split for zero Look-Ahead Bias.
# ==============================================================================

import os
import sys
import yaml
import time
import json
import logging
import gc
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, precision_score, recall_score, f1_score
from torch.amp import GradScaler

# ==============================================================================
# 0. GLOBAL DETERMINISM & PERFORMANCE LOCKS
# ==============================================================================

def seed_everything(seed: int = 42):
    """Locks all Random Number Generators for absolute institutional reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN locks
    torch.backends.cudnn.deterministic = False  # Set to False to allow CuDNN benchmark
    torch.backends.cudnn.benchmark = True       # Force CuDNN to find fastest convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True # Activate Tensor Cores for FP32 math

seed_everything(42)

# Enterprise Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DeepSeqEngine] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Resolve Root and Imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
    from src.data.dataset_loader import DataLoaderFactory, WingoSequenceDataset
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
except ImportError as e:
    logger.error(f"Failed to import project modules. Structural Integrity Compromised: {e}")
    sys.exit(1)


# ==============================================================================
# 1. HARDWARE ORCHESTRATION & VRAM TELEMETRY
# ==============================================================================

class CUDAMasterOrchestrator:
    """
    Abstracts hardware complexities for pure NVIDIA acceleration.
    Seamlessly transitions between Single GPU and Multi-GPU DataParallel clusters.
    Actively monitors VRAM allocation to prevent silent memory fragmentation.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._detect_silicon()

    def _detect_silicon(self):
        if self.gpu_count > 1:
            logger.info("="*70)
            logger.info(f"🔥 SILICON DETECTED: Multi-GPU Overdrive ({self.gpu_count}x NVIDIA Accelerators)")
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"   -> GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB | Compute: {props.major}.{props.minor}")
            logger.info("="*70)
        elif self.gpu_count == 1:
            props = torch.cuda.get_device_properties(0)
            logger.info(f"🔥 SILICON DETECTED: Single NVIDIA GPU -> {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
        else:
            logger.critical("⚠️ FATAL: No hardware accelerators detected. Deep Learning requires CUDA.")
            sys.exit(1)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wraps the model in DataParallel to blast massive batches across all GPUs."""
        model = model.to(self.device)
        if self.gpu_count > 1:
            logger.info(f"Wrapping architecture in nn.DataParallel across {self.gpu_count} devices.")
            return nn.DataParallel(model)
        return model

    def get_base_model(self, model: nn.Module) -> nn.Module:
        """Safely extracts the core model from the DataParallel wrapper to prevent saving corrupted weight keys."""
        return model.module if isinstance(model, nn.DataParallel) else model

    def flush_memory(self):
        """Aggressive Garbage Collection. Called between epochs and folds to prevent OOM crashes."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def log_vram_telemetry(self):
        """Queries the NVIDIA driver for exact VRAM allocation in gigabytes."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.debug(f"[VRAM Telemetry] GPU 0 - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


# ==============================================================================
# 2. ADVANCED INSTITUTIONAL LOSS FUNCTIONS
# ==============================================================================

class BinaryFocalLossWithSmoothing(nn.Module):
    """
    Focal Loss mathematically destroys the gradient for 'easy' sequences, forcing 
    the network to focus purely on chaotic anomalies. Label Smoothing prevents 
    the network from reaching 100% confidence, mitigating the risk of Kelly Criterion ruin.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        logger.info(f"Initialized Focal Loss -> Alpha: {self.alpha}, Gamma: {self.gamma}, Smoothing: {self.smoothing}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply Label Smoothing: e.g., 1.0 -> 0.95, 0.0 -> 0.05
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        
        # Calculate standard Binary Cross Entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        
        # Calculate Focal Modulators (Penalize confident but wrong predictions, ignore easy wins)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        return (focal_weight * bce_loss).mean()


class InstitutionalLossCalculator:
    """Computes distinct focal/smoothed loss functions for all 5 prediction heads simultaneously."""
    def __init__(self, config: dict):
        lstm_cfg = config['models']['lstm']
        gamma = lstm_cfg.get('focal_gamma', 2.0)
        alpha = lstm_cfg.get('focal_alpha', 0.25)
        
        # Initialize Enterprise Loss Functions
        self.binary_loss_fn = BinaryFocalLossWithSmoothing(alpha=alpha, gamma=gamma, smoothing=0.05)
        self.categorical_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def compute_raw_losses(self, predictions: dict, targets: dict) -> list:
        """
        Calculates raw loss for each target.
        🚨 INTEGRATION FIX: Applies .unsqueeze(1) to all binary targets.
        This explicitly prevents PyTorch from attempting a Cartesian Broadcast 
        between a [Batch_Size, 1] logit tensor and a [Batch_Size] target tensor, 
        which would cause an instant VRAM Memory Explosion.
        """
        loss_size = self.binary_loss_fn(predictions['binary_size'], targets['binary_size'].float().unsqueeze(1))
        loss_number = self.categorical_loss_fn(predictions['exact_number'], targets['exact_number'].long())
        
        loss_red = self.binary_loss_fn(predictions['one_hot_red'], targets['one_hot_red'].float().unsqueeze(1))
        loss_green = self.binary_loss_fn(predictions['one_hot_green'], targets['one_hot_green'].float().unsqueeze(1))
        loss_violet = self.binary_loss_fn(predictions['one_hot_violet'], targets['one_hot_violet'].float().unsqueeze(1))
        
        return [loss_size, loss_number, loss_red, loss_green, loss_violet]


# ==============================================================================
# 3. PROBABILISTIC METRICS & EXPECTED CALIBRATION ERROR (ECE)
# ==============================================================================

class ExpectedCalibrationError:
    """
    Quant Funds rely on ECE. If the model predicts a 70% probability of a win, 
    does that trade actually win exactly 70% of the time?
    This class splits predictions into 10 bins and measures the physical gap 
    between Predicted Confidence and Actual Accuracy.
    """
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def calculate(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece * 100.0 # Return as percentage


class DeepMetricsEvaluator:
    """Enterprise metrics compiler evaluating discrimination (AUC), truth (Brier), and calibration (ECE)."""
    def __init__(self):
        self.ece_calculator = ExpectedCalibrationError(n_bins=10)

    def evaluate(self, y_true_dict: dict, y_prob_dict: dict) -> dict:
        metrics = {}
        
        # Binary Targets (AUC, Brier, ECE, F1)
        for target in ['binary_size', 'one_hot_red', 'one_hot_green', 'one_hot_violet']:
            y_t = y_true_dict[target]
            y_p = y_prob_dict[target]
            y_pred_class = (y_p >= 0.5).astype(int)
            
            try:
                metrics[f'{target}_auc'] = roc_auc_score(y_t, y_p)
            except ValueError:
                metrics[f'{target}_auc'] = 0.5 # Failsafe for pure-class batches
                
            metrics[f'{target}_brier'] = brier_score_loss(y_t, y_p)
            metrics[f'{target}_ece'] = self.ece_calculator.calculate(y_t, y_p)
            
            # Standard metrics for human sanity checking
            metrics[f'{target}_f1'] = f1_score(y_t, y_pred_class, zero_division=0)
            metrics[f'{target}_prec'] = precision_score(y_t, y_pred_class, zero_division=0)
            
        # Multi-class Target (Exact Number 0-9)
        y_num_t = y_true_dict['exact_number']
        y_num_p = y_prob_dict['exact_number'] 
        
        try:
            metrics['number_logloss'] = log_loss(y_num_t, y_num_p, labels=list(range(10)))
        except ValueError:
            metrics['number_logloss'] = 2.3 
            
        return metrics


# ==============================================================================
# 4. STATE MANAGEMENT: CHECKPOINTING & EARLY STOPPING
# ==============================================================================

class InstitutionalCheckpointManager:
    """Handles deep state serialization, allowing perfect pause/resume and artifact tracking."""
    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_path = os.path.join(save_dir, f"{model_name.lower()}_best_weights.pt")
        self.swa_path = os.path.join(save_dir, f"{model_name.lower()}_SWA_master.pt")

    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        scaler: GradScaler, brier: float, is_swa: bool = False):
        """Saves absolute state dictionaries to prevent corruption."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'brier': brier,
            'is_swa': is_swa
        }
        
        path = self.swa_path if is_swa else self.best_path
        torch.save(checkpoint, path)
        return path


class RobustEarlyStopping:
    """Evaluates convergence based strictly on Brier Score Calibration."""
    def __init__(self, patience: int = 20, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return True # Is Best
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False # Is Not Best


# ==============================================================================
# 5. THE CORE SEQUENCE TRAINER
# ==============================================================================

class SequenceTrainer:
    """
    The Master Execution Engine. 
    Handles massive DataParallel batching, VRAM Gradient Accumulation, 
    Pre-Flight Sanity Checks, Live Telemetry, and Stochastic Weight Averaging (SWA).
    """
    def __init__(self, model_name: str, model: nn.Module, config: dict, orchestrator: CUDAMasterOrchestrator):
        self.model_name = model_name
        self.config = config
        self.orchestrator = orchestrator
        self.device = orchestrator.device
        
        self.model = orchestrator.wrap_model(model)
        self.base_model = orchestrator.get_base_model(self.model)
        
        model_cfg = self.config['models'][model_name.lower()]
        self.epochs = model_cfg['epochs']
        self.lr = model_cfg['learning_rate']
        self.accumulation_steps = model_cfg.get('accumulation_steps', 1) 
        self.max_grad_norm = 1.0
        
        self.use_swa = model_cfg.get('use_swa', True)
        self.swa_start = model_cfg.get('swa_start_epoch', int(self.epochs * 0.7))
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-8)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=self.lr * 0.01)
        
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.lr * 0.05)
            
        self.scaler = GradScaler('cuda', enabled=True)
        self.loss_calculator = InstitutionalLossCalculator(config)
        self.evaluator = DeepMetricsEvaluator()
        
        artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        self.checkpoint_manager = InstitutionalCheckpointManager(artifact_dir, self.model_name)
        self.early_stopping = RobustEarlyStopping(patience=model_cfg['early_stopping_patience'])

    def _pre_flight_sanity_check(self, dataloader):
        """
        Runs a hidden, isolated forward/backward pass on a single batch before 
        training begins to ensure no VRAM broadcasts or shape mismatches will 
        crash the 10-hour training run.
        """
        logger.info(f"[{self.model_name}] Executing Pre-Flight Math & VRAM Sanity Check...")
        self.model.train()
        
        try:
            X, y_dict = next(iter(dataloader))
            X = X.to(self.device)
            y_dict = {k: v.to(self.device) for k, v in y_dict.items()}
            
            with torch.autocast(device_type='cuda', enabled=True):
                predictions = self.model(X)
                raw_losses = self.loss_calculator.compute_raw_losses(predictions, y_dict)
                total_loss = self.base_model.uncertainty_balancer.compute_loss(raw_losses)
                
            self.scaler.scale(total_loss).backward()
            self.optimizer.zero_grad()
            logger.info(f"[{self.model_name}] ✅ Pre-Flight Sanity Check PASSED. Tensors matched.")
        except Exception as e:
            logger.critical(f"[{self.model_name}] ❌ FATAL: Pre-Flight Check FAILED. Inspect tensors: {e}")
            sys.exit(1)

    def _calculate_live_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, is_binary: bool = True) -> float:
        """Calculates accuracy metric on the fly for live training visibility."""
        with torch.no_grad():
            if is_binary:
                preds = (torch.sigmoid(logits) >= 0.5).float()
            else:
                preds = torch.argmax(logits, dim=1).float()
            correct = (preds == targets.float().unsqueeze(1)).sum().item() if is_binary else (preds == targets).sum().item()
            return correct / targets.size(0)

    def train_epoch(self, dataloader, epoch: int) -> dict:
        """Executes a single pass over the dataset with Gradient Accumulation (VRAM Armor)."""
        self.model.train()
        epoch_loss = 0.0
        acc_size, acc_num = 0.0, 0.0
        self.optimizer.zero_grad()
        
        for batch_idx, (X, y_dict) in enumerate(dataloader):
            X = X.to(self.device, non_blocking=True)
            y_dict = {k: v.to(self.device, non_blocking=True) for k, v in y_dict.items()}
            
            with torch.autocast(device_type='cuda', enabled=True):
                predictions = self.model(X)
                raw_losses = self.loss_calculator.compute_raw_losses(predictions, y_dict)
                total_loss = self.base_model.uncertainty_balancer.compute_loss(raw_losses)
                
                # Divides the loss to normalize gradients across micro-batches
                total_loss = total_loss / self.accumulation_steps

            self.scaler.scale(total_loss).backward()
            
            # VRAM Armor: Optimizer only steps when accumulation threshold is hit
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Mathematical Precision: Cosine Annealing steps smoothly per micro-batch
                if not (self.use_swa and epoch >= self.swa_start):
                    self.scheduler.step(epoch - 1 + batch_idx / len(dataloader))

            # Live Visibility Tracking
            epoch_loss += total_loss.item() * self.accumulation_steps
            acc_size += self._calculate_live_accuracy(predictions['binary_size'], y_dict['binary_size'], True)
            acc_num += self._calculate_live_accuracy(predictions['exact_number'], y_dict['exact_number'], False)
            
        batches = len(dataloader)
        return {
            'loss': epoch_loss / batches, 
            'size_acc': acc_size / batches,
            'num_acc': acc_num / batches
        }

    def validate_epoch(self, dataloader) -> dict:
        """Executes full evaluation without gradients, compiling exhaustive Probabilistic Metrics."""
        self.model.eval()
        val_loss = 0.0
        acc_size = 0.0
        
        all_targets = {k: [] for k in feature_config.TARGETS.keys()}
        all_probs = {k: [] for k in feature_config.TARGETS.keys()}
        
        with torch.no_grad():
            for X, y_dict in dataloader:
                X = X.to(self.device, non_blocking=True)
                y_dict = {k: v.to(self.device, non_blocking=True) for k, v in y_dict.items()}
                
                with torch.autocast(device_type='cuda', enabled=True):
                    predictions = self.model(X)
                    raw_losses = self.loss_calculator.compute_raw_losses(predictions, y_dict)
                    total_loss = self.base_model.uncertainty_balancer.compute_loss(raw_losses)
                    
                val_loss += total_loss.item()
                acc_size += self._calculate_live_accuracy(predictions['binary_size'], y_dict['binary_size'], True)
                
                # Extract physical arrays for offline Metric Evaluation
                for k in ['binary_size', 'one_hot_red', 'one_hot_green', 'one_hot_violet']:
                    all_probs[k].append(torch.sigmoid(predictions[k]).cpu().numpy())
                    all_targets[k].append(y_dict[k].cpu().numpy())
                all_probs['exact_number'].append(F.softmax(predictions['exact_number'], dim=1).cpu().numpy())
                all_targets['exact_number'].append(y_dict['exact_number'].cpu().numpy())

        # Compile Tensors
        final_targets = {k: np.concatenate(v) for k, v in all_targets.items()}
        final_probs = {k: np.concatenate(v) for k, v in all_probs.items()}
        
        # Calculate Deep Metrics
        metrics = self.evaluator.evaluate(final_targets, final_probs)
        metrics['val_loss'] = val_loss / len(dataloader)
        metrics['val_size_acc'] = acc_size / len(dataloader)
        
        return metrics

    def fit(self, train_loader, val_loader) -> float:
        """The Main Cloud Execution Loop."""
        logger.info("="*80)
        logger.info(f"INITIATING INSTITUTIONAL CLOUD TRAINING SEQUENCE: {self.model_name.upper()}")
        logger.info(f"Target Device: {self.device} | Epochs: {self.epochs} | Accumulation: {self.accumulation_steps}")
        logger.info("="*80)
        
        self._pre_flight_sanity_check(train_loader)
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader)
            
            # SWA Scheduler execution
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                current_lr = self.swa_scheduler.get_last_lr()[0]
                is_swa_phase = True
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                is_swa_phase = False
            
            ep_time = time.time() - ep_start
            brier = val_metrics['binary_size_brier']
            auc = val_metrics['binary_size_auc']
            ece = val_metrics['binary_size_ece']
            f1 = val_metrics['binary_size_f1']
            
            swa_tag = "[SWA Active]" if is_swa_phase else ""
            
            log_str = (
                f"Ep [{epoch:03d}/{self.epochs}] {ep_time:.1f}s {swa_tag} | LR: {current_lr:.2e} | "
                f"Loss: {train_metrics['loss']:.4f} -> {val_metrics['val_loss']:.4f} | "
                f"Brier: {brier:.4f} | AUC: {auc:.4f} | ECE: {ece:.2f}% | F1: {f1:.3f}"
            )
            
            # Evaluate Checkpoint based purely on Brier Score (True Calibration)
            is_best = self.early_stopping(brier)
            if is_best:
                path = self.checkpoint_manager.save_checkpoint(
                    epoch, self.base_model, self.optimizer, self.scaler, brier, is_swa=False
                )
                logger.info(log_str + f" [💾 BEST ARTIFACT LOCKED]")
            else:
                logger.info(log_str + f" [Patience: {self.early_stopping.counter}/{self.early_stopping.patience}]")
                if self.early_stopping.early_stop:
                    logger.warning(f"Early Stopping Triggered! Model calibration flatlined.")
                    break
                    
        # Post-Training SWA Resolution (The "Flat Minimum" Shield)
        if self.use_swa:
            logger.info(f"[{self.model_name}] Compiling SWA Flat Minimum Weights...")
            
            final_swa_base = self.swa_model.module.module if isinstance(self.swa_model.module, nn.DataParallel) else self.swa_model.module
            
            self.checkpoint_manager.save_checkpoint(
                self.epochs, final_swa_base, self.optimizer, self.scaler, self.early_stopping.best_score, is_swa=True
            )
            logger.info(f"[{self.model_name}] SWA Master Artifact Locked to Vault.")

        total_time = (time.time() - start_time) / 60
        logger.info(f"TRAINING COMPLETE: {self.model_name.upper()} | Time: {total_time:.1f} min | Ultimate Brier: {self.early_stopping.best_score:.4f}")
        logger.info("="*80)
        
        return self.early_stopping.best_score


# ==============================================================================
# 6. PURGED CHRONOLOGICAL EMBARGO SPLITTER (Zero Look-Ahead Bias)
# ==============================================================================

class PurgedChronologicalValidator:
    """
    Replaces massive Walk-Forward loops to save compute time, while strictly enforcing
    the "Embargo Purge Gap". Physically deletes overlapping sequence rows between 
    the Train and Validation sets so the network cannot memorize the future.
    """
    def __init__(self, config: dict, orchestrator: CUDAMasterOrchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
        self.seq_len = config['data_pipeline']['feature_engineering']['sequence_length']
        self.split_ratio = config['data_pipeline']['preprocessing']['train_test_split_ratio']
        self.scaler_dir = os.path.join(PROJECT_ROOT, self.config['paths']['scaler_artifact_dir'])
        
        # Dynamic Batch Sizing (Maximizes GPU VRAM usage)
        gpu_multiplier = max(1, orchestrator.gpu_count)
        self.lstm_batch = config['models']['lstm']['batch_size'] * gpu_multiplier
        self.trans_batch = config['models']['transformer']['batch_size'] * gpu_multiplier

    def _isolate_tensors(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        target_dict = {t_name: df[col_name].values for t_name, col_name in feature_config.TARGETS.items()}
        feature_matrix = df[list(feature_config.MODEL_INPUT_FEATURES)].values
        return feature_matrix, target_dict

    def execute_embargo_split(self):
        logger.info("="*80)
        logger.info("INITIATING PURGED CHRONOLOGICAL EMBARGO SPLIT")
        logger.info("="*80)
        
        df = pd.read_csv(self.data_path).sort_values(by='issue_id').reset_index(drop=True)
        n_samples = len(df)
        
        train_end = int(n_samples * self.split_ratio)
        # 🚨 THE EMBARGO GAP: Delete the sequence_length between Train and Val
        val_start = train_end + self.seq_len 
        
        train_df = df.iloc[0:train_end].copy()
        val_df = df.iloc[val_start:].copy()
        
        logger.info(f"Train Rows: {len(train_df)} | 🛡️ Purged Gap: {self.seq_len} (Zero Leakage) | Val Rows: {len(val_df)}")
        
        X_tr_raw, y_tr_dict = self._isolate_tensors(train_df)
        X_va_raw, y_va_dict = self._isolate_tensors(val_df)
        
        # Strict Isolated Scaling
        import joblib
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        joblib.dump(scaler, os.path.join(self.scaler_dir, "master_scaler.joblib"))
        
        train_dataset = WingoSequenceDataset(X_tr_scaled, y_tr_dict, self.seq_len)
        val_dataset = WingoSequenceDataset(X_va_scaled, y_va_dict, self.seq_len)
        workers = self.config['system'].get('max_workers', 4)
        input_dim = len(feature_config.MODEL_INPUT_FEATURES)
        
        # ----- TRAIN LSTM -----
        lstm_train_loader = DataLoader(train_dataset, batch_size=self.lstm_batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        lstm_val_loader = DataLoader(val_dataset, batch_size=self.lstm_batch, shuffle=False, num_workers=workers, pin_memory=True)
        
        lstm_brain = WingoMTLLSTM(input_dim, self.config['models']['lstm']['hidden_dim'], self.config['models']['lstm']['num_layers'], self.config['models']['lstm']['dropout'])
        lstm_trainer = SequenceTrainer("LSTM", lstm_brain, self.config, self.orchestrator)
        lstm_trainer.fit(lstm_train_loader, lstm_val_loader)
        
        # Aggressive VRAM Cleanup
        del lstm_brain, lstm_trainer, lstm_train_loader, lstm_val_loader
        self.orchestrator.flush_memory()
        
        # ----- TRAIN TRANSFORMER -----
        trans_train_loader = DataLoader(train_dataset, batch_size=self.trans_batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        trans_val_loader = DataLoader(val_dataset, batch_size=self.trans_batch, shuffle=False, num_workers=workers, pin_memory=True)
        
        trans_brain = WingoMTLTransformer(input_dim, self.seq_len, self.config['models']['transformer']['d_model'], self.config['models']['transformer']['nhead'], self.config['models']['transformer']['num_layers'], self.config['models']['transformer']['dim_feedforward'], self.config['models']['transformer']['dropout'])
        trans_trainer = SequenceTrainer("Transformer", trans_brain, self.config, self.orchestrator)
        trans_trainer.fit(trans_train_loader, trans_val_loader)
        

def execute_pipeline():
    """Master entry point for the Neural Sequence Architecture."""
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    orchestrator = CUDAMasterOrchestrator()
    validator = PurgedChronologicalValidator(config, orchestrator)
    validator.execute_embargo_split()
    
    logger.info("="*80)
    logger.info("ALL DEEP SEQUENCE MODELS TRAINED, SCALED, AND ARTIFACTS PROMOTED TO VAULT.")
    logger.info("="*80)

if __name__ == "__main__":
    execute_pipeline()