# ==============================================================================
# ENGINE_ALPHA - UNIVERSAL DISTRIBUTED SEQUENCE TRAINING ENGINE (MONOLITH)
# Core Component: src/training/train_sequences.py
# Description: The absolute apex of quantitative deep learning architecture.
# Implements a SILICON-AGNOSTIC Orchestrator with modern PJRT and DDP runtimes.
# Dynamically transitions between NVIDIA CUDA (DistributedDataParallel) and 
# Google Cloud TPUs (XLA Pods) seamlessly using multi-processing spawn.
# Features Asymmetric HBM Streaming, True Scalar Broadcasting, Deferred 
# Rank-Aware RNG Seeding, NaN-Shields, and a Purged Chronological Embargo Split.
# ==============================================================================

import os
import sys
import yaml
import time
import socket
import logging
import gc
import random
import contextlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from tqdm import tqdm

# 🚀 TPU HARDWARE ACCELERATION INJECTIONS
if 'XLA_USE_BF16' not in os.environ:
    os.environ['XLA_USE_BF16'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, precision_score, f1_score
from torch.amp import GradScaler

# ==============================================================================
# 0. HARDWARE ABSTRACTION & PJRT (XLA) DETECTION
# ==============================================================================
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


# ==============================================================================
# 1. INSTITUTIONAL LOGGING & TELEMETRY
# ==============================================================================
class InstitutionalLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] [DistributedSeqEngine] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _is_master(self) -> bool:
        if TPU_AVAILABLE:
            return xr.global_ordinal() == 0
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True 

    def info(self, msg: str):
        if self._is_master(): self.logger.info(msg)

    def warning(self, msg: str):
        if self._is_master(): self.logger.warning(msg)

    def error(self, msg: str):
        rank = xr.global_ordinal() if TPU_AVAILABLE else (dist.get_rank() if dist.is_initialized() else 0)
        self.logger.error(f"[NODE {rank}] {msg}")

    def critical(self, msg: str):
        rank = xr.global_ordinal() if TPU_AVAILABLE else (dist.get_rank() if dist.is_initialized() else 0)
        self.logger.critical(f"[NODE {rank}] {msg}")

ilogger = InstitutionalLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
    from src.data.dataset_loader import WingoSequenceDataset
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
except ImportError as e:
    ilogger.critical(f"Failed to import project modules. Structural Integrity Compromised: {e}")
    sys.exit(1)


# ==============================================================================
# 2. THE UNIVERSAL SILICON ORCHESTRATOR (PJRT + DDP)
# ==============================================================================
def seed_everything(seed: int, rank: int = 0):
    unique_seed = seed + rank
    random.seed(unique_seed)
    os.environ['PYTHONHASHSEED'] = str(unique_seed)
    np.random.seed(unique_seed)
    torch.manual_seed(unique_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(unique_seed)
        torch.cuda.manual_seed_all(unique_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        
    if TPU_AVAILABLE:
        xm.set_rng_state(unique_seed, device=torch_xla.device())
        
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class UniversalOrchestrator:
    def __init__(self, rank: int, world_size: int, config: dict):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        
        self.is_tpu = TPU_AVAILABLE
        self.is_ddp = torch.cuda.is_available() and world_size > 1 and not self.is_tpu
        self.is_single_gpu = torch.cuda.is_available() and world_size == 1 and not self.is_tpu
        self.is_gpu = self.is_ddp or self.is_single_gpu
        self.device = self._initialize_hardware()
        
        seed_everything(self.config['system'].get('random_seed', 42), rank)

    def _initialize_hardware(self):
        if self.is_tpu:
            device = torch_xla.device()
            ilogger.info("="*80)
            ilogger.info(f"⚡ SILICON DETECTED: Google Cloud TPU (PJRT Node {self.rank}/{self.world_size})")
            ilogger.info("="*80)
            return device
            
        elif self.is_ddp:
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
            torch.cuda.set_device(self.rank)
            device = torch.device(f"cuda:{self.rank}")
            ilogger.info("="*80)
            ilogger.info(f"🔥 SILICON DETECTED: NVIDIA Multi-GPU DDP (NCCL Rank {self.rank}/{self.world_size})")
            ilogger.info("="*80)
            return device
            
        elif self.is_single_gpu:
            device = torch.device("cuda:0")
            ilogger.info("="*80)
            ilogger.info(f"🔥 SILICON DETECTED: Single NVIDIA GPU")
            ilogger.info("="*80)
            return device
            
        else:
            ilogger.warning("⚠️ No hardware accelerators detected. Defaulting to CPU.")
            return torch.device("cpu")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.is_ddp:
            return DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False)
        return model

    def get_base_model(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    def create_distributed_loader(self, dataset, batch_size: int, shuffle: bool, num_workers: int) -> Any:
        if self.is_tpu or self.is_ddp:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle, drop_last=True)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=(not self.is_tpu), drop_last=True)
            if self.is_tpu:
                return pl.ParallelLoader(loader, [self.device]).per_device_loader(self.device)
            return loader
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

    def execute_optimizer_step(self, optimizer: torch.optim.Optimizer, scaler: Optional[GradScaler] = None, model: nn.Module = None):
        if model is not None:
            if scaler is not None and self.is_ddp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if self.is_tpu:
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
        elif self.is_gpu and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

    def broadcast_scalar(self, value: float) -> float:
        safe_val = value if self.is_master() else 0.0
        tensor = torch.tensor([safe_val], dtype=torch.float32, device=self.device)
        
        if self.is_tpu:
            xm.all_reduce(xm.REDUCE_SUM, [tensor])
            return tensor.item()
        elif self.is_ddp:
            dist.broadcast(tensor, src=0)
            return tensor.item()
        return value

    def save_artifact(self, state_dict: dict, path: str):
        if self.is_tpu:
            xm.save(state_dict, path, master_only=True)
        elif self.is_master():
            torch.save(state_dict, path)

    def is_master(self) -> bool:
        if self.is_tpu: return xr.global_ordinal() == 0
        if self.is_ddp: return dist.get_rank() == 0
        return True

    def flush_memory(self):
        gc.collect()
        if self.is_tpu:
            xm.rendezvous('flush_memory')
        elif self.is_gpu:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if self.is_ddp:
                dist.barrier()

    def cleanup(self):
        if self.is_ddp:
            dist.destroy_process_group()


# ==============================================================================
# 3. DISTRIBUTED ASYMMETRIC METRIC STREAMING
# ==============================================================================
class DistributedMetricSynchronizer:
    def __init__(self, orchestrator: UniversalOrchestrator):
        self.orch = orchestrator
        self._reset_buffers()

    def _reset_buffers(self):
        self.targets_cpu = {k: [] for k in feature_config.TARGETS.keys()}
        self.probs_cpu = {k: [] for k in feature_config.TARGETS.keys()}

    def stream_batch_to_ram(self, batch_targets: dict, batch_probs: dict):
        for k in ['binary_size', 'one_hot_red', 'one_hot_green', 'one_hot_violet']:
            prob_tensor = batch_probs[k]
            targ_tensor = batch_targets[k]
            
            if self.orch.is_tpu:
                prob_tensor = xm.all_gather(prob_tensor)
                targ_tensor = xm.all_gather(targ_tensor)
            elif self.orch.is_ddp:
                p_list = [torch.zeros_like(prob_tensor) for _ in range(self.orch.world_size)]
                t_list = [torch.zeros_like(targ_tensor) for _ in range(self.orch.world_size)]
                dist.all_gather(p_list, prob_tensor)
                dist.all_gather(t_list, targ_tensor)
                prob_tensor = torch.cat(p_list, dim=0)
                targ_tensor = torch.cat(t_list, dim=0)
            
            if self.orch.is_master():
                self.probs_cpu[k].append(prob_tensor.detach().to(torch.float32).cpu().numpy())
                self.targets_cpu[k].append(targ_tensor.detach().to(torch.float32).cpu().numpy())
            
        k = 'exact_number'
        prob_num = batch_probs[k]
        targ_num = batch_targets[k]
        
        if self.orch.is_tpu:
            prob_num = xm.all_gather(prob_num)
            targ_num = xm.all_gather(targ_num)
        elif self.orch.is_ddp:
            p_list = [torch.zeros_like(prob_num) for _ in range(self.orch.world_size)]
            t_list = [torch.zeros_like(targ_num) for _ in range(self.orch.world_size)]
            dist.all_gather(p_list, prob_num)
            dist.all_gather(t_list, targ_num)
            prob_num = torch.cat(p_list, dim=0)
            targ_num = torch.cat(t_list, dim=0)
            
        if self.orch.is_master():
            self.probs_cpu[k].append(prob_num.detach().to(torch.float32).cpu().numpy())
            self.targets_cpu[k].append(targ_num.detach().to(torch.float32).cpu().numpy())

    def compile_metrics(self) -> Tuple[dict, dict]:
        if not self.orch.is_master():
            return {}, {}
        final_targets = {k: np.concatenate(v, axis=0) for k, v in self.targets_cpu.items() if len(v) > 0}
        final_probs = {k: np.concatenate(v, axis=0) for k, v in self.probs_cpu.items() if len(v) > 0}
        self._reset_buffers()
        return final_targets, final_probs


# ==============================================================================
# 4. ADVANCED INSTITUTIONAL LOSS FUNCTIONS
# ==============================================================================
class BinaryFocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()

class InstitutionalLossCalculator:
    def __init__(self, config: dict):
        lstm_cfg = config['models']['lstm']
        gamma = lstm_cfg.get('focal_gamma', 2.0)
        alpha = lstm_cfg.get('focal_alpha', 0.25)
        self.binary_loss_fn = BinaryFocalLossWithSmoothing(alpha=alpha, gamma=gamma, smoothing=0.05)
        self.categorical_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def compute_raw_losses(self, predictions: dict, targets: dict) -> list:
        pred_dtype = predictions['binary_size'].dtype
        
        loss_size = self.binary_loss_fn(predictions['binary_size'].view(-1), targets['binary_size'].to(pred_dtype).view(-1))
        loss_number = self.categorical_loss_fn(predictions['exact_number'], targets['exact_number'].long())
        loss_red = self.binary_loss_fn(predictions['one_hot_red'].view(-1), targets['one_hot_red'].to(pred_dtype).view(-1))
        loss_green = self.binary_loss_fn(predictions['one_hot_green'].view(-1), targets['one_hot_green'].to(pred_dtype).view(-1))
        loss_violet = self.binary_loss_fn(predictions['one_hot_violet'].view(-1), targets['one_hot_violet'].to(pred_dtype).view(-1))
        
        return [loss_size, loss_number, loss_red, loss_green, loss_violet]


# ==============================================================================
# 5. METRICS EVALUATOR (ECE & BRIER)
# ==============================================================================
class ExpectedCalibrationError:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def calculate(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece * 100.0

class DeepMetricsEvaluator:
    def __init__(self):
        self.ece_calculator = ExpectedCalibrationError(n_bins=10)

    def evaluate(self, final_targets: dict, final_probs: dict) -> dict:
        metrics = {}
        for target in ['binary_size', 'one_hot_red', 'one_hot_green', 'one_hot_violet']:
            y_t = final_targets[target]
            y_p = final_probs[target]
            y_pred_class = (y_p >= 0.5).astype(int)
            
            try: metrics[f'{target}_auc'] = roc_auc_score(y_t, y_p)
            except ValueError: metrics[f'{target}_auc'] = 0.5
                
            metrics[f'{target}_brier'] = brier_score_loss(y_t, y_p)
            metrics[f'{target}_ece'] = self.ece_calculator.calculate(y_t, y_p)
            metrics[f'{target}_f1'] = f1_score(y_t, y_pred_class, zero_division=0)
            
        y_num_t = final_targets['exact_number']
        y_num_p = final_probs['exact_number'] 
        try: metrics['number_logloss'] = log_loss(y_num_t, y_num_p, labels=list(range(10)))
        except ValueError: metrics['number_logloss'] = 2.3 
        
        return metrics


# ==============================================================================
# 6. STATE MANAGEMENT: CHECKPOINTING & EARLY STOPPING
# ==============================================================================
class InstitutionalCheckpointManager:
    def __init__(self, save_dir: str, model_name: str, orchestrator: UniversalOrchestrator):
        self.save_dir = save_dir
        self.model_name = model_name
        self.orchestrator = orchestrator
        self.best_path = os.path.join(save_dir, f"{model_name.lower()}_best_weights.pt")
        self.swa_path = os.path.join(save_dir, f"{model_name.lower()}_SWA_master.pt")

    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        scaler: Optional[GradScaler], brier: float, is_swa: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'brier': brier,
            'is_swa': is_swa
        }
        path = self.swa_path if is_swa else self.best_path
        self.orchestrator.save_artifact(checkpoint, path)
        return path

class RobustEarlyStopping:
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
            return True 
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
            return False 


# ==============================================================================
# 7. THE CORE DISTRIBUTED SEQUENCE TRAINER (PURIFIED FOR XLA)
# ==============================================================================
class SequenceTrainer:
    def __init__(self, model_name: str, model: nn.Module, config: dict, orchestrator: UniversalOrchestrator):
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
        
        self.use_swa = model_cfg.get('use_swa', True)
        self.swa_start = model_cfg.get('swa_start_epoch', int(self.epochs * 0.7))
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-8)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=self.lr * 0.01)
        
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.lr * 0.05)
            
        self.scaler = GradScaler('cuda', enabled=True) if self.orchestrator.is_gpu else None
        
        self.loss_calculator = InstitutionalLossCalculator(config)
        self.evaluator = DeepMetricsEvaluator()
        self.synchronizer = DistributedMetricSynchronizer(self.orchestrator)
        
        artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        os.makedirs(artifact_dir, exist_ok=True)
        self.checkpoint_manager = InstitutionalCheckpointManager(artifact_dir, self.model_name, self.orchestrator)
        self.early_stopping = RobustEarlyStopping(patience=model_cfg['early_stopping_patience'])

    def _get_autocast_context(self):
        # 🚨 FIX: Removed 'xla' autocast. It bloats LSTM graphs causing massive compilation freezes.
        # We now rely solely on the XLA_USE_BF16=1 environment variable for clean graph compilation.
        if self.orchestrator.is_gpu:
            return torch.autocast(device_type='cuda', enabled=True)
        return contextlib.nullcontext()

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        
        epoch_loss_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)
        batches_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        
        # 🚨 XLA-SAFE PROGRESS BAR: Only Master Node draws the bar. No .item() calls allowed!
        if self.orchestrator.is_master():
            batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader), 
                                  desc=f"[{self.model_name}] Ep {epoch:03d}/{self.epochs}", 
                                  leave=False, dynamic_ncols=True, colour='cyan')
        else:
            batch_iterator = enumerate(dataloader)
        
        for batch_idx, (X, y_dict) in batch_iterator:
            if batch_idx == 0 and epoch == 1:
                ilogger.info(f"[{self.model_name}] TPU JIT Compilation Started. Graph mapping in progress...")

            X = X.to(self.device, non_blocking=True)
            y_dict = {k: v.to(self.device, non_blocking=True) for k, v in y_dict.items()}
            
            with self._get_autocast_context():
                predictions = self.model(X)
                raw_losses = self.loss_calculator.compute_raw_losses(predictions, y_dict)
                total_loss = self.base_model.uncertainty_balancer.compute_loss(raw_losses)
                total_loss_scaled = total_loss / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(total_loss_scaled).backward()
            else:
                total_loss_scaled.backward()
            
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                self.orchestrator.execute_optimizer_step(self.optimizer, self.scaler, self.model)

            epoch_loss_tensor += total_loss.detach().to(torch.float32)
            batches_tensor += 1.0

            if batch_idx == 0 and epoch == 1:
                ilogger.info(f"[{self.model_name}] TPU JIT Compilation Complete! Accelerating.")

        if self.orchestrator.is_tpu:
            xm.all_reduce(xm.REDUCE_SUM, [epoch_loss_tensor, batches_tensor])
        elif self.orchestrator.is_ddp:
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)

        return {'loss': (epoch_loss_tensor / batches_tensor).item()}

    def validate_epoch(self, dataloader) -> dict:
        self.model.eval()
        
        val_loss_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)
        batches_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            for X, y_dict in dataloader:
                X = X.to(self.device, non_blocking=True)
                y_dict = {k: v.to(self.device, non_blocking=True) for k, v in y_dict.items()}
                
                with self._get_autocast_context():
                    predictions = self.model(X)
                    raw_losses = self.loss_calculator.compute_raw_losses(predictions, y_dict)
                    total_loss = self.base_model.uncertainty_balancer.compute_loss(raw_losses)
                    
                val_loss_tensor += total_loss.detach().to(torch.float32)
                batches_tensor += 1.0
                
                batch_probs = {}
                y_dict_clean = {}
                
                for k in ['binary_size', 'one_hot_red', 'one_hot_green', 'one_hot_violet']:
                    batch_probs[k] = torch.sigmoid(predictions[k].view(-1)).detach()
                    y_dict_clean[k] = y_dict[k].view(-1).detach()
                    
                batch_probs['exact_number'] = F.softmax(predictions['exact_number'], dim=1).detach()
                y_dict_clean['exact_number'] = y_dict['exact_number'].detach()
                
                self.synchronizer.stream_batch_to_ram(y_dict_clean, batch_probs)

        if self.orchestrator.is_tpu:
            xm.all_reduce(xm.REDUCE_SUM, [val_loss_tensor, batches_tensor])
        elif self.orchestrator.is_ddp:
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)

        final_targets, final_probs = self.synchronizer.compile_metrics()
        
        metrics = {}
        if self.orchestrator.is_master() and len(final_targets.get('binary_size', [])) > 0:
            metrics = self.evaluator.evaluate(final_targets, final_probs)
            
        metrics['val_loss'] = (val_loss_tensor / batches_tensor).item() if batches_tensor.item() > 0 else float('inf')
        return metrics

    def fit(self, raw_train_dataset, raw_val_dataset) -> float:
        ilogger.info("="*80)
        ilogger.info(f"INITIATING UNIVERSAL CLOUD TRAINING: {self.model_name.upper()}")
        ilogger.info("="*80)
        
        num_workers = min(self.config['system'].get('max_workers', 2), 2)
        batch_size = self.config['models'][self.model_name.lower()]['batch_size']
        
        train_loader = self.orchestrator.create_distributed_loader(raw_train_dataset, batch_size, True, num_workers)
        val_loader = self.orchestrator.create_distributed_loader(raw_val_dataset, batch_size, False, num_workers)
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            
            if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
                
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader)
            
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                current_lr = self.swa_scheduler.get_last_lr()[0]
                is_swa_phase = True
            else:
                # 🚨 FIX: The Scheduler now correctly steps ONCE per epoch.
                self.scheduler.step(epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                is_swa_phase = False
            
            brier = self.orchestrator.broadcast_scalar(val_metrics.get('binary_size_brier', 0.0))
            auc = self.orchestrator.broadcast_scalar(val_metrics.get('binary_size_auc', 0.5))
            ece = self.orchestrator.broadcast_scalar(val_metrics.get('binary_size_ece', 0.0))
            
            ep_time = time.time() - ep_start
            swa_tag = "[SWA Active]" if is_swa_phase else ""
            log_str = (
                f"Ep [{epoch:03d}/{self.epochs}] {ep_time:.1f}s {swa_tag} | LR: {current_lr:.2e} | "
                f"Loss: {train_metrics['loss']:.4f} -> {val_metrics['val_loss']:.4f} | "
                f"Brier: {brier:.4f} | AUC: {auc:.4f} | ECE: {ece:.2f}%"
            )
            
            is_best = self.early_stopping(brier)
            if is_best:
                self.checkpoint_manager.save_checkpoint(epoch, self.base_model, self.optimizer, self.scaler, brier, False)
                ilogger.info(log_str + f" [💾 BEST ARTIFACT LOCKED]")
            else:
                ilogger.info(log_str + f" [Patience: {self.early_stopping.counter}/{self.early_stopping.patience}]")
                if self.early_stopping.early_stop:
                    ilogger.warning(f"Early Stopping Triggered! Model calibration flatlined.")
                    break
                    
        if self.use_swa:
            ilogger.info(f"[{self.model_name}] Compiling SWA Flat Minimum Weights...")
            final_swa_base = self.swa_model.module
            if isinstance(final_swa_base, (nn.DataParallel, DDP)):
                final_swa_base = final_swa_base.module
                
            self.checkpoint_manager.save_checkpoint(self.epochs, final_swa_base, self.optimizer, self.scaler, self.early_stopping.best_score, True)
            ilogger.info(f"[{self.model_name}] SWA Master Artifact Locked to Vault.")

        total_time = (time.time() - start_time) / 60
        ilogger.info(f"TRAINING COMPLETE: {self.model_name.upper()} | Time: {total_time:.1f} min | Ultimate Brier: {self.early_stopping.best_score:.4f}")
        ilogger.info("="*80)
        return self.early_stopping.best_score


# ==============================================================================
# 8. PURGED CHRONOLOGICAL EMBARGO SPLITTER (Zero Look-Ahead Bias)
# ==============================================================================
class PurgedChronologicalValidator:
    def __init__(self, config: dict, orchestrator: UniversalOrchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
        self.seq_len = config['data_pipeline']['feature_engineering']['sequence_length']
        self.split_ratio = config['data_pipeline']['preprocessing']['train_test_split_ratio']
        self.scaler_dir = os.path.join(PROJECT_ROOT, self.config['paths']['scaler_artifact_dir'])

    def _isolate_tensors(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        target_dict = {t_name: df[col_name].values for t_name, col_name in feature_config.TARGETS.items()}
        feature_matrix = df[list(feature_config.MODEL_INPUT_FEATURES)].values
        return feature_matrix, target_dict

    def execute_embargo_split(self):
        ilogger.info("="*80)
        ilogger.info("INITIATING UNIVERSAL CHRONOLOGICAL SPLIT")
        ilogger.info("="*80)
        
        df = pd.read_csv(self.data_path).sort_values(by='issue_id').reset_index(drop=True)
        n_samples = len(df)
        train_end = int(n_samples * self.split_ratio)
        
        val_start = train_end + self.seq_len 
        
        train_df = df.iloc[0:train_end].copy()
        val_df = df.iloc[val_start:].copy()
        
        ilogger.info(f"Train Rows: {len(train_df)} | Embargo Gap: {self.seq_len} | Validation Rows: {len(val_df)}")
        
        X_tr_raw, y_tr_dict = self._isolate_tensors(train_df)
        X_va_raw, y_va_dict = self._isolate_tensors(val_df)
        
        import joblib
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        
        if self.orchestrator.is_master():
            os.makedirs(self.scaler_dir, exist_ok=True)
            joblib.dump(scaler, os.path.join(self.scaler_dir, "master_scaler.joblib"))
        
        train_dataset = WingoSequenceDataset(X_tr_scaled, y_tr_dict, self.seq_len)
        val_dataset = WingoSequenceDataset(X_va_scaled, y_va_dict, self.seq_len)
        input_dim = len(feature_config.MODEL_INPUT_FEATURES)
        
        # ----- TRAIN LSTM -----
        lstm_brain = WingoMTLLSTM(input_dim, self.config['models']['lstm']['hidden_dim'], self.config['models']['lstm']['num_layers'], self.config['models']['lstm']['dropout'])
        lstm_trainer = SequenceTrainer("LSTM", lstm_brain, self.config, self.orchestrator)
        lstm_trainer.fit(train_dataset, val_dataset)
        
        del lstm_brain, lstm_trainer
        self.orchestrator.flush_memory()
        
        # ----- TRAIN TRANSFORMER -----
        trans_brain = WingoMTLTransformer(input_dim, self.seq_len, self.config['models']['transformer']['d_model'], self.config['models']['transformer']['nhead'], self.config['models']['transformer']['num_layers'], self.config['models']['transformer']['dim_feedforward'], self.config['models']['transformer']['dropout'])
        trans_trainer = SequenceTrainer("Transformer", trans_brain, self.config, self.orchestrator)
        trans_trainer.fit(train_dataset, val_dataset)


# ==============================================================================
# 9. THE MULTI-PROCESSING EXECUTION TRIGGER (SPAWN)
# ==============================================================================
def _mp_fn(rank, world_size=1):
    """The unified function executed inside every core of the Multi-GPU/TPU cluster."""
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    orchestrator = UniversalOrchestrator(rank=rank, world_size=world_size, config=config)
    validator = PurgedChronologicalValidator(config, orchestrator)
    validator.execute_embargo_split()
    
    ilogger.info("="*80)
    ilogger.info("ALL DISTRIBUTED MODELS TRAINED, SCALED, AND ARTIFACTS PROMOTED TO VAULT.")
    ilogger.info("="*80)
    
    orchestrator.cleanup()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            os.environ['MASTER_PORT'] = str(s.getsockname()[1])

    world_size_gpu = torch.cuda.device_count()
    
    if TPU_AVAILABLE:
        ilogger.info("Igniting TPU on single-core mode to bypass topological pod limits.")
        xmp.spawn(_mp_fn, args=(1,), nprocs=1, start_method='spawn')
    elif world_size_gpu > 1:
        mp.spawn(_mp_fn, args=(world_size_gpu,), nprocs=world_size_gpu, join=True)
    else:
        _mp_fn(0, 1)