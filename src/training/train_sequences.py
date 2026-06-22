# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE MULTI-TASK SEQUENCE TRAINING ENGINE
# Core Component: src/training/train_sequences.py
# Description: High-performance cloud training loop for LSTM and Transformer.
# Implements AMP (Mixed Precision), Gradient Accumulation, Multi-Task Loss 
# Balancing, and Cosine Annealing Learning Rate scheduling.
# ==============================================================================

import os
import sys
import yaml
import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SequenceTrainer] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Resolve Root and Imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.data.dataset_loader import DataLoaderFactory
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


class MultiTaskLossCalculator:
    """
    Computes distinct loss functions for all 5 prediction heads simultaneously.
    """
    @staticmethod
    def compute_raw_losses(predictions: dict, targets: dict) -> list:
        """
        Calculates the raw loss for each target before the Uncertainty Balancer weights them.
        Returns a list: [loss_size, loss_number, loss_red, loss_green, loss_violet]
        """
        # 1. Binary Size (BCE with Logits)
        loss_size = F.binary_cross_entropy_with_logits(
            predictions['binary_size'], targets['binary_size'].float()
        )
        
        # 2. Exact Number (Cross Entropy for Multi-Class)
        # Note: CrossEntropyLoss expects logits of shape (Batch, Classes) and targets of shape (Batch,)
        loss_number = F.cross_entropy(
            predictions['exact_number'], targets['exact_number'].long()
        )
        
        # 3. Colors (BCE with Logits)
        loss_red = F.binary_cross_entropy_with_logits(
            predictions['one_hot_red'], targets['one_hot_red'].float()
        )
        loss_green = F.binary_cross_entropy_with_logits(
            predictions['one_hot_green'], targets['one_hot_green'].float()
        )
        loss_violet = F.binary_cross_entropy_with_logits(
            predictions['one_hot_violet'], targets['one_hot_violet'].float()
        )
        
        return [loss_size, loss_number, loss_red, loss_green, loss_violet]


class SequenceTrainer:
    """
    The Orchestrator. Handles the rigorous mathematics of backpropagation 
    and artifact persistence.
    """
    def __init__(self, model_name: str, model: nn.Module, config: dict, device: torch.device):
        self.model_name = model_name
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Extract hyperparameters
        model_cfg = self.config['models'][model_name.lower()]
        self.epochs = model_cfg['epochs']
        self.lr = model_cfg['learning_rate']
        self.patience = model_cfg['early_stopping_patience']
        
        # Institutional Optimizers
        # AdamW separates weight decay from the gradient update, crucial for Transformers
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=1e-4, 
            eps=1e-8
        )
        
        # Cosine Annealing bounces the learning rate to escape local minima
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=self.lr * 0.01
        )
        
        # Automatic Mixed Precision Scaler
        self.scaler = GradScaler()
        
        # Artifact Paths
        self.artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.save_path = os.path.join(self.artifact_dir, f"{model_name.lower()}_best_weights.pt")
        
        # Gradient Accumulation (Hardcoded to 1 for standard, scale up if Kaggle VRAM dies)
        self.accumulation_steps = 1
        self.max_grad_norm = 1.0

    def _calculate_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, is_binary: bool = True):
        """Calculates accuracy metric on the fly for logging."""
        with torch.no_grad():
            if is_binary:
                preds = (torch.sigmoid(logits) >= 0.5).float()
            else:
                preds = torch.argmax(logits, dim=1).float()
            correct = (preds == targets).sum().item()
            return correct / targets.size(0)

    def train_epoch(self, dataloader) -> dict:
        self.model.train()
        epoch_loss = 0.0
        acc_size, acc_num = 0.0, 0.0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (X, y_dict) in enumerate(dataloader):
            # Move data to GPU
            X = X.to(self.device, non_blocking=True)
            y_dict = {k: v.to(self.device, non_blocking=True) for k, v in y_dict.items()}
            
            # --- 1. Automatic Mixed Precision (AMP) Forward Pass ---
            with autocast():
                # Forward
                predictions = self.model(X)
                
                # Calculate the 5 individual raw losses
                raw_losses = MultiTaskLossCalculator.compute_raw_losses(predictions, y_dict)
                
                # Pass to the uncertainty balancer built into the models
                # It mathematically scales them based on task difficulty
                total_loss = self.model.uncertainty_balancer.compute_loss(raw_losses)
                
                # Normalize loss for gradient accumulation
                total_loss = total_loss / self.accumulation_steps

            # --- 2. Backward Pass & Scaling ---
            self.scaler.scale(total_loss).backward()
            
            # --- 3. Optimization Step ---
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                # Gradient Clipping prevents mathematically impossible leaps in deep networks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Step optimizer and update scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step(batch_idx / len(dataloader)) # Update LR per batch

            # Metrics gathering
            epoch_loss += total_loss.item() * self.accumulation_steps
            acc_size += self._calculate_accuracy(predictions['binary_size'], y_dict['binary_size'], True)
            acc_num += self._calculate_accuracy(predictions['exact_number'], y_dict['exact_number'], False)
            
        batches = len(dataloader)
        return {
            'loss': epoch_loss / batches,
            'size_acc': acc_size / batches,
            'num_acc': acc_num / batches
        }

    def validate_epoch(self, dataloader) -> dict:
        """Runs validation without calculating gradients to save memory and time."""
        self.model.eval()
        val_loss = 0.0
        acc_size, acc_num = 0.0, 0.0
        
        with torch.no_grad():
            for X, y_dict in dataloader:
                X = X.to(self.device, non_blocking=True)
                y_dict = {k: v.to(self.device, non_blocking=True) for k, v in y_dict.items()}
                
                with autocast():
                    predictions = self.model(X)
                    raw_losses = MultiTaskLossCalculator.compute_raw_losses(predictions, y_dict)
                    total_loss = self.model.uncertainty_balancer.compute_loss(raw_losses)
                    
                val_loss += total_loss.item()
                acc_size += self._calculate_accuracy(predictions['binary_size'], y_dict['binary_size'], True)
                acc_num += self._calculate_accuracy(predictions['exact_number'], y_dict['exact_number'], False)
                
        batches = len(dataloader)
        return {
            'val_loss': val_loss / batches,
            'val_size_acc': acc_size / batches,
            'val_num_acc': acc_num / batches
        }

    def fit(self, train_loader, val_loader):
        """The Master Execution Loop."""
        logger.info("="*60)
        logger.info(f"INITIATING CLOUD TRAINING SEQUENCE: {self.model_name.upper()}")
        logger.info(f"Target Device: {self.device} | Epochs: {self.epochs} | AMP: Enabled")
        logger.info("="*60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = []
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            
            # Train and Validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            ep_time = time.time() - ep_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Compile Logs
            log_str = (
                f"Epoch [{epoch:03d}/{self.epochs:03d}] - {ep_time:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Size Acc: {val_metrics['val_size_acc']:.1%} | "
                f"Num Acc: {val_metrics['val_num_acc']:.1%}"
            )
            
            # Checkpointing & Early Stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # Save the model state dict (Enterprise best practice)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, self.save_path)
                
                logger.info(log_str + " [💾 NEW BEST ARTIFACT SAVED]")
            else:
                patience_counter += 1
                logger.info(log_str + f" [Patience: {patience_counter}/{self.patience}]")
                
                if patience_counter >= self.patience:
                    logger.warning(f"Early Stopping Triggered! Model hasn't improved in {self.patience} epochs.")
                    break
                    
            # Record history
            history.append({**train_metrics, **val_metrics})

        total_time = (time.time() - start_time) / 60
        logger.info("="*60)
        logger.info(f"TRAINING COMPLETE: {self.model_name.upper()}")
        logger.info(f"Total Time: {total_time:.1f} minutes | Best Val Loss: {best_val_loss:.4f}")
        logger.info(f"Artifact Location: {self.save_path}")
        logger.info("="*60)


def execute_pipeline():
    """Main function to load data, instantiate models, and trigger training."""
    
    # 1. Load Global Configuration
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Engine Authorized. Running on: {device}")

    # 2. Prepare DataLoaders
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    if not os.path.exists(data_path):
        logger.critical(f"Processed dataset not found at {data_path}. Run feature_factory.py first.")
        sys.exit(1)
        
    factory = DataLoaderFactory(config_path)
    train_loader, val_loader, input_dim = factory.create_dataloaders(data_path)
    
    # Update config input_dim dynamically so models know how many features exist
    config['models']['lstm']['input_dim'] = input_dim
    config['models']['transformer']['input_dim'] = input_dim

    # --------------------------------------------------------------------------
    # 3. TRAIN LSTM ENGINE
    # --------------------------------------------------------------------------
    lstm_brain = WingoMTLLSTM(
        input_dim=input_dim,
        hidden_dim=config['models']['lstm']['hidden_dim'],
        num_layers=config['models']['lstm']['num_layers'],
        dropout=config['models']['lstm']['dropout']
    )
    lstm_trainer = SequenceTrainer("LSTM", lstm_brain, config, device)
    lstm_trainer.fit(train_loader, val_loader)

    # --------------------------------------------------------------------------
    # 4. TRAIN TRANSFORMER ENGINE
    # --------------------------------------------------------------------------
    transformer_brain = WingoMTLTransformer(
        input_dim=input_dim,
        seq_len=config['data_pipeline']['feature_engineering']['sequence_length'],
        d_model=config['models']['transformer']['d_model'],
        nhead=config['models']['transformer']['nhead'],
        num_layers=config['models']['transformer']['num_layers'],
        dim_feedforward=config['models']['transformer']['dim_feedforward'],
        dropout=config['models']['transformer']['dropout']
    )
    transformer_trainer = SequenceTrainer("Transformer", transformer_brain, config, device)
    transformer_trainer.fit(train_loader, val_loader)

    logger.info("ALL DEEP SEQUENCE MODELS TRAINED AND SERIALIZED.")


if __name__ == "__main__":
    execute_pipeline()