# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE VAE TRAINING ENGINE & ANOMALY CALIBRATOR
# Core Component: src/training/train_autoencoder.py
# Description: Unsupervised execution loop for the 1D-Conv VAE.
# Implements Cyclical Beta-Annealing to prevent Posterior Collapse, and 
# calculates the absolute Anomaly Threshold for live inference triggering.
# ==============================================================================

import os
import sys
import yaml
import time
import json
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [VAETrainer] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.data.dataset_loader import DataLoaderFactory
    from src.models.autoencoder import WingoTemporalVAE
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


class CyclicalBetaScheduler:
    """
    Fu et al. (2019) Cyclical Annealing Schedule.
    Gradually increases the KL-Divergence weight (Beta) from 0 to 1 over multiple 
    cycles. This prevents the VAE from ignoring the latent space distribution 
    during early training epochs.
    """
    def __init__(self, total_steps: int, cycles: int = 4, ratio: float = 0.5):
        self.total_steps = total_steps
        self.cycles = cycles
        self.ratio = ratio # What fraction of the cycle is spent increasing Beta

    def get_beta(self, current_step: int) -> float:
        period = self.total_steps / self.cycles
        step_in_cycle = current_step % period
        
        # Calculate the proportion of the increasing phase
        tau = step_in_cycle / (period * self.ratio)
        
        if tau >= 1.0:
            return 1.0 # Cap Beta at 1.0 for the remainder of the cycle
        else:
            return tau # Linear increase


class VAETrainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        vae_cfg = self.config['models']['autoencoder']
        self.epochs = vae_cfg['epochs']
        self.lr = vae_cfg['learning_rate']
        self.quantile = vae_cfg['anomaly_threshold_quantile']
        
        # Using AdamW with significant weight decay for smoother latent mappings
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        self.scaler = GradScaler()
        
        self.artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['unsupervised_artifact_dir'])
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.save_path = os.path.join(self.artifact_dir, "temporal_vae_weights.pt")
        self.threshold_path = os.path.join(self.artifact_dir, "anomaly_threshold.json")

    def fit(self, train_loader, val_loader):
        logger.info("="*60)
        logger.info("INITIATING VAE UNSUPERVISED TRAINING SEQUENCE")
        logger.info(f"Target Device: {self.device} | Epochs: {self.epochs}")
        logger.info("="*60)
        
        total_steps = len(train_loader) * self.epochs
        beta_scheduler = CyclicalBetaScheduler(total_steps=total_steps, cycles=4)
        
        best_val_loss = float('inf')
        global_step = 0
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            self.model.train()
            
            epoch_recon_loss = 0.0
            epoch_kld_loss = 0.0
            current_beta = 0.0
            
            for X, _ in train_loader:
                X = X.to(self.device, non_blocking=True)
                current_beta = beta_scheduler.get_beta(global_step)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(X)
                    losses = self.model.loss_function(
                        outputs['reconstructed'], X, outputs['mu'], outputs['logvar'], beta=current_beta
                    )
                
                self.scaler.scale(losses['total_loss']).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_recon_loss += losses['recon_loss'].item()
                epoch_kld_loss += losses['kld_loss'].item()
                global_step += 1

            # Validation Phase (Beta fixed at 1.0 for true evaluation)
            self.model.eval()
            val_loss, val_recon, val_kld = 0.0, 0.0, 0.0
            with torch.no_grad():
                for X, _ in val_loader:
                    X = X.to(self.device, non_blocking=True)
                    with autocast():
                        outputs = self.model(X)
                        v_losses = self.model.loss_function(
                            outputs['reconstructed'], X, outputs['mu'], outputs['logvar'], beta=1.0
                        )
                    val_loss += v_losses['total_loss'].item()
                    val_recon += v_losses['recon_loss'].item()
                    val_kld += v_losses['kld_loss'].item()

            batches = len(train_loader)
            v_batches = len(val_loader)
            
            log_str = (
                f"Epoch [{epoch:03d}/{self.epochs:03d}] - {time.time() - ep_start:.1f}s | "
                f"Beta: {current_beta:.3f} | "
                f"Train Recon: {epoch_recon_loss/batches:.4f} | "
                f"Val Recon: {val_recon/v_batches:.4f} | "
                f"Val KLD: {val_kld/v_batches:.4f}"
            )
            
            if val_loss / v_batches < best_val_loss:
                best_val_loss = val_loss / v_batches
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(log_str + " [💾 BEST ARTIFACT SAVED]")
            else:
                logger.info(log_str)

        logger.info(f"VAE Training Complete. Total Time: {(time.time() - start_time)/60:.1f}m")
        self._calibrate_anomaly_threshold(train_loader)

    def _calibrate_anomaly_threshold(self, dataloader):
        """
        After training, we feed normal data through the VAE to calculate its baseline 
        reconstruction error. The 95th percentile of this error becomes the exact 
        boundary for what ENGINE_ALPHA considers a "PRNG Seed Anomaly".
        """
        logger.info("="*60)
        logger.info("CALIBRATING ANOMALY THRESHOLD ALGORITHM")
        
        # Load best weights
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(self.device)
                with autocast():
                    outputs = self.model(X)
                    # We calculate MSE per sequence in the batch
                    mse_per_sequence = torch.mean((outputs['reconstructed'] - X) ** 2, dim=[1, 2])
                    reconstruction_errors.extend(mse_per_sequence.cpu().numpy().tolist())
                    
        # Calculate mathematical boundary
        reconstruction_errors = np.array(reconstruction_errors)
        threshold = np.quantile(reconstruction_errors, self.quantile)
        mean_err = np.mean(reconstruction_errors)
        max_err = np.max(reconstruction_errors)
        
        logger.info(f"Threshold Calibration Complete:")
        logger.info(f" -> Mean Error: {mean_err:.5f}")
        logger.info(f" -> Max Normal Error: {max_err:.5f}")
        logger.info(f" -> {self.quantile * 100}% Anomaly Threshold Trigger: {threshold:.5f}")
        
        # Save threshold for live_engine.py
        calibration_data = {
            "anomaly_threshold_mse": float(threshold),
            "baseline_mean_mse": float(mean_err),
            "quantile_used": float(self.quantile)
        }
        with open(self.threshold_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)
            
        logger.info(f"Calibration Artifact saved to {self.threshold_path}")
        logger.info("="*60)


def execute_vae_pipeline():
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    factory = DataLoaderFactory(config_path)
    
    # VAE can use a much larger batch size than the Transformer
    factory.batch_size_lstm = config['models']['autoencoder']['batch_size']
    train_loader, val_loader, input_dim = factory.create_dataloaders(data_path)
    
    seq_len = config['data_pipeline']['feature_engineering']['sequence_length']
    bottleneck = config['models']['autoencoder']['bottleneck_dim']
    
    vae_model = WingoTemporalVAE(input_dim=input_dim, sequence_length=seq_len, latent_dim=bottleneck)
    
    trainer = VAETrainer(vae_model, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    execute_vae_pipeline()