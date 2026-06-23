# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE VAE TRAINING ENGINE & LATENT GMM CALIBRATOR
# Core Component: src/training/train_autoencoder.py
# Description: Unsupervised execution loop for the 1D-Conv VAE.
# Implements Multi-GPU DataParallel scaling, Precision-Split Autocasting, 
# Beta-Lock Cyclical Annealing, and fits a Bayesian Gaussian Mixture Model (GMM)
# on the Latent Space to calculate true structural Mahalanobis probabilities.
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
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

# 🚀 ULTRA-MAX PERFORMANCE TWEAKS
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

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
    Advanced Cyclical Annealing Schedule with Terminal Beta-Lock.
    Gradually increases the KL-Divergence weight (Beta) from 0 to 1 over multiple cycles.
    CRITICAL UPGRADE: Enforces a "cooldown" period where Beta is permanently locked 
    at 1.0 for the final X% of training to ensure the posterior does not collapse 
    and the latent space fully stabilizes before GMM extraction.
    """
    def __init__(self, total_steps: int, cycles: int = 4, ratio: float = 0.5, cooldown_fraction: float = 0.20):
        self.total_steps = total_steps
        self.cycles = cycles
        self.ratio = ratio
        self.cooldown_fraction = cooldown_fraction
        self.cooldown_start_step = int(total_steps * (1.0 - cooldown_fraction))

    def get_beta(self, current_step: int) -> float:
        # 🛡️ Beta-Lock: If we are in the final phase of training, hold Beta at absolute 1.0
        if current_step >= self.cooldown_start_step:
            return 1.0

        # Calculate standard cyclic phase
        active_steps = self.cooldown_start_step
        period = active_steps / self.cycles
        step_in_cycle = current_step % period
        tau = step_in_cycle / (period * self.ratio)
        
        return 1.0 if tau >= 1.0 else tau


class LatentSpaceEvaluator:
    """
    Replaces static MSE pixel-by-pixel anomaly detection.
    Extracts the 'DNA' (16-D bottleneck mu vector) of all sequences and fits a 
    Bayesian Gaussian Mixture Model. Calculates log-probabilities (Mahalanobis distance) 
    to detect true structural algorithm shifts rather than surface-level noise.
    """
    def __init__(self, n_components: int = 5, covariance_type: str = 'full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        
        # Bayesian GMM automatically determines the optimal number of effective clusters
        self.gmm = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=500,
            n_init=3,
            weight_concentration_prior_type='dirichlet_process',
            random_state=42
        )
        self.scaler = StandardScaler()

    def fit_and_evaluate(self, latent_vectors: np.ndarray, quantile: float = 0.05) -> Tuple[float, float, float]:
        """
        Fits the GMM and calculates the structural anomaly threshold based on 
        the log-likelihood of the normal training sequences.
        """
        logger.info(f"Fitting Bayesian GMM on Latent Space Manifold: Shape {latent_vectors.shape}")
        
        # Scale latents for GMM numerical stability
        scaled_latents = self.scaler.fit_transform(latent_vectors)
        self.gmm.fit(scaled_latents)
        
        # Calculate Log-Probabilities (Higher is more normal, Lower is anomalous)
        log_probs = self.gmm.score_samples(scaled_latents)
        
        # Threshold is the lowest X% of log-probabilities
        # Example: A 0.05 quantile means the bottom 5% most structurally weird sequences are flagged
        threshold_log_prob = np.quantile(log_probs, quantile)
        mean_log_prob = np.mean(log_probs)
        min_log_prob = np.min(log_probs)
        
        return float(threshold_log_prob), float(mean_log_prob), float(min_log_prob)

    def save_artifacts(self, artifact_dir: str):
        gmm_path = os.path.join(artifact_dir, "latent_gmm_model.joblib")
        scaler_path = os.path.join(artifact_dir, "latent_gmm_scaler.joblib")
        joblib.dump(self.gmm, gmm_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Latent GMM Engine and Scaler locked to: {artifact_dir}")


class VAETrainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.device = device
        self.config = config
        
        # 🚀 MULTI-GPU AUTO-DETECTION AND WRAPPING
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1 and self.device.type == 'cuda':
            logger.info(f"🔥 ENGAGING MULTI-GPU OVERDRIVE: Distributing VAE across {self.gpu_count} GPUs!")
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)
            
        vae_cfg = self.config['models']['autoencoder']
        self.epochs = vae_cfg['epochs']
        self.lr = vae_cfg['learning_rate']
        self.anomaly_quantile = 1.0 - vae_cfg['anomaly_threshold_quantile'] # Invert for log-prob (e.g. 0.95 -> 0.05)
        
        # AdamW with weight decay keeps the latent space perfectly smooth
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3, eps=1e-8)
        self.scaler = GradScaler()
        
        self.artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['unsupervised_artifact_dir'])
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.save_path = os.path.join(self.artifact_dir, "temporal_vae_weights.pt")
        self.threshold_path = os.path.join(self.artifact_dir, "latent_anomaly_threshold.json")

    @property
    def base_model(self):
        """Safely extracts the core model from the DataParallel wrapper."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def fit(self, train_loader, val_loader):
        logger.info("="*70)
        logger.info("INITIATING VAE UNSUPERVISED TRAINING SEQUENCE")
        logger.info(f"Target Device: {self.device} | Epochs: {self.epochs} | GPUs: {max(1, self.gpu_count)}")
        logger.info("="*70)
        
        total_steps = len(train_loader) * self.epochs
        # Allocate the last 20% of epochs to absolute Beta=1.0 for latent settling
        beta_scheduler = CyclicalBetaScheduler(total_steps=total_steps, cycles=4, cooldown_fraction=0.20)
        
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
                
                # 🚀 PHASE 1: HYPER-SPEED FORWARD PASS (FP16 / Mixed Precision)
                with autocast():
                    outputs = self.model(X)
                
                # 🛡️ PHASE 2: PRECISION SPLIT SHIELD (FP32)
                # Force the output tensors back into 32-bit floats before calculating exponents.
                # Clamp the logvar mathematically to prevent e^(20+) infinity explosions!
                recon_fp32 = outputs['reconstructed'].float()
                mu_fp32 = outputs['mu'].float()
                logvar_fp32 = torch.clamp(outputs['logvar'].float(), min=-30.0, max=20.0)
                
                losses = self.base_model.loss_function(
                    recon_fp32, X.float(), mu_fp32, logvar_fp32, beta=current_beta
                )
                
                # Phase 3: Scaled Backpropagation
                self.scaler.scale(losses['total_loss']).backward()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_recon_loss += losses['recon_loss'].item()
                epoch_kld_loss += losses['kld_loss'].item()
                global_step += 1

            # Validation Phase
            self.model.eval()
            val_loss, val_recon, val_kld = 0.0, 0.0, 0.0
            
            with torch.no_grad():
                for X, _ in val_loader:
                    X = X.to(self.device, non_blocking=True)
                    
                    with autocast():
                        outputs = self.model(X)
                        
                    recon_fp32 = outputs['reconstructed'].float()
                    mu_fp32 = outputs['mu'].float()
                    logvar_fp32 = torch.clamp(outputs['logvar'].float(), min=-30.0, max=20.0)
                    
                    v_losses = self.base_model.loss_function(
                        recon_fp32, X.float(), mu_fp32, logvar_fp32, beta=1.0
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
                torch.save(self.base_model.state_dict(), self.save_path)
                logger.info(log_str + " [💾 BEST ARTIFACT SAVED]")
            else:
                logger.info(log_str)

        logger.info(f"VAE Training Complete. Total Time: {(time.time() - start_time)/60:.1f}m")
        
        # Transition to Structural Evaluation
        self._calibrate_latent_anomaly_threshold(train_loader)

    def _calibrate_latent_anomaly_threshold(self, dataloader):
        """
        Executes the Latent Space Isolation Protocol.
        Passes the entire training set through the frozen network to extract the 
        structural DNA (mu), then fits the Bayesian GMM.
        """
        logger.info("="*70)
        logger.info("CALIBRATING BAYESIAN LATENT STRUCTURAL ANOMALY THRESHOLD")
        logger.info("="*70)
        
        self.base_model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        
        latent_vectors = []
        
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = self.model(X)
                    
                # Extract the 16-D bottleneck mu vector for every sequence in the batch
                mu_batch = outputs['mu'].float().cpu().numpy()
                latent_vectors.append(mu_batch)
                    
        latent_matrix = np.vstack(latent_vectors)
        
        # Instantiate and fit the GMM Evaluator
        evaluator = LatentSpaceEvaluator(n_components=5, covariance_type='full')
        threshold, mean_log, min_log = evaluator.fit_and_evaluate(latent_matrix, quantile=self.anomaly_quantile)
        
        logger.info(f"Latent Structural Calibration Complete:")
        logger.info(f" -> Mean Log-Probability: {mean_log:.5f} (Normal Sequence)")
        logger.info(f" -> Absolute Min Log-Probability: {min_log:.5f}")
        logger.info(f" -> {self.anomaly_quantile * 100}% Anomaly Threshold (Trigger below): {threshold:.5f}")
        
        # Save structural thresholds and the GMM artifacts
        calibration_data = {
            "latent_anomaly_log_prob_threshold": float(threshold),
            "baseline_mean_log_prob": float(mean_log),
            "quantile_used": float(self.anomaly_quantile),
            "note": "Any sequence scoring lower than the threshold log-prob is structurally anomalous."
        }
        with open(self.threshold_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)
            
        evaluator.save_artifacts(self.artifact_dir)
        logger.info(f"Calibration Manifest saved to {self.threshold_path}")
        logger.info("="*70)


def execute_vae_pipeline():
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    factory = DataLoaderFactory(config_path)
    
    # 🚀 DYNAMIC BATCH MULTIPLIER
    base_batch = config['models']['autoencoder']['batch_size']
    gpu_multiplier = max(1, torch.cuda.device_count())
    factory.batch_size_lstm = base_batch * gpu_multiplier
    
    logger.info(f"Dynamic Batch Sizing: Scaling {base_batch} x {gpu_multiplier} GPUs = {factory.batch_size_lstm} seqs/batch!")
    
    train_loader, val_loader, input_dim = factory.create_dataloaders(data_path)
    
    seq_len = config['data_pipeline']['feature_engineering']['sequence_length']
    bottleneck = config['models']['autoencoder']['bottleneck_dim']
    
    vae_model = WingoTemporalVAE(input_dim=input_dim, sequence_length=seq_len, latent_dim=bottleneck)
    
    trainer = VAETrainer(vae_model, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    execute_vae_pipeline()