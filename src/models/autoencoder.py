# ==============================================================================
# ENGINE_ALPHA - ADVANCED NEURAL ARCHITECTURE: TEMPORAL VARIATIONAL AUTOENCODER
# Core Component: src/models/autoencoder.py
# Description: 1D-Convolutional VAE for PRNG Anomaly Detection.
# Escapes the "flattening" problem of dense networks by sweeping causal 
# convolutions over the time-series. Calculates Kullback-Leibler Divergence 
# to mathematically prove if the server's random seed has fractured.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ResidualConv1DBlock(nn.Module):
    """
    1D Convolutional Residual Block.
    Preserves spatial/temporal relationships better than dense linear layers.
    Using Mish activation (Self-Regularized Non-Monotonic Activation) which
    penetrates deep networks better than ReLU or SiLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ResidualConv1DBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.mish1 = nn.Mish()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.mish2 = nn.Mish()
        
        # Skip connection alignment
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        return self.mish2(x + residual)


class WingoTemporalVAE(nn.Module):
    """
    Temporal Variational Autoencoder (VAE).
    Compresses sequence history into a Mean (Mu) and Variance (LogVar) space.
    If the platform generates a sequence that violates the learned Gaussian 
    distribution, the KL-Divergence loss explodes, acting as our Anomaly Trigger.
    """
    def __init__(self, input_dim: int, sequence_length: int = 60, latent_dim: int = 32):
        super(WingoTemporalVAE, self).__init__()
        logger.info(f"Initializing WingoTemporalVAE -> Input: {input_dim}, Seq: {sequence_length}, Latent: {latent_dim}")
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # We need to calculate the flattened size after the Conv1D downsampling
        # Assuming sequence_length = 60, after two stride=2 blocks, len becomes 15
        self.downsampled_len = sequence_length // 4 
        self.flattened_size = 128 * self.downsampled_len
        
        # ----------------------------------------------------------------------
        # 1. THE ENCODER (Compressing Time)
        # ----------------------------------------------------------------------
        # PyTorch Conv1d expects shape: (Batch, Channels/Features, Sequence_Length)
        self.encoder_conv = nn.Sequential(
            # Block 1: Keep length 60, increase channels to 64
            ResidualConv1DBlock(input_dim, 64, kernel_size=3, stride=1, padding=1),
            # Block 2: Downsample length to 30, increase channels to 128
            ResidualConv1DBlock(64, 128, kernel_size=3, stride=2, padding=1),
            # Block 3: Downsample length to 15, keep channels 128
            ResidualConv1DBlock(128, 128, kernel_size=3, stride=2, padding=1)
        )
        
        # Probabilistic Latent Space Mappings
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # ----------------------------------------------------------------------
        # 2. THE DECODER (Reconstructing Chaos)
        # ----------------------------------------------------------------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.Mish()
        )
        
        self.decoder_conv = nn.Sequential(
            # Block 1: Upsample length from 15 to 30
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.Mish(),
            # Block 2: Upsample length from 30 to 60
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.Mish(),
            # Final output mapping back to original features
            nn.Conv1d(32, input_dim, kernel_size=3, stride=1, padding=1)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The "Reparameterization Trick".
        Allows gradients to flow backward through the stochastic random sampling process.
        z = mu + standard_deviation * epsilon (where epsilon is random normal noise)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            return mu + epsilon * std
        else:
            # During live inference, we don't inject random noise; we use the pure mean
            return mu

    def forward(self, x: torch.Tensor) -> dict:
        """
        x input shape from DataLoader: (Batch, Seq_Len, Features)
        Conv1d requires: (Batch, Features, Seq_Len)
        """
        batch_size = x.size(0)
        
        # Transpose for Convolution
        x_conv = x.transpose(1, 2) # (Batch, Features, Seq_Len)
        
        # Encode
        encoded = self.encoder_conv(x_conv) # (Batch, 128, 15)
        flattened = encoded.view(batch_size, -1) # (Batch, 128 * 15)
        
        # Extract Probabilistic Latent Space
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        
        # Sample Latent Vector Z
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded_flat = self.decoder_fc(z)
        decoded_reshaped = decoded_flat.view(batch_size, 128, self.downsampled_len)
        reconstructed = self.decoder_conv(decoded_reshaped) # (Batch, Features, Seq_Len)
        
        # Transpose back to match original input
        reconstructed = reconstructed.transpose(1, 2) # (Batch, Seq_Len, Features)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar
        }

    def loss_function(self, reconstructed: torch.Tensor, original: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> dict:
        """
        Custom VAE Loss Function.
        Combines Mean Squared Error (Reconstruction) with KL-Divergence (Distribution Integrity).
        """
        # 1. Reconstruction Loss (How well did we rebuild the sequence?)
        recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
        
        # 2. KL Divergence (How far did the sequence stray from normal Gaussian randomness?)
        # Mathematical formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Total Loss (Beta allows us to weight the importance of the KL penalty)
        total_loss = recon_loss + (beta * kld_loss)
        
        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kld_loss': kld_loss}