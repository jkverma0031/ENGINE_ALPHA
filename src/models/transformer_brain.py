# ==============================================================================
# ENGINE_ALPHA - ADVANCED NEURAL ARCHITECTURE: TIME2VEC MTL TRANSFORMER
# Core Component: src/models/transformer_brain.py
# Description: Extreme-scale Time-Series mapping using:
# 1. Time2Vec: Learnable temporal embedding layers (Kazemi et al., 2019).
# 2. SwiGLU: LLaMA-style Gated Feed-Forward networks for complex boundaries.
# 3. Pre-Norm architecture for stable deep-layer gradient flow.
# 4. Stochastic Depth (DropPath) for extreme regularization against PRNG noise.
# ==============================================================================

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

class Time2Vec(nn.Module):
    """
    Replaces static sine/cosine positional encodings.
    Time2Vec explicitly learns periodic algorithms (like a PRNG's cyclic seed shift)
    by training the frequencies and phase shifts of the sine waves directly.
    """
    def __init__(self, sequence_length: int, out_dim: int):
        super(Time2Vec, self).__init__()
        self.out_dim = out_dim
        
        # Linear (Non-periodic) time mapping
        self.w0 = nn.parameter.Parameter(torch.randn(sequence_length, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(sequence_length, 1))
        
        # Periodic time mapping (Sine waves with learnable frequencies)
        self.w = nn.parameter.Parameter(torch.randn(sequence_length, out_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(sequence_length, out_dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a dummy tensor just to get the batch size and device
        batch_size = x.size(0)
        seq_len = self.w0.size(0)
        
        # Time steps tensor
        tau = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1) # (Seq, 1)
        
        # Linear component
        v1 = tau * self.w0 + self.b0 # (Seq, 1)
        
        # Periodic component
        v2 = torch.sin(tau * self.w + self.b) # (Seq, out_dim - 1)
        
        # Concatenate and broadcast to batch size
        t2v = torch.cat([v1, v2], dim=-1) # (Seq, out_dim)
        t2v = t2v.unsqueeze(0).expand(batch_size, -1, -1) # (Batch, Seq, out_dim)
        
        return t2v


class DropPath(nn.Module):
    """
    Stochastic Depth per sample. 
    Drops entire residual branches during training to create an ensemble of shallower 
    networks, vastly reducing overfitting on noisy financial/betting data.
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Work with any tensor dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() # binarize
        
        output = x.div(keep_prob) * random_tensor
        return output


class SwiGLU(nn.Module):
    """
    LLaMA-3 Style Feed Forward Network.
    Swish-Gated Linear Unit significantly outperforms ReLU in attention networks
    when searching for highly complex, non-linear mathematical boundaries.
    """
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate and Activation
        gate = self.silu(self.fc1(x))
        x_val = self.fc2(x)
        x = gate * x_val
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x


class PreNormEncoderLayer(nn.Module):
    """
    Custom Transformer Block using Pre-Norm architecture.
    Standard PyTorch Transformers use Post-Norm, which causes deep layer gradients 
    to vanish. Pre-Norm allows for infinite scaling of depth.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1, drop_path: float = 0.1):
        super(PreNormEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward, drop=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention Block (Pre-Norm)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path1(attn_out)
        
        # FFN Block (Pre-Norm)
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.drop_path2(ffn_out)
        
        return x


class TaskUncertaintyWeights(nn.Module):
    """Reused from LSTM: Homoscedastic uncertainty loss balancing."""
    def __init__(self, num_tasks: int = 5):
        super(TaskUncertaintyWeights, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def compute_loss(self, losses: list) -> torch.Tensor:
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


class WingoMTLTransformer(nn.Module):
    """
    The Ultimate PRNG Sequence Transformer.
    """
    def __init__(self, input_dim: int, seq_len: int = 60, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.2):
        super(WingoMTLTransformer, self).__init__()
        logger.info(f"Initializing WingoMTLTransformer -> d_model: {d_model}, layers: {num_layers}")
        
        # 1. Feature & Time Embeddings
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.time_embedding = Time2Vec(seq_len, d_model)
        
        # 2. Deep Transformer Core
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                drop_path=0.1 * (i / max(1, num_layers - 1)) # Stocastic Depth Scaling
            ) for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        # 3. Temporal Convolution Pooling
        # Better than average pooling; it sweeps across the sequence to find local motifs
        self.conv_pool = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        
        # 4. Shared Extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 5. Specialized Multi-Task Heads
        self.head_size = nn.Sequential(nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1))
        self.head_number = nn.Sequential(nn.Linear(d_model, 128), nn.SiLU(), nn.Linear(128, 10))
        self.head_red = nn.Linear(d_model, 1)
        self.head_green = nn.Linear(d_model, 1)
        self.head_violet = nn.Linear(d_model, 1)
        
        # 6. Automatic Loss Balancer
        self.uncertainty_balancer = TaskUncertaintyWeights(num_tasks=5)

    def forward(self, x: torch.Tensor) -> dict:
        # Step 1: Embed Features and Time
        feat_embed = self.feature_projection(x)
        time_embed = self.time_embedding(x)
        
        # Additive embedding
        x = feat_embed + time_embed
        
        # Step 2: Transformer Blocks
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x) # (Batch, Seq, d_model)
        
        # Step 3: Conv1D Pooling over the temporal dimension
        x = x.transpose(1, 2) # (Batch, d_model, Seq)
        x = self.conv_pool(x)
        x = torch.max(x, dim=2)[0] # Global Max Pooling -> (Batch, d_model)
        
        # Step 4: Extract shared features
        shared = self.shared_fc(x)
        
        # Step 5: Route to heads
        return {
            'binary_size': self.head_size(shared).squeeze(-1),
            'exact_number': self.head_number(shared),
            'one_hot_red': self.head_red(shared).squeeze(-1),
            'one_hot_green': self.head_green(shared).squeeze(-1),
            'one_hot_violet': self.head_violet(shared).squeeze(-1)
        }