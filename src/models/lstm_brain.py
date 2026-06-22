# ==============================================================================
# ENGINE_ALPHA - ADVANCED NEURAL ARCHITECTURE: GRN-AUGMENTED MULTI-TASK LSTM
# Core Component: src/models/lstm_brain.py
# Description: Ultra-deep, sequence-mapping Recurrent Engine featuring:
# 1. Gated Residual Networks (GRN) for noise suppression.
# 2. Variational Dropout for sequence stabilization.
# 3. Multi-Head Temporal Attention for extreme long-range dependencies.
# 4. Homoscedastic Task Uncertainty for dynamic MTL Loss balancing.
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class VariationalDropout(nn.Module):
    """
    Standard dropout drops different features at every time step, destroying 
    sequence memory. Variational Dropout locks the dropout mask, dropping the 
    exact same features across the entire temporal window.
    """
    def __init__(self, dropout: float):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0:
            return x
        # x shape: (Batch, Seq_Len, Features)
        # Create a mask for (Batch, 1, Features) and broadcast it across Seq_Len
        mask = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli_(1.0 - self.dropout)
        mask = mask / (1.0 - self.dropout)
        return x * mask


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) based on the Temporal Fusion Transformer architecture.
    Applies non-linear processing to filter out PRNG noise before it hits the LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Gating Mechanism (GLU)
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Skip Connection Projection (if dimensions mismatch)
        self.skip_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)
        
        # Non-linear processing
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # GLU Gating
        gate_out = self.gate(x)
        gate_act, gate_sig = gate_out.chunk(2, dim=-1)
        x = gate_act * torch.sigmoid(gate_sig)
        
        # Add & Norm
        return self.norm(residual + x)


class MultiHeadTemporalAttention(nn.Module):
    """
    Upgraded from single-head to Multi-Head Attention. 
    Allows the model to look at multiple different past events simultaneously 
    (e.g., Head 1 looks for color streaks, Head 2 looks for lag spikes).
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(MultiHeadTemporalAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = lstm_outputs.size()
        
        # In temporal sequence prediction, we use the final time-step as the Query
        query = lstm_outputs[:, -1:, :] # (Batch, 1, Hidden)
        keys = lstm_outputs             # (Batch, Seq, Hidden)
        values = lstm_outputs           # (Batch, Seq, Hidden)
        
        # Project and reshape into multiple heads
        Q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(values).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply weights to Values
        context = torch.matmul(attention_weights, V) # (Batch, Heads, 1, Head_Dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1) # (Batch, Hidden)
        
        return self.out_proj(context)


class TaskUncertaintyWeights(nn.Module):
    """
    Kendall et al. (2018) Multi-Task Loss Weighting.
    The network learns to scale the loss of Size vs Number vs Color automatically
    based on the homoscedastic uncertainty of each task.
    """
    def __init__(self, num_tasks: int = 5):
        super(TaskUncertaintyWeights, self).__init__()
        # Initialize log variances at 0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def compute_loss(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List of scalar tensors [loss_size, loss_num, loss_red, loss_green, loss_violet]
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


class WingoMTLLSTM(nn.Module):
    """
    The Ultimate PRNG Sequence LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super(WingoMTLLSTM, self).__init__()
        logger.info(f"Initializing WingoMTLLSTM -> Input: {input_dim}, Hidden: {hidden_dim}, Layers: {num_layers}")
        
        # 1. Feature Filtering Layer
        self.grn = GatedResidualNetwork(input_dim, hidden_dim, dropout)
        
        # 2. Recurrent Core
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 # We use custom Variational Dropout between layers manually if needed
        )
        self.var_dropout = VariationalDropout(dropout)
        
        # 3. Attention Engine
        self.attention = MultiHeadTemporalAttention(hidden_dim, num_heads=8)
        
        # 4. Multi-Task Shared Bottleneck
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 5. Dedicated Task Heads (Heavily parameterized to prevent bottlenecking)
        self.head_size = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.head_number = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 10)
        )
        self.head_red = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.head_green = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.head_violet = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        
        # 6. Automatic Loss Balancer
        self.uncertainty_balancer = TaskUncertaintyWeights(num_tasks=5)
        
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name: nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name: nn.init.orthogonal_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1.0) # Forget gate bias = 1

    def forward(self, x: torch.Tensor) -> dict:
        # Step 1: Filter noise
        x = self.grn(x)
        
        # Step 2: Apply temporal-consistent dropout
        x = self.var_dropout(x)
        
        # Step 3: Recurrent mapping
        lstm_out, _ = self.lstm(x)
        
        # Step 4: Multi-Head Attention mapping
        context = self.attention(lstm_out)
        
        # Step 5: Extract shared embeddings
        shared = self.shared_fc(context)
        
        # Step 6: Multi-Task Predictions
        return {
            'binary_size': self.head_size(shared).squeeze(-1),
            'exact_number': self.head_number(shared),
            'one_hot_red': self.head_red(shared).squeeze(-1),
            'one_hot_green': self.head_green(shared).squeeze(-1),
            'one_hot_violet': self.head_violet(shared).squeeze(-1)
        }