# ==============================================================================
# ENGINE_ALPHA - ADVANCED NEURAL ARCHITECTURE: NOISY DUELING DQN (D3QN)
# Core Component: src/models/dqn_agent.py
# Description: Institutional-grade RL Agent featuring:
# 1. Factorized Noisy Linear Layers for continuous, safe mathematical exploration.
# 2. Dueling Architecture isolating overall financial state from specific bet advantages.
# 3. State-Attention to mathematically weigh the confidence of upstream models.
# 4. Embedded Prioritized Experience Replay (PER) memory matrix structure.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear Layer (Fortunato et al. 2017).
    Replaces Epsilon-Greedy. The network learns a parameter 'sigma' which injects 
    variance directly into its weights. This allows the bot to explore betting 
    strategies safely, driven by the network's own uncertainty.
    """
    def __init__(self, in_features: int, out_features: int, initial_sigma: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard learnable weights (Mu)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # Learnable variance scaling (Sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Non-learnable noise buffers (Updated per forward pass during training)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.initial_sigma = initial_sigma
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.initial_sigma / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.initial_sigma / math.sqrt(self.in_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Factorized Gaussian noise scaling."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Generates new noise tensors. Must be called every step during training."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Add learned variance to standard weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During live casino execution, disable noise for deterministic optimal moves
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


class StateAttentionGate(nn.Module):
    """
    Evaluates the input state vector and learns which model to trust.
    If the XGBoost probability is acting erratically, this gate mathematically 
    suppresses it so the RL agent doesn't blow the bankroll on a false signal.
    """
    def __init__(self, state_dim: int):
        super(StateAttentionGate, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
            nn.Sigmoid() # Outputs a squashed 0-1 multiplier for every feature
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(state)
        # Scale the original state by the learned confidence weights
        return state * attention_weights


class DuelingNoisyDQNBrain(nn.Module):
    """
    The Ultimate Financial Execution Engine.
    Combines Dueling Architecture with Learned Exploration Noise and State Attention.
    """
    def __init__(self, state_dim: int = 8, action_dim: int = 5):
        """
        state_dim (8): [LSTM_Prob, Trans_Prob, XGB_Prob, VAE_Anomaly_Score, 
                        Current_Drawdown, Win_Streak, Bankroll_Pct, Risk_Free_Rate]
        action_dim (5): [0: No Bet, 1: 1% Bankroll, 2: 2.5% Bankroll, 3: 5% Bankroll, 4: 10% Bankroll]
        """
        super(DuelingNoisyDQNBrain, self).__init__()
        logger.info(f"Initializing DuelingNoisyDQNBrain -> State: {state_dim}, Actions: {action_dim}")
        
        # 1. State Pre-processing
        self.state_gate = StateAttentionGate(state_dim)
        
        # 2. Shared Core
        self.shared_features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        
        # 3. Dueling Stream: VALUE (V)
        # How safe is our current bankroll situation?
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            NoisyLinear(64, 1) # Single scalar output for the state value
        )
        
        # 4. Dueling Stream: ADVANTAGE (A)
        # Which specific bet size yields the highest mathematical expectation?
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            NoisyLinear(64, action_dim) # Outputs score per action
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Filter untrustworthy probabilities
        gated_state = self.state_gate(state)
        
        # Extract features
        features = self.shared_features(gated_state)
        
        # Calculate Value and Advantages
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling Aggregation Equation
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
        
    def reset_noise(self):
        """Must be called before every training step to generate new exploration matrices."""
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ==============================================================================
# RL UTILITY: PRIORITIZED EXPERIENCE REPLAY (PER) BUFFER
# Embedded directly into the agent architecture file for modularity.
# ==============================================================================

class PrioritizedReplayBuffer:
    """
    Standard RL agents learn from random past experiences.
    PER forces the agent to frequently re-train on the mistakes that cost it the most money.
    Uses proportional prioritization.
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Saves a transition. New transitions get the maximum priority initially."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
        """
        Samples a batch based on priority weightings.
        Calculates Importance Sampling (IS) weights to correct bias introduced by PER.
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # P(i) = p_i^alpha / sum(p_i^alpha)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Select indices based on probability
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate Importance Sampling (IS) Weights: w_i = (N * P(i)) ^ -beta
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize to keep gradients stable
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Unzip samples
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            indices,
            weights
        )
        
    def update_priorities(self, batch_indices: list, batch_priorities: np.ndarray):
        """Updates the priority of the sampled batch based on the TD-Error (loss)."""
        for idx, priority in zip(batch_indices, batch_priorities):
            # Add a tiny constant (1e-5) to prevent priority from ever reaching exact 0
            self.priorities[idx] = priority + 1e-5
            
    def __len__(self):
        return len(self.buffer)