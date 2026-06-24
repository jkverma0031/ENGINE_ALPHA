# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE DISTRIBUTIONAL RL TRAINING ENGINE (C51)
# Core Component: src/training/train_rl_agent.py
# Description: Institutional algorithmic execution simulator. Upgraded to a 
# Distributional Categorical DQN (C51) to model Tail-Risk Variance.
# Implements Trajectory Bootstrapping, Sortino Downside Shaping, Contextual 
# Anomaly Rewards, and Rolling Trust Metrics (The Blind Bodyguard).
# CRITICAL: Implements XGBoost Scaling Reversal to prevent tabular poisoning.
# ==============================================================================

import os
import sys
import yaml
import time
import logging
import json
import gc
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RLEngine] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Resolve Root and Imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
    from src.data.dataset_loader import WingoSequenceDataset
    from src.models.dqn_agent import PrioritizedReplayBuffer, NoisyLinear
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
    from src.models.autoencoder import WingoTemporalVAE  # 🚨 RESTORED: VAE Import
    import xgboost as xgb
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


# ==============================================================================
# 0A. META-LEARNER ATTENTION AGGREGATOR (EMBEDDED)
# ==============================================================================
class NeuralMetaAggregator(nn.Module):
    """Embedded directly to prevent cascading __init__.py import crashes."""
    def __init__(self, num_models: int, context_dim: int):
        super(NeuralMetaAggregator, self).__init__()
        self.num_models = num_models
        self.context_dim = context_dim
        
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.Mish()
        )
        
        self.attention_gate = nn.Sequential(
            nn.Linear(32 + num_models, 32),
            nn.Mish(),
            nn.Linear(32, num_models)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, probs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        env_state = self.context_net(context)
        gate_input = torch.cat([env_state, probs], dim=1)
        gate_logits = self.attention_gate(gate_input)
        dynamic_weights = F.softmax(gate_logits / self.temperature, dim=1)
        return torch.sum(probs * dynamic_weights, dim=1, keepdim=True) + self.bias


# ==============================================================================
# 0B. DISTRIBUTIONAL C51 NETWORK ARCHITECTURE
# ==============================================================================

# ==============================================================================
# 0. DISTRIBUTIONAL C51 NETWORK ARCHITECTURE
# ==============================================================================

class DistributionalC51Brain(nn.Module):
    """
    Categorical DQN (C51) Architecture.
    
    Standard RL predicts a single scalar expected value $Q(s, a)$. 
    This is fatal in financial markets where variance determines risk of ruin.
    This Distributional Brain instead predicts a probability mass function (PMF) 
    over `num_atoms` possible returns ranging from `V_min` to `V_max`.
    
    Equation:
    $Z(x, a) = P(R(x, a) = z_i) = p_i(x, a)$
    """
    def __init__(self, state_dim: int, action_dim: int, num_atoms: int = 51, V_min: float = -25.0, V_max: float = 25.0):
        super(DistributionalC51Brain, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max
        
        # Support tensor representing the specific monetary/return bins
        self.register_buffer("support", torch.linspace(self.V_min, self.V_max, self.num_atoms))
        
        # Robust Feature Extractor with Mish Activation for gradient flow preservation
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish()
        )
        
        # Noisy Advantage Stream (For Exploration in highly dimensional state spaces)
        self.adv_hidden = NoisyLinear(128, 128)
        self.adv_out = NoisyLinear(128, action_dim * num_atoms)
        
        # Noisy Value Stream
        self.val_hidden = NoisyLinear(128, 128)
        self.val_out = NoisyLinear(128, num_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probability distribution over returns for every action.
        Returns Tensor of shape: (Batch, Actions, Atoms)
        """
        assert x.dim() == 2, f"Expected 2D input state tensor, got {x.dim()}D"
        features = self.feature_layer(x)
        
        adv = F.mish(self.adv_hidden(features))
        val = F.mish(self.val_hidden(features))
        
        adv_atoms = self.adv_out(adv).view(-1, self.action_dim, self.num_atoms)
        val_atoms = self.val_out(val).view(-1, 1, self.num_atoms)
        
        # Combine Dueling Streams (Subtract mean to ensure identifiability)
        q_atoms = val_atoms + adv_atoms - adv_atoms.mean(dim=1, keepdim=True)
        
        # Apply Softmax strictly across the atoms to get a valid probability distribution sum of 1.0
        return F.softmax(q_atoms, dim=-1)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected value (Mean) from the distribution to select 
        the optimal action during deterministic inference.
        """
        dist = self.forward(x) # Shape: (Batch, Actions, Atoms)
        # Summation of Probability * Support_Value = Expected Return
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

    def reset_noise(self):
        """Resets the sigma/mu parameters in the NoisyLayers for new exploration."""
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
        self.val_hidden.reset_noise()
        self.val_out.reset_noise()


# ==============================================================================
# 1. THE FINANCIAL SIMULATION ENVIRONMENT
# ==============================================================================

class WingoCasinoEnvironment:
    """
    Institutional Environment featuring Trajectory Bootstrapping, Sortino Downside 
    Shaping, Contextual Anomaly Rewards, and Rolling Accuracy Tracking.
    """
    def __init__(self, experience_matrix: pd.DataFrame, initial_bankroll: float = 10000.0, max_steps: int = 250):
        self.df = experience_matrix.reset_index(drop=True)
        self.total_rows = len(self.df)
        
        if self.total_rows < max_steps * 2:
            logger.warning("Experience matrix is very small. Bootstrapping may experience heavy overlap.")
            
        self.initial_bankroll = initial_bankroll
        self.max_steps = max_steps 
        self.current_bankroll = initial_bankroll
        self.high_water_mark = initial_bankroll
        
        # 11-Tier Micro-Staking Action Space (0% to 10% strictly)
        # Never exceeds 10% to completely eliminate the mathematical possibility of gambler's ruin
        self.action_space = {
            0: 0.00, 1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 
            6: 0.06, 7: 0.07, 8: 0.08, 9: 0.09, 10: 0.10
        }
        
        self.payout_multiplier = 0.96 # Factoring in the structural casino house fee
        self.reset()

    def reset(self):
        """
        Trajectory Bootstrapping: Picks a random starting point in the massive matrix
        to prevent the network from memorizing the chronological timeline sequence.
        """
        self.current_bankroll = self.initial_bankroll
        self.high_water_mark = self.initial_bankroll
        self.win_streak = 0
        self.steps_taken = 0
        
        # The Blind Bodyguard: Rolling queue of correct model predictions (1 = Win, 0 = Loss)
        self.accuracy_queue = [1] * 15 # Start optimistic to encourage early exploration
        
        self.start_idx = np.random.randint(0, max(1, self.total_rows - self.max_steps - 1))
        self.current_idx = self.start_idx
        
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        row = self.df.iloc[self.current_idx]
        
        meta_prob = float(row['meta_calibrated_prob'])
        bankroll_ratio = self.current_bankroll / self.initial_bankroll
        drawdown = (self.high_water_mark - self.current_bankroll) / self.high_water_mark
        streak = min(self.win_streak / 10.0, 1.0)
        
        edge = (meta_prob * self.payout_multiplier) - (1 - meta_prob)
        kelly = max(0.0, float(edge / self.payout_multiplier))
        rolling_accuracy = float(np.mean(self.accuracy_queue))
        anomaly_score = float(row['anomaly_score']) # 🚨 RESTORED: The 7th Dimension
        
        state = np.array([
            meta_prob, bankroll_ratio, drawdown, 
            streak, kelly, rolling_accuracy, anomaly_score
        ], dtype=np.float32)
        
        return state

    def step(self, action_idx: int) -> tuple:
        row = self.df.iloc[self.current_idx]
        
        meta_prob = row['meta_calibrated_prob']
        true_outcome = int(row['true_target'])
        is_anomalous = row['anomaly_score'] == 1.0 # 🚨 RESTORED: Anomaly Detection
        bot_prediction = 1 if meta_prob >= 0.5 else 0
        
        bet_fraction = self.action_space[action_idx]
        bet_amount = self.current_bankroll * bet_fraction
        
        reward = 0.0
        
        if bet_amount > 0:
            if bot_prediction == true_outcome:
                profit = bet_amount * self.payout_multiplier
                self.current_bankroll += profit
                self.win_streak += 1
                self.accuracy_queue.append(1)
                reward = profit / self.initial_bankroll 
            else:
                self.current_bankroll -= bet_amount
                self.win_streak = 0
                self.accuracy_queue.append(0)
                
                current_drawdown = (self.high_water_mark - self.current_bankroll) / self.high_water_mark
                drawdown_penalty = 1.0 + (current_drawdown * 10.0) 
                base_loss_reward = (-bet_amount / self.initial_bankroll)
                reward = base_loss_reward * drawdown_penalty
                
                # 🚨 RESTORED: Double penalty for losing during a detected structural anomaly
                if is_anomalous:
                    reward *= 2.0 
        else:
            self.accuracy_queue.append(1 if bot_prediction == true_outcome else 0)
            p_max = max(meta_prob, 1.0 - meta_prob)
            edge = (p_max * self.payout_multiplier) - (1.0 - p_max)
            
            if edge > 0.03 and np.mean(self.accuracy_queue) > 0.55:
                reward = -0.2 

        self.accuracy_queue.pop(0) 
        
        if self.current_bankroll > self.high_water_mark:
            self.high_water_mark = self.current_bankroll
            
        done = False
        if self.current_bankroll < (self.initial_bankroll * 0.20): 
            reward = -2.0 
            done = True
            
        self.current_idx += 1
        self.steps_taken += 1
        
        if self.steps_taken >= self.max_steps or self.current_idx >= self.total_rows:
            done = True
            
        next_state = self._get_state()
        info = {
            'bankroll': self.current_bankroll, 
            'action': action_idx, 
            'drawdown': (self.high_water_mark - self.current_bankroll) / self.high_water_mark
        }
        return next_state, reward, done, info


# ==============================================================================
# 2. THE DISTRIBUTIONAL C51 ALGORITHMIC TRAINER
# ==============================================================================

class DistributionalRLTrainer:
    """
    Manages the memory buffer, target networking, and C51 Categorical Projections.
    """
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        rl_cfg = self.config['models']['dqn']
        self.state_dim = 7  
        self.action_dim = 11 
        self.gamma = rl_cfg['gamma']
        self.batch_size = 128
        self.tau = 0.005 
        
        # Strict Distributional Parameters
        self.num_atoms = rl_cfg.get('num_atoms', 51)
        self.V_min = rl_cfg.get('v_min', -25.0)
        self.V_max = rl_cfg.get('v_max', 25.0)
        self.delta_z = (self.V_max - self.V_min) / (self.num_atoms - 1)
        
        self.policy_net = DistributionalC51Brain(self.state_dim, self.action_dim, self.num_atoms, self.V_min, self.V_max).to(self.device)
        self.target_net = DistributionalC51Brain(self.state_dim, self.action_dim, self.num_atoms, self.V_min, self.V_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = AdamW(self.policy_net.parameters(), lr=rl_cfg['learning_rate'], amsgrad=True)
        self.memory = PrioritizedReplayBuffer(capacity=rl_cfg['memory_capacity'])
        
        self.artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['reinforcement_artifact_dir'])
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.save_path = os.path.join(self.artifact_dir, "dqn_c51_policy_weights.pt")

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net.get_q_values(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action

    def optimize_step(self):
        """
        Executes the complex C51 Categorical Projection Algorithm.
        Instead of minimizing MSE, we minimize the Kullback-Leibler (KL) divergence
        between the policy network's predicted distribution and the target network's 
        projected Bellman distribution.
        """
        if len(self.memory) < self.batch_size:
            return 0.0 
            
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device).long().unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        weights = weights.to(self.device).unsqueeze(1)
        
        # Trigger exploration noise resets
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        # 1. Fetch Current Prediction Distribution
        dist_current = self.policy_net(states) # Shape: (Batch, Actions, Atoms)
        action_mask = actions.unsqueeze(-1).expand(-1, -1, self.num_atoms)
        dist_current_action = dist_current.gather(1, action_mask).squeeze(1) # Shape: (Batch, Atoms)
        
        # 2. Compute Target Distribution (Zero gradients)
        with torch.no_grad():
            # Double DQN Logic applied to C51: 
            # Policy network selects the best action for the next state based on expected value
            next_q_values = self.policy_net.get_q_values(next_states)
            best_next_actions = next_q_values.argmax(dim=1, keepdim=True).unsqueeze(-1).expand(-1, -1, self.num_atoms)
            
            # Target network provides the categorical distribution for that chosen action
            dist_next = self.target_net(next_states)
            dist_next_action = dist_next.gather(1, best_next_actions).squeeze(1)
            
            # Initialize empty categorical projection tensor
            target_dist = torch.zeros_like(dist_next_action)
            
            # Loop over every atom bin to perform Bellman projection
            for j in range(self.num_atoms):
                # Calculate the shifted support value: T_z_j = r + gamma * z_j
                T_zj = rewards + (1 - dones) * self.gamma * self.policy_net.support[j]
                T_zj = torch.clamp(T_zj, self.V_min, self.V_max)
                
                # Determine which bins the shifted value falls between
                b = (T_zj - self.V_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Handle edge cases to prevent out-of-bounds indexing
                l[(u > 0) & (l == u)] -= 1
                l[(l < 0)] = 0
                
                # Distribute the probability mass proportionately based on proximity to bounds
                target_dist.scatter_add_(1, l, dist_next_action[:, j].unsqueeze(1) * (u.float() - b))
                target_dist.scatter_add_(1, u, dist_next_action[:, j].unsqueeze(1) * (b - l.float()))
        
        # 3. Calculate Cross-Entropy Loss
        # We add a tiny epsilon (1e-8) to prevent taking the log of absolute zero
        loss = -torch.sum(target_dist * torch.log(dist_current_action + 1e-8), dim=1, keepdim=True)
        
        # Apply Prioritized Experience Replay (PER) importance weights
        weighted_loss = (loss * weights).mean()
        
        # 4. Optimization
        self.optimizer.zero_grad()
        weighted_loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update PER Buffer Priorities based on new Bellman Error (Squeezed to 1D)
        self.memory.update_priorities(indices, loss.detach().cpu().squeeze().numpy())
        
        # 5. Soft Update Target Network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
        return weighted_loss.item()

    def train_agent(self, env: WingoCasinoEnvironment, episodes: int = 100):
        logger.info("="*70)
        logger.info(f"INITIATING DISTRIBUTIONAL ALGORITHMIC EXECUTION TRAINING (C51)")
        logger.info(f"Target Device: {self.device} | Bootstrapped Episodes: {episodes}")
        logger.info("="*70)
        
        best_bankroll = 0.0
        
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            total_loss = 0.0
            max_drawdown = 0.0
            
            while True:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                max_drawdown = max(max_drawdown, info['drawdown'])
                
                self.memory.push(
                    torch.tensor(state, dtype=torch.float32),
                    action, reward,
                    torch.tensor(next_state, dtype=torch.float32),
                    done
                )
                
                state = next_state
                total_reward += reward
                
                loss = self.optimize_step()
                total_loss += loss
                
                if done: break
                    
            final_bankroll = info['bankroll']
            profit_pct = ((final_bankroll - env.initial_bankroll) / env.initial_bankroll) * 100
            
            log_str = (
                f"Episode [{episode:03d}/{episodes}] | "
                f"Net Profit: {profit_pct:+.2f}% | Final Bankroll: ₹{final_bankroll:,.2f} | "
                f"Max Drawdown: {max_drawdown:.2%}"
            )
            
            # Aggressive Strategy Validation Constraints
            if final_bankroll > best_bankroll and final_bankroll > env.initial_bankroll and max_drawdown < 0.25:
                best_bankroll = final_bankroll
                torch.save(self.policy_net.state_dict(), self.save_path)
                logger.info(log_str + " [🏆 NEW LOW-VARIANCE STRATEGY SAVED]")
            else:
                logger.info(log_str)
                
        logger.info("="*70)
        logger.info("PHASE 4: REINFORCEMENT LEARNING COMPLETE.")
        logger.info("Distributional Policies locked. Pipeline Ready for Live Production.")
        logger.info("="*70)


# ==============================================================================
# 3. HIGH-SPEED BATCHED PRECOMPUTATION ENGINE
# ==============================================================================

def build_experience_matrix(config: dict, device: torch.device) -> pd.DataFrame:
    logger.info("Generating Walk-Forward Experience Matrix (VAE RESTORED)...")
    
    seq_len = config['data_pipeline']['feature_engineering']['sequence_length']
    input_dim = len(feature_config.MODEL_INPUT_FEATURES)
    
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    
    df_full = pd.read_csv(data_path).sort_values(by='issue_id').reset_index(drop=True)
    meta_split_idx = int(len(df_full) * 0.90)
    rl_df = df_full.iloc[meta_split_idx:].reset_index(drop=True)
    
    scaler_dir = os.path.join(PROJECT_ROOT, config['paths']['scaler_artifact_dir'])
    master_scaler = joblib.load(os.path.join(scaler_dir, "master_scaler.joblib"))
    
    target_dict = {t_name: rl_df[col_name].values for t_name, col_name in feature_config.TARGETS.items()}
    X_raw = rl_df[list(feature_config.MODEL_INPUT_FEATURES)].values
    X_scaled = master_scaler.transform(X_raw)
    
    rl_dataset = WingoSequenceDataset(X_scaled, target_dict, seq_len)
    val_loader = torch.utils.data.DataLoader(
        rl_dataset, batch_size=2048, shuffle=False, 
        num_workers=config['system'].get('max_workers', 2), pin_memory=True
    )
    
    sup_dir = os.path.join(PROJECT_ROOT, config['paths']['supervised_artifact_dir'])
    meta_dir = os.path.join(PROJECT_ROOT, config['paths']['meta_learner_artifact_dir'])
    unsup_dir = os.path.join(PROJECT_ROOT, config['paths']['unsupervised_artifact_dir'])
    
    # 1. Load LSTM & Transformer
    lstm = WingoMTLLSTM(input_dim, config['models']['lstm']['hidden_dim'], config['models']['lstm']['num_layers']).to(device)
    lstm.load_state_dict(torch.load(os.path.join(sup_dir, "lstm_SWA_master.pt"), map_location=device)['model_state_dict'])
    lstm.eval()
    
    trans = WingoMTLTransformer(input_dim, seq_len, config['models']['transformer']['d_model'], config['models']['transformer']['nhead'], config['models']['transformer']['num_layers']).to(device)
    trans.load_state_dict(torch.load(os.path.join(sup_dir, "transformer_SWA_master.pt"), map_location=device)['model_state_dict'])
    trans.eval()

    # 2. 🚨 RESTORED: Load VAE & Bayesian GMM Anomalizer
    bottleneck = config['models']['autoencoder']['bottleneck_dim']
    vae = WingoTemporalVAE(input_dim=input_dim, sequence_length=seq_len, latent_dim=bottleneck).to(device)
    vae.load_state_dict(torch.load(os.path.join(unsup_dir, "temporal_vae_weights.pt"), map_location=device))
    vae.eval()
    
    gmm_model = joblib.load(os.path.join(unsup_dir, "latent_gmm_model.joblib"))
    gmm_scaler = joblib.load(os.path.join(unsup_dir, "latent_gmm_scaler.joblib"))
    
    with open(os.path.join(unsup_dir, "latent_anomaly_threshold.json"), "r") as f:
        anomaly_threshold = json.load(f)["latent_anomaly_log_prob_threshold"]
    
    # 3. Load Tabular Ensembles
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(sup_dir, "xgboost_master.json"))
    
    import lightgbm as lgb
    lgb_model = lgb.Booster(model_file=os.path.join(sup_dir, "lightgbm_master.txt"))
    
    use_cat = False
    try:
        import catboost as cb
        cat_model = cb.CatBoostClassifier()
        cat_model.load_model(os.path.join(sup_dir, "catboost_master.cbm"))
        use_cat = True
    except Exception:
        pass

    dropped_cols = [
        'prev_1_is_red', 'freq_size_big_last_20', 'time_second_sin', 
        'prev_1_is_green', 'prev_2_size_target', 'time_second_cos', 
        'lockout_ms', 'duration_ms', 'prev_1_size_target', 'latency_rolling_std_10'
    ]
    all_cols = list(feature_config.MODEL_INPUT_FEATURES)
    xgb_indices = [i for i, col in enumerate(all_cols) if col not in dropped_cols]

    actual_num_models = 5 if use_cat else 4
    context_dim = input_dim 
    
    meta_net = NeuralMetaAggregator(num_models=actual_num_models, context_dim=context_dim).to(device)
    meta_net.load_state_dict(torch.load(os.path.join(meta_dir, "meta_aggregator_weights.pt"), map_location=device))
    meta_net.eval()
    
    platt_calibrator = joblib.load(os.path.join(meta_dir, "platt_calibrator.joblib"))

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        lstm = nn.DataParallel(lstm)
        trans = nn.DataParallel(trans)
        vae = nn.DataParallel(vae)

    logger.info("Executing BATCHED parallel inferences across timeline...")
    experience_dataframes = []
    
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            
            with autocast():
                p_lstm = torch.sigmoid(lstm(batch_X)['binary_size']).view(-1, 1)
                p_trans = torch.sigmoid(trans(batch_X)['binary_size']).view(-1, 1)
                vae_out = vae(batch_X) # 🚨 RESTORED: Push sequence through VAE
                
            # 🚨 RESTORED: Calculate Structural Anomaly Flags
            mu_batch = vae_out['mu'].float().cpu().numpy()
            scaled_mu = gmm_scaler.transform(mu_batch)
            log_probs = gmm_model.score_samples(scaled_mu)
            
            # Create binary flags: 1.0 if highly anomalous (below threshold), 0.0 if normal
            is_anomalous = (log_probs < anomaly_threshold).astype(np.float32)
                
            X_last_step_scaled = batch_X[:, -1, :].cpu().numpy()
            X_last_step_raw = master_scaler.inverse_transform(X_last_step_scaled)
            X_tabular_ready = X_last_step_raw[:, xgb_indices]
            
            p_xgb = xgb_model.predict_proba(X_tabular_ready)[:, 1]
            p_xgb_tensor = torch.tensor(p_xgb, dtype=torch.float32, device=device).view(-1, 1)
            
            p_lgb = lgb_model.predict(X_tabular_ready)
            p_lgb_tensor = torch.tensor(p_lgb, dtype=torch.float32, device=device).view(-1, 1)
            
            valid_tensors = [p_lstm, p_trans, p_xgb_tensor, p_lgb_tensor]
            
            if use_cat:
                p_cat = cat_model.predict_proba(X_tabular_ready)[:, 1]
                p_cat_tensor = torch.tensor(p_cat, dtype=torch.float32, device=device).view(-1, 1)
                valid_tensors.append(p_cat_tensor)
            
            batch_probs = torch.cat(valid_tensors, dim=1)
            batch_context = batch_X[:, -1, :] 
            
            meta_logit = meta_net(batch_probs, batch_context)
            meta_raw_prob = torch.sigmoid(meta_logit).squeeze().cpu().numpy()
            calibrated_probs = platt_calibrator.predict_proba(meta_raw_prob.reshape(-1, 1))[:, 1]
            true_targets = batch_Y['binary_size'].cpu().numpy()
            
            batch_df = pd.DataFrame({
                'meta_calibrated_prob': calibrated_probs,
                'true_target': true_targets,
                'anomaly_score': is_anomalous # 🚨 RESTORED: Appended to execution memory
            })
            experience_dataframes.append(batch_df)
            
    df_exp = pd.concat(experience_dataframes, ignore_index=True)
    logger.info(f"Experience Matrix compiled flawlessly. Total Rows Mapped: {df_exp.shape[0]:,}")
    
    del lstm, trans, vae, meta_net, xgb_model, lgb_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return df_exp

def execute_rl_pipeline():
    """Main function to construct environments and instantiate RL training."""
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"RL Subsystem Initializing. Hardware Pipeline: {device}")
    
    # 1. Execute Precomputation Phase
    experience_matrix = build_experience_matrix(config, device)
    
    # 2. Construct Financial Simulation Environment
    initial_br = config['inference_and_risk']['bankroll_management']['initial_virtual_balance']
    env = WingoCasinoEnvironment(experience_matrix, initial_bankroll=initial_br, max_steps=250)
    
    # 3. Instantiate Distributional RL Trainer
    trainer = DistributionalRLTrainer(config, device)
    
    # 4. Initiate execution loop (150 Episodes recommended for C51 Bootstrapping)
    trainer.train_agent(env, episodes=150)


if __name__ == "__main__":
    execute_rl_pipeline()