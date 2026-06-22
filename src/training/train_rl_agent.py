# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE RL TRAINING ENGINE (D3QN)
# Core Component: src/training/train_rl_agent.py
# Description: Financial execution simulator. Trains the Dueling Noisy DQN 
# Agent to maximize bankroll and manage risk using Prioritized Experience Replay 
# and Walk-Forward OOF precomputed probabilities.
# ==============================================================================

import os
import sys
import yaml
import time
import logging
import numpy as np
import pandas as pd
import joblib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

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
    from src.models.dqn_agent import DuelingNoisyDQNBrain, PrioritizedReplayBuffer
    
    # We must import the base models to build the precomputed experience matrix
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
    from src.models.autoencoder import WingoTemporalVAE
    from src.training.train_meta_learner import NeuralMetaAggregator
    import xgboost as xgb
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


# ==============================================================================
# 1. THE FINANCIAL SIMULATION ENVIRONMENT
# ==============================================================================

class WingoCasinoEnvironment:
    """
    OpenAI Gym-style Environment. 
    Simulates the casino's response to the bot's bets, tracking bankroll, 
    drawdown, and penalizing ruin.
    """
    def __init__(self, experience_matrix: pd.DataFrame, initial_bankroll: float = 10000.0):
        self.df = experience_matrix.reset_index(drop=True)
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_bankroll = initial_bankroll
        
        # Action mappings (Percentage of current bankroll to risk)
        self.action_space = {
            0: 0.00,  # No Bet (Skip)
            1: 0.01,  # Risk 1%
            2: 0.025, # Risk 2.5%
            3: 0.05,  # Risk 5%
            4: 0.10   # Risk 10% (Maximum allowable)
        }
        
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.win_streak = 0
        
        # WinGo payout mechanism (Standard 1:1 payout minus 2-4% platform fee)
        # Betting 100 on Big/Small returns 196 (96 profit).
        self.payout_multiplier = 0.96 

    def reset(self):
        """Restarts the simulation episode."""
        self.current_bankroll = self.initial_bankroll
        self.max_bankroll = self.initial_bankroll
        self.current_step = 0
        self.win_streak = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Constructs the 7-Dimensional State Vector for the DQN.
        Matches global_config.yaml state_dim exactly.
        """
        row = self.df.iloc[self.current_step]
        
        # 1. Meta-Aggregator Calibrated Probability (0.0 to 1.0)
        meta_prob = row['meta_calibrated_prob']
        # 2. VAE Anomaly Score (Normalized roughly to 0-1 range for neural nets)
        vae_score = min(row['vae_error'] / row['threshold'], 2.0) 
        # 3. Bankroll Ratio (How rich are we compared to start?)
        bankroll_ratio = self.current_bankroll / self.initial_bankroll
        # 4. Maximum Drawdown (How far are we from our all-time high?)
        drawdown = (self.max_bankroll - self.current_bankroll) / self.max_bankroll
        # 5. Win Streak (Normalized)
        streak = min(self.win_streak / 10.0, 1.0)
        # 6. Kelly Criterion Fraction (Mathematical optimal bet size)
        # Edge = (Probability * Payout) - (1 - Probability)
        edge = (meta_prob * self.payout_multiplier) - (1 - meta_prob)
        kelly = max(0, edge / self.payout_multiplier)
        # 7. Model Agreement (Volatility/Confidence indicator)
        agreement = np.std([row['lstm_prob'], row['trans_prob'], row['xgb_prob']])
        
        state = np.array([meta_prob, vae_score, bankroll_ratio, drawdown, streak, kelly, agreement], dtype=np.float32)
        return state

    def step(self, action_idx: int) -> tuple:
        """
        Executes the bot's action, advances time, and calculates the reward.
        Returns: (next_state, reward, done, info)
        """
        row = self.df.iloc[self.current_step]
        
        meta_prob = row['meta_calibrated_prob']
        true_outcome = row['true_target'] # 1 for Big, 0 for Small
        
        # Agent bets in the direction of the Meta-Learner
        bot_prediction = 1 if meta_prob >= 0.5 else 0
        
        bet_fraction = self.action_space[action_idx]
        bet_amount = self.current_bankroll * bet_fraction
        
        reward = 0.0
        
        # 1. Financial Execution
        if bet_amount > 0:
            if bot_prediction == true_outcome:
                # WIN
                profit = bet_amount * self.payout_multiplier
                self.current_bankroll += profit
                self.win_streak += 1
                reward = profit / self.initial_bankroll # Normalize reward scaling for the neural network
            else:
                # LOSS
                self.current_bankroll -= bet_amount
                self.win_streak = 0
                reward = -bet_amount / self.initial_bankroll
        else:
            # NO BET: Penalize slightly if the bot ignored a massive mathematical edge
            edge = (max(meta_prob, 1-meta_prob) * self.payout_multiplier) - (1 - max(meta_prob, 1-meta_prob))
            if edge > 0.05:
                reward = -0.005 # Opportunity cost penalty
                
        # 2. Bankroll Tracking
        if self.current_bankroll > self.max_bankroll:
            self.max_bankroll = self.current_bankroll
            
        # 3. Check for Bankruptcy (Ruin)
        done = False
        if self.current_bankroll < (self.initial_bankroll * 0.10): # 90% drawdown triggers stop-loss
            reward = -10.0 # Massive ruin penalty
            done = True
            
        # advance time
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        next_state = self._get_state()
        
        info = {'bankroll': self.current_bankroll, 'action': action_idx}
        return next_state, reward, done, info


# ==============================================================================
# 2. THE D3QN ALGORITHMIC TRAINER
# ==============================================================================

class RLBotTrainer:
    """
    Handles Prioritized Experience Replay, Double Q-Learning Loss, 
    and Polyak Target Syncing.
    """
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        rl_cfg = self.config['models']['dqn']
        self.state_dim = rl_cfg['state_dim']
        self.action_dim = rl_cfg['action_dim']
        self.gamma = rl_cfg['gamma']
        self.batch_size = 64
        self.tau = 0.005 # Polyak Averaging constant for soft updates
        
        # Double DQN Architecture requires a Policy Net (Actors) and a Target Net (Evaluators)
        self.policy_net = DuelingNoisyDQNBrain(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingNoisyDQNBrain(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network never trains directly
        
        self.optimizer = AdamW(self.policy_net.parameters(), lr=rl_cfg['learning_rate'], amsgrad=True)
        self.memory = PrioritizedReplayBuffer(capacity=rl_cfg['memory_capacity'])
        
        self.artifact_dir = os.path.join(PROJECT_ROOT, self.config['paths']['reinforcement_artifact_dir'])
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.save_path = os.path.join(self.artifact_dir, "dqn_policy_weights.pt")

    def select_action(self, state: np.ndarray) -> int:
        """
        NoisyNets handle exploration automatically via parameter noise.
        We simply take the argmax of the policy network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action

    def optimize_step(self):
        """
        The Core RL Mathematical Update.
        Samples from PER, computes Bellman Error, and updates network weights.
        """
        if len(self.memory) < self.batch_size:
            return 0.0 # Burn-in period (wait until buffer has enough memories)
            
        # 1. Sample from Prioritized Replay Buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        weights = weights.to(self.device).unsqueeze(1)
        
        # Reset NoisyNet parameters to generate fresh exploration vectors
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        # 2. Compute Current Q-Values
        # Gather the Q-value specifically for the action that was actually taken
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 3. Compute Target Q-Values (Double DQN Logic)
        with torch.no_grad():
            # Policy net decides the *best action* for the next state
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates the *value* of that chosen action
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Bellman Equation
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # 4. Calculate TD-Error & Loss
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        # Huber Loss (Smooth L1) is extremely robust to outliers in financial data
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        # Multiply by PER Importance Sampling weights and average
        loss = (loss * weights).mean()
        
        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 6. Update PER Priorities based on the new TD-Errors
        self.memory.update_priorities(indices, td_errors)
        
        # 7. Soft Update Target Network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()

    def train_agent(self, env: WingoCasinoEnvironment, episodes: int = 50):
        logger.info("="*60)
        logger.info(f"INITIATING ALGORITHMIC EXECUTION TRAINING (D3QN)")
        logger.info(f"Target Device: {self.device} | Episodes: {episodes}")
        logger.info("="*60)
        
        best_bankroll = 0.0
        
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            total_loss = 0.0
            steps = 0
            
            while True:
                # Agent takes action
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Push memory to PER Buffer
                self.memory.push(
                    torch.tensor(state, dtype=torch.float32),
                    action,
                    reward,
                    torch.tensor(next_state, dtype=torch.float32),
                    done
                )
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Optimize
                loss = self.optimize_step()
                total_loss += loss
                
                if done:
                    break
                    
            final_bankroll = info['bankroll']
            profit_pct = ((final_bankroll - env.initial_bankroll) / env.initial_bankroll) * 100
            
            log_str = (
                f"Episode [{episode}/{episodes}] | Steps: {steps} | "
                f"Net Profit: {profit_pct:+.2f}% | "
                f"Final Bankroll: ₹{final_bankroll:,.2f}"
            )
            
            # Save the agent if it figured out a highly profitable strategy without going bankrupt
            if final_bankroll > best_bankroll and final_bankroll > env.initial_bankroll:
                best_bankroll = final_bankroll
                torch.save(self.policy_net.state_dict(), self.save_path)
                logger.info(log_str + " [🏆 NEW TRADING STRATEGY SAVED]")
            else:
                logger.info(log_str)
                
        logger.info("="*60)
        logger.info("PHASE 4: REINFORCEMENT LEARNING COMPLETE.")
        logger.info("All Deep Learning and Trading Policy Artifacts are now strictly locked.")
        logger.info("="*60)


# ==============================================================================
# 3. PRECOMPUTATION ENGINE
# Runs the entire dataset through all frozen models to build the Experience Matrix
# ==============================================================================

def build_experience_matrix(config: dict, device: torch.device) -> pd.DataFrame:
    logger.info("Generating Walk-Forward Experience Matrix. Spinning up Base Brains...")
    
    seq_len = config['data_pipeline']['feature_engineering']['sequence_length']
    input_dim = len(feature_config.MODEL_INPUT_FEATURES)
    
    # 1. Load Data (Strictly Validation Set to simulate Out-Of-Sample trading)
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_path'])
    df = pd.read_csv(data_path).sort_values(by='issue_id').reset_index(drop=True).dropna()
    split_idx = int(len(df) * config['data_pipeline']['preprocessing']['train_test_split_ratio'])
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    scaler_path = os.path.join(PROJECT_ROOT, config['paths']['scaler_artifact_dir'], "master_scaler.joblib")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(val_df[feature_config.MODEL_INPUT_FEATURES].values)
    Y_true = val_df[feature_config.TARGETS['binary_size']].values
    
    val_dataset = WingoSequenceDataset(X_scaled, {'binary_size': Y_true}, seq_len)
    
    # 2. Load Frozen Brains
    sup_dir = os.path.join(PROJECT_ROOT, config['paths']['supervised_artifact_dir'])
    unsup_dir = os.path.join(PROJECT_ROOT, config['paths']['unsupervised_artifact_dir'])
    meta_dir = os.path.join(PROJECT_ROOT, config['paths']['meta_learner_artifact_dir'])
    
    # Base Models
    lstm = WingoMTLLSTM(input_dim, config['models']['lstm']['hidden_dim'], config['models']['lstm']['num_layers']).to(device)
    lstm.load_state_dict(torch.load(os.path.join(sup_dir, "lstm_best_weights.pt"), map_location=device)['model_state_dict'])
    lstm.eval()
    
    trans = WingoMTLTransformer(input_dim, seq_len, config['models']['transformer']['d_model'], config['models']['transformer']['nhead'], config['models']['transformer']['num_layers']).to(device)
    trans.load_state_dict(torch.load(os.path.join(sup_dir, "transformer_best_weights.pt"), map_location=device)['model_state_dict'])
    trans.eval()
    
    vae = WingoTemporalVAE(input_dim, seq_len, config['models']['autoencoder']['bottleneck_dim']).to(device)
    vae.load_state_dict(torch.load(os.path.join(unsup_dir, "temporal_vae_weights.pt"), map_location=device))
    vae.eval()
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(sup_dir, "xgboost_master.json"))
    
    # Meta Aggregator & Calibrator
    meta_net = NeuralMetaAggregator(num_models=4).to(device)
    meta_net.load_state_dict(torch.load(os.path.join(meta_dir, "meta_aggregator_weights.pt"), map_location=device))
    meta_net.eval()
    
    platt_calibrator = joblib.load(os.path.join(meta_dir, "platt_calibrator.joblib"))
    
    with open(os.path.join(unsup_dir, "anomaly_threshold.json"), "r") as f:
        anomaly_threshold = json.load(f)['anomaly_threshold_mse']

    # 3. Precompute Loop
    logger.info("Executing parallel inferences across timeline...")
    experience_data = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            x_seq, y_dict = val_dataset[i]
            x_tensor = x_seq.unsqueeze(0).to(device)
            
            # Base Inferences
            p_lstm = torch.sigmoid(lstm(x_tensor)['binary_size']).item()
            p_trans = torch.sigmoid(trans(x_tensor)['binary_size']).item()
            p_xgb = xgb_model.predict_proba(x_seq[-1].numpy().reshape(1, -1))[0][1]
            vae_err = F.mse_loss(vae(x_tensor)['reconstructed'], x_tensor).item()
            
            # Meta Inference
            meta_input = torch.tensor([[p_lstm, p_trans, p_xgb, vae_err]], dtype=torch.float32, device=device)
            meta_logit = meta_net(meta_input)
            meta_raw_prob = torch.sigmoid(meta_logit).item()
            
            # Platt Scaling Calibration
            calibrated_prob = platt_calibrator.predict_proba([[meta_raw_prob]])[0][1]
            
            experience_data.append({
                'lstm_prob': p_lstm,
                'trans_prob': p_trans,
                'xgb_prob': p_xgb,
                'vae_error': vae_err,
                'threshold': anomaly_threshold,
                'meta_calibrated_prob': calibrated_prob,
                'true_target': y_dict['binary_size'].item()
            })
            
    df_exp = pd.DataFrame(experience_data)
    logger.info(f"Experience Matrix compiled. Size: {df_exp.shape}")
    return df_exp


def execute_rl_pipeline():
    config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    
    # 1. Build Precomputed Environment Data
    experience_matrix = build_experience_matrix(config, device)
    
    # 2. Instantiate Casino Environment
    # We pass the virtual bankroll limit set in global_config
    initial_br = config['inference_and_risk']['bankroll_management']['initial_virtual_balance']
    env = WingoCasinoEnvironment(experience_matrix, initial_bankroll=initial_br)
    
    # 3. Train D3QN Bot
    trainer = RLBotTrainer(config, device)
    trainer.train_agent(env, episodes=50) # 50 simulated walk-forwards


if __name__ == "__main__":
    execute_rl_pipeline()