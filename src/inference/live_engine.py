# ==============================================================================
# ENGINE_ALPHA - AUTONOMOUS LIVE INFERENCE ORCHESTRATOR
# Core Component: src/inference/live_engine.py
# Description: The Grand Master Loop. Synchronizes with the live platform clock,
# orchestrates data through the FeatureFactory, runs Multi-Task and Unsupervised 
# inferences, evaluates Risk via D3QN, and manages stateful Virtual Bankroll.
# ==============================================================================

import os
import sys
import yaml
import time
import json
import logging
import datetime
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn.functional as F
import xgboost as xgb

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [LiveEngine] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Resolve Root and Imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
    from src.data.feature_factory import FeatureFactory
    from src.models.lstm_brain import WingoMTLLSTM
    from src.models.transformer_brain import WingoMTLTransformer
    from src.models.autoencoder import WingoTemporalVAE
    from src.training.train_meta_learner import NeuralMetaAggregator
    from src.models.dqn_agent import DuelingNoisyDQNBrain
    
    # Utilities
    from extras.web_scraper import WinGoLiveScraper
    from src.utils.threading_pool import AsyncDatabaseWriter
    from src.inference.telemetry_collector import DriftAssasin
except ImportError as e:
    logger.critical(f"Failed to import project modules. Ensure phase 1-4 files exist: {e}")
    sys.exit(1)


class LiveInferenceEngine:
    """
    The Grand Orchestrator. 
    Loads all frozen mathematical artifacts and runs the infinite 30-second execution cycle.
    """
    def __init__(self, config_path: str = None):
        logger.info("="*60)
        logger.info("ENGINE_ALPHA LIVE INFERENCE INITIALIZATION SEQUENCE STARTED")
        logger.info("="*60)
        
        # 1. Load Configurations
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['system']['device'] if torch.cuda.is_available() else "cpu")
        logger.info(f"Authorized Compute Engine: {self.device}")
        
        # 2. Financial & Risk Parameters
        risk_cfg = self.config['inference_and_risk']['bankroll_management']
        self.initial_bankroll = risk_cfg['initial_virtual_balance']
        self.current_bankroll = self.initial_bankroll
        self.max_bankroll = self.initial_bankroll
        self.max_drawdown_stop = risk_cfg['maximum_drawdown_hard_stop']
        self.payout_multiplier = 0.96 # Standard 4% house edge fee
        self.win_streak = 0
        
        # RL Action Mapping
        self.action_space = {0: 0.00, 1: 0.01, 2: 0.025, 3: 0.05, 4: 0.10}
        
        # 3. System Timers
        self.loop_frequency = self.config['inference_and_risk']['live_loop_frequency_seconds']
        self.safety_buffer = self.config['inference_and_risk']['execution_safety_buffer_seconds']
        
        # 4. Instantiate Utility Engines
        self.scraper = WinGoLiveScraper(config_path)
        self.factory = FeatureFactory(config_path)
        self.db_writer = AsyncDatabaseWriter()
        self.drift_assassin = DriftAssasin(config_path)
        
        # 5. Load Mathematical Artifacts
        self.seq_len = self.config['data_pipeline']['feature_engineering']['sequence_length']
        self.input_dim = len(feature_config.MODEL_INPUT_FEATURES)
        self._load_frozen_artifacts()
        
        logger.info("="*60)
        logger.info("ALL SYSTEMS GREEN. ENGINE_ALPHA IS ARMED AND READY.")
        logger.info("="*60)

    def _load_frozen_artifacts(self):
        """Massive memory-mapping function to load 8 separate intelligence files into VRAM."""
        logger.info("Accessing Model Artifact Vaults...")
        
        sup_dir = os.path.join(PROJECT_ROOT, self.config['paths']['supervised_artifact_dir'])
        unsup_dir = os.path.join(PROJECT_ROOT, self.config['paths']['unsupervised_artifact_dir'])
        meta_dir = os.path.join(PROJECT_ROOT, self.config['paths']['meta_learner_artifact_dir'])
        rl_dir = os.path.join(PROJECT_ROOT, self.config['paths']['reinforcement_artifact_dir'])
        scaler_dir = os.path.join(PROJECT_ROOT, self.config['paths']['scaler_artifact_dir'])
        
        # 1. The Scaler (Required to normalize live data exactly like training data)
        self.scaler = joblib.load(os.path.join(scaler_dir, "master_scaler.joblib"))
        
        # 2. LSTM Brain
        self.lstm = WingoMTLLSTM(self.input_dim, self.config['models']['lstm']['hidden_dim'], self.config['models']['lstm']['num_layers']).to(self.device)
        self.lstm.load_state_dict(torch.load(os.path.join(sup_dir, "lstm_best_weights.pt"), map_location=self.device)['model_state_dict'])
        self.lstm.eval()
        
        # 3. Transformer Brain
        self.transformer = WingoMTLTransformer(self.input_dim, self.seq_len, self.config['models']['transformer']['d_model'], self.config['models']['transformer']['nhead'], self.config['models']['transformer']['num_layers']).to(self.device)
        self.transformer.load_state_dict(torch.load(os.path.join(sup_dir, "transformer_best_weights.pt"), map_location=self.device)['model_state_dict'])
        self.transformer.eval()
        
        # 4. XGBoost Brain
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(os.path.join(sup_dir, "xgboost_master.json"))
        
        # 5. Temporal VAE (Anomaly Detector)
        self.vae = WingoTemporalVAE(self.input_dim, self.seq_len, self.config['models']['autoencoder']['bottleneck_dim']).to(self.device)
        self.vae.load_state_dict(torch.load(os.path.join(unsup_dir, "temporal_vae_weights.pt"), map_location=self.device))
        self.vae.eval()
        
        with open(os.path.join(unsup_dir, "anomaly_threshold.json"), "r") as f:
            self.vae_anomaly_threshold = json.load(f)['anomaly_threshold_mse']
            
        # 6. Neural Meta-Aggregator & Platt Calibrator
        self.meta_net = NeuralMetaAggregator(num_models=4).to(self.device)
        self.meta_net.load_state_dict(torch.load(os.path.join(meta_dir, "meta_aggregator_weights.pt"), map_location=self.device))
        self.meta_net.eval()
        self.platt_calibrator = joblib.load(os.path.join(meta_dir, "platt_calibrator.joblib"))
        
        # 7. D3QN Policy Network
        self.dqn = DuelingNoisyDQNBrain(self.config['models']['dqn']['state_dim'], self.config['models']['dqn']['action_dim']).to(self.device)
        self.dqn.load_state_dict(torch.load(os.path.join(rl_dir, "dqn_policy_weights.pt"), map_location=self.device))
        self.dqn.eval() # Eval mode disables exploration noise for deterministic execution
        
        logger.info("[✓] 8/8 Artifacts Loaded Successfully into VRAM.")

    def _execute_deep_inference(self, live_df: pd.DataFrame) -> dict:
        """
        The Core Calculation Engine.
        Processes the raw scraper dataframe through the FeatureFactory, scales it, 
        and extracts the ultimate probabilistic edge.
        """
        # 1. Feature Engineering
        engineered_df = self.factory.build_features(live_df)
        
        if len(engineered_df) < self.seq_len:
            logger.error(f"Insufficient engineered rows ({len(engineered_df)}) to build sequence of length {self.seq_len}.")
            return None
            
        # 2. Extract the most recent Sequence Window
        # We need exactly `seq_len` rows. The last row represents the immediate past (t-1).
        sequence_window = engineered_df.iloc[-self.seq_len:].copy()
        target_issue_id = sequence_window.iloc[-1]['issue_id'] + 1 # We are predicting the NEXT issue
        
        # Extract purely the feature matrix defined in schema
        raw_features = sequence_window[feature_config.MODEL_INPUT_FEATURES].values
        
        # 3. Apply Standard Scaler
        scaled_features = self.scaler.transform(raw_features)
        
        # Convert to PyTorch Tensor: Shape (1, Seq_Len, Features)
        x_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 4. Base Model Inferences
        with torch.no_grad():
            p_lstm = torch.sigmoid(self.lstm(x_tensor)['binary_size']).item()
            p_trans = torch.sigmoid(self.transformer(x_tensor)['binary_size']).item()
            
            # XGBoost only needs the latest row (t-1)
            x_flat = scaled_features[-1].reshape(1, -1)
            p_xgb = self.xgb.predict_proba(x_flat)[0][1]
            
            # VAE Anomaly Extraction
            vae_out = self.vae(x_tensor)
            vae_error = F.mse_loss(vae_out['reconstructed'], x_tensor).item()
            
            # 5. Meta-Aggregation
            meta_input = torch.tensor([[p_lstm, p_trans, p_xgb, vae_error]], dtype=torch.float32, device=self.device)
            meta_logit = self.meta_net(meta_input)
            raw_meta_prob = torch.sigmoid(meta_logit).item()
            
            # 6. Platt Scaling Calibration
            calibrated_prob = self.platt_calibrator.predict_proba([[raw_meta_prob]])[0][1]
            
        logger.debug(f"Deep Inference OK. Meta-Prob: {calibrated_prob:.4f}")
        
        return {
            'target_issue_id': target_issue_id,
            'lstm_prob': p_lstm,
            'trans_prob': p_trans,
            'xgb_prob': p_xgb,
            'vae_error': vae_error,
            'meta_calibrated_prob': calibrated_prob
        }

    def _calculate_rl_action(self, inference_data: dict) -> dict:
        """Constructs the financial state vector and queries the D3QN for a bet size."""
        meta_prob = inference_data['meta_calibrated_prob']
        vae_score = min(inference_data['vae_error'] / self.vae_anomaly_threshold, 2.0)
        bankroll_ratio = self.current_bankroll / self.initial_bankroll
        drawdown = (self.max_bankroll - self.current_bankroll) / self.max_bankroll
        streak = min(self.win_streak / 10.0, 1.0)
        
        edge = (meta_prob * self.payout_multiplier) - (1 - meta_prob)
        kelly = max(0, edge / self.payout_multiplier)
        agreement = np.std([inference_data['lstm_prob'], inference_data['trans_prob'], inference_data['xgb_prob']])
        
        # Build 7-Dim State Vector
        state_vec = np.array([meta_prob, vae_score, bankroll_ratio, drawdown, streak, kelly, agreement], dtype=np.float32)
        state_tensor = torch.tensor(state_vec, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.dqn(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            
        bet_fraction = self.action_space[action_idx]
        bet_amount = self.current_bankroll * bet_fraction
        
        # Convert Probability to Prediction Logic (0 = Small, 1 = Big)
        predicted_size = 1 if meta_prob >= 0.5 else 0
        confidence = meta_prob if predicted_size == 1 else (1.0 - meta_prob)
        
        return {
            'action_idx': action_idx,
            'bet_amount': bet_amount,
            'predicted_size': predicted_size,
            'confidence': confidence
        }

    def _sync_chronometer(self):
        """
        Ensures the bot executes exactly at the top of the 30-second window.
        Prevents drifting out of sync with the casino's actual PRNG draw.
        """
        current_time = time.time()
        # Find exactly how many seconds we are past the nearest 30-second mark
        remainder = current_time % self.loop_frequency
        
        # If we are at remainder 29.5, we wait 0.5 seconds to hit the exact start of the next cycle
        wait_time = self.loop_frequency - remainder
        
        logger.info(f"Synchronizing Chronometer... Waiting {wait_time:.2f}s for the next execution window.")
        time.sleep(wait_time)

    def run_live_loop(self):
        """The Infinite Autonomous Execution Cycle."""
        
        # Initial synchronization with the global clock
        self._sync_chronometer()
        
        try:
            while True:
                loop_start_time = time.time()
                logger.info("-" * 50)
                logger.info(f"⚡ EXECUTING TRADE CYCLE | Bankroll: ₹{self.current_bankroll:,.2f}")
                
                # 1. Scrape Live Network Data
                # We fetch 100 records to ensure we have enough history to build a length 60 sequence
                live_df = self.scraper.fetch_latest_history(limit=100)
                
                if live_df is None or len(live_df) < self.seq_len + 10:
                    logger.warning("Scraper returned insufficient data. Aborting this cycle and awaiting next window.")
                    self._sync_chronometer()
                    continue
                    
                # 2. Check for Risk Safety (Max Drawdown Stop-Loss)
                current_drawdown = (self.max_bankroll - self.current_bankroll) / self.max_bankroll
                if current_drawdown >= self.max_drawdown_stop:
                    logger.critical(f"FATAL: Maximum Drawdown reached ({current_drawdown:.2%}). Halting operations.")
                    break
                    
                # 3. Deep Math Inference
                inf_start = time.time()
                inference = self._execute_deep_inference(live_df)
                if inference is None:
                    self._sync_chronometer()
                    continue
                    
                # 4. Reinforcement Learning Execution
                rl_decision = self._calculate_rl_action(inference)
                
                inf_latency = time.time() - inf_start
                
                # 5. Assemble and Log
                issue_id = inference['target_issue_id']
                target_str = "BIG" if rl_decision['predicted_size'] == 1 else "SMALL"
                
                logger.info(f"🎯 Target Issue: {issue_id} | Confidence: {rl_decision['confidence']:.2%}")
                
                if rl_decision['action_idx'] == 0:
                    logger.info("🛡️ DQN Action: SKIP BET (No mathematical edge found).")
                else:
                    logger.info(f"💸 DQN Action: BET ₹{rl_decision['bet_amount']:.2f} on {target_str}")
                    
                # Fire Async Database Log instantly
                db_payload = {
                    'issue_id': issue_id,
                    'lstm_prob': inference['lstm_prob'],
                    'trans_prob': inference['trans_prob'],
                    'xgb_prob': inference['xgb_prob'],
                    'vae_error': inference['vae_error'],
                    'meta_calibrated_prob': inference['meta_calibrated_prob'],
                    'predicted_size': rl_decision['predicted_size'],
                    'dqn_action': rl_decision['action_idx'],
                    'bet_amount': rl_decision['bet_amount']
                }
                self.db_writer.log_prediction(db_payload)
                
                # ==============================================================
                # 6. THE RESOLUTION PHASE
                # The bet is placed. Now we must wait for the casino to draw the 
                # number, fetch the result, and update our internal bankroll.
                # ==============================================================
                
                # Wait until the 30-second betting window has closed
                time_elapsed = time.time() - loop_start_time
                remaining_wait = self.loop_frequency - time_elapsed
                
                if remaining_wait > 0:
                    logger.debug(f"Inference complete in {time_elapsed:.3f}s. Waiting {remaining_wait:.2f}s for draw...")
                    time.sleep(remaining_wait)
                    
                # Give the casino API a 2-second buffer to update their databases
                time.sleep(2.0)
                
                # Fetch fresh data to see what actually happened
                resolution_df = self.scraper.fetch_latest_history(limit=5)
                
                if resolution_df is not None:
                    # Find the row matching the issue we just bet on
                    resolved_row = resolution_df[resolution_df['issue_id'] == issue_id]
                    
                    if not resolved_row.empty:
                        actual_size = int(resolved_row.iloc[0]['Size_Target'])
                        actual_str = "BIG" if actual_size == 1 else "SMALL"
                        
                        profit = 0.0
                        if rl_decision['bet_amount'] > 0:
                            if rl_decision['predicted_size'] == actual_size:
                                profit = rl_decision['bet_amount'] * self.payout_multiplier
                                self.current_bankroll += profit
                                self.win_streak += 1
                                logger.info(f"✅ WIN! Result: {actual_str}. Profit: +₹{profit:.2f}")
                                if self.current_bankroll > self.max_bankroll:
                                    self.max_bankroll = self.current_bankroll
                            else:
                                profit = -rl_decision['bet_amount']
                                self.current_bankroll += profit
                                self.win_streak = 0
                                logger.warning(f"❌ LOSS! Result: {actual_str}. Loss: -₹{abs(profit):.2f}")
                        else:
                            logger.info(f"⚪ SKIPPED. Result was: {actual_str}.")
                            
                        # Update the SQLite database asynchronously
                        self.db_writer.update_actual_result(issue_id, actual_size, profit)
                    else:
                        logger.warning(f"Could not find resolution for Issue {issue_id} in recent fetch.")
                
                # Synchronize for the next cycle
                self._sync_chronometer()

        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt detected. Initiating Graceful Shutdown...")
            self.db_writer.shutdown()
            logger.info("ENGINE_ALPHA Powered Down.")
            sys.exit(0)
        except Exception as e:
            logger.critical(f"FATAL SYSTEM FAILURE in Live Engine: {e}")
            self.db_writer.shutdown()
            sys.exit(1)


if __name__ == "__main__":
    engine = LiveInferenceEngine()
    engine.run_live_loop()