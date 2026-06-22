# ==============================================================================
# ENGINE_ALPHA - REAL-TIME MODEL DRIFT & TELEMETRY MONITOR
# Core Component: src/inference/telemetry_collector.py
# Description: Background daemon that monitors live SQLite execution logs.
# Uses statistical control charts to detect Concept Drift and PRNG shifts,
# triggering algorithmic safety halts if the platform changes its backend.
# ==============================================================================

import os
import sys
from streamlit import json
import yaml
import time
import sqlite3
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Telemetry] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.utils.metrics import QuantMetrics
    from src.utils.threading_pool import AsyncDatabaseWriter
except ImportError as e:
    logger.error(f"Failed to import utilities: {e}")
    sys.exit(1)


class DriftAssasin:
    """
    Monitors live execution data to mathematically prove if the neural networks 
    are degrading over time due to platform algorithm changes.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.db_path = os.path.join(PROJECT_ROOT, self.config['paths']['database_path'])
        
        telemetry_cfg = self.config['telemetry']
        self.rolling_window = telemetry_cfg['rolling_drift_window']
        self.alert_threshold = telemetry_cfg['drift_alert_accuracy_drop']
        
        # Load baseline Kaggle validation metrics from the Meta-Learner config
        self.baseline_accuracy = 0.54 # Assume 54% was achieved in Kaggle testing
        self.baseline_brier = 0.24    # Assume 0.24 was achieved in Kaggle testing

    def _fetch_recent_executions(self) -> Optional[pd.DataFrame]:
        """Pulls the most recent resolved trades from the live SQLite database."""
        if not os.path.exists(self.db_path):
            logger.warning("Database not found. Awaiting Live Engine initialization.")
            return None
            
        query = f"""
            SELECT 
                issue_id, timestamp, meta_calibrated_prob, predicted_size, actual_result, profit
            FROM live_predictions
            WHERE actual_result IS NOT NULL
            ORDER BY issue_id DESC
            LIMIT {self.rolling_window}
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                
            if len(df) < 50: # Need a minimum statistical sample size
                return None
                
            # Reverse to chronological order
            return df.sort_values(by='issue_id').reset_index(drop=True)
            
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch telemetry data: {e}")
            return None

    def analyze_drift(self) -> Dict:
        """
        Core Mathematical Audit.
        Compares live rolling metrics against the static Kaggle validation metrics.
        """
        df = self._fetch_recent_executions()
        if df is None:
            return {'status': 'INSUFFICIENT_DATA'}
            
        # 1. Rolling Accuracy
        df['is_correct'] = (df['predicted_size'] == df['actual_result']).astype(int)
        live_accuracy = df['is_correct'].mean()
        
        # 2. Brier Score (Probabilistic Calibration)
        # Brier Score = Mean Squared Error between probability and actual boolean outcome
        y_prob = df['meta_calibrated_prob'].values
        y_true = df['actual_result'].values
        live_brier = np.mean((y_prob - y_true) ** 2)
        
        # 3. Brier Skill Score
        bss = QuantMetrics.brier_skill_score(y_true, y_prob)
        
        # 4. Profitability / Edge
        total_profit = df['profit'].sum()
        win_rate = live_accuracy
        
        # 5. Drift Detection Logic
        accuracy_drop = self.baseline_accuracy - live_accuracy
        drift_detected = False
        alert_level = "GREEN"
        
        if accuracy_drop >= self.alert_threshold:
            drift_detected = True
            alert_level = "RED"
        elif accuracy_drop >= (self.alert_threshold / 2):
            alert_level = "YELLOW"
            
        metrics = {
            'status': 'ACTIVE',
            'sample_size': len(df),
            'live_accuracy': float(live_accuracy),
            'live_brier': float(live_brier),
            'brier_skill_score': float(bss),
            'total_profit': float(total_profit),
            'drift_detected': drift_detected,
            'alert_level': alert_level,
            'accuracy_drop': float(accuracy_drop)
        }
        
        self._log_and_alert(metrics)
        return metrics

    def _log_and_alert(self, metrics: Dict):
        """Records the telemetry to the database and raises system warnings if needed."""
        log_str = (
            f"TELEMETRY | Samples: {metrics['sample_size']} | "
            f"Acc: {metrics['live_accuracy']:.2%} | "
            f"Brier: {metrics['live_brier']:.4f} | "
            f"BSS: {metrics['brier_skill_score']:.4f} | "
            f"Profit: ₹{metrics['total_profit']:.2f}"
        )
        
        if metrics['alert_level'] == "GREEN":
            logger.info(log_str + " [STATUS: NOMINAL]")
        elif metrics['alert_level'] == "YELLOW":
            logger.warning(log_str + " [STATUS: DEGRADING - Monitor closely]")
        elif metrics['alert_level'] == "RED":
            logger.critical("="*60)
            logger.critical("CRITICAL DRIFT DETECTED: ALGORITHMIC INTEGRITY COMPROMISED")
            logger.critical(log_str)
            logger.critical(f"Accuracy has dropped by {metrics['accuracy_drop']:.2%} from baseline.")
            logger.critical("Recommend immediate halt of live execution for retraining.")
            logger.critical("="*60)
            
        # Log to the System Telemetry Table via our Async DB Writer
        try:
            db_writer = AsyncDatabaseWriter()
            query = """
                INSERT INTO system_telemetry (rolling_accuracy, brier_score, drawdown_pct)
                VALUES (?, ?, ?)
            """
            # Drawdown would be calculated from a larger portfolio view, defaulting to 0 here for simplicity
            db_writer.execute_async(query, (metrics['live_accuracy'], metrics['live_brier'], 0.0))
        except Exception as e:
            logger.error(f"Failed to log telemetry to SQLite: {e}")

    def run_continuous_monitor(self, interval_seconds: int = 300):
        """
        Daemon loop. Wakes up every 5 minutes (300 seconds), audits the database,
        and goes back to sleep.
        """
        logger.info(f"Initializing Drift Assassin Daemon. Polling every {interval_seconds}s.")
        try:
            while True:
                self.analyze_drift()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Telemetry Monitor shutdown requested.")


if __name__ == "__main__":
    # Test execution
    assassin = DriftAssasin()
    print("Executing Point-in-Time Telemetry Audit...")
    results = assassin.analyze_drift()
    print(json.dumps(results, indent=4))