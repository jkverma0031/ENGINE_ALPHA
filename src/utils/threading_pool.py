# ==============================================================================
# ENGINE_ALPHA - NON-BLOCKING ASYNC DATABASE POOL
# Core Component: src/utils/threading_pool.py
# Description: Dedicated background worker queue for I/O operations. 
# Ensures that saving telemetry and logs to the SQLite database never blocks 
# or delays the 30-second real-time inference loop.
# ==============================================================================

import os
import sys
import sqlite3
import logging
import threading
import queue
import time
import yaml
from contextlib import contextmanager

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AsyncDB] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class AsyncDatabaseWriter:
    """
    Singleton Background Database Writer.
    Uses the Producer-Consumer pattern. The main inference loop acts as the 
    Producer, instantly dropping SQL commands into the RAM queue. A dedicated 
    Consumer thread reads the queue and executes hard-drive writes in the background.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Thread-safe Singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AsyncDatabaseWriter, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Sets up the database connection and the background worker thread."""
        config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.db_path = os.path.join(PROJECT_ROOT, self.config['paths']['database_path'])
        except Exception:
            self.db_path = os.path.join(PROJECT_ROOT, "database", "wingo_history.db")
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # The thread-safe Queue (Infinite size)
        self.sql_queue = queue.Queue()
        
        # Control flags
        self.is_running = True
        
        # Execute Table Generation synchronously on startup
        self._create_schema_if_missing()
        
        # Start the background Daemon Thread
        # Daemon=True ensures the thread is killed automatically when the main script ends
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background SQLite Thread initialized and waiting for commands.")

    @contextmanager
    def _get_connection(self):
        """Context manager for safe database connections."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            yield conn
        finally:
            conn.close()

    def _create_schema_if_missing(self):
        """Generates the required tables for ENGINE_ALPHA telemetry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Live Predictions Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_predictions (
                    issue_id INTEGER PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    lstm_prob REAL,
                    trans_prob REAL,
                    xgb_prob REAL,
                    vae_error REAL,
                    meta_calibrated_prob REAL,
                    predicted_size INTEGER,
                    dqn_action INTEGER,
                    bet_amount REAL,
                    actual_result INTEGER NULL,
                    profit REAL NULL
                )
            """)
            
            # Table 2: Model Drift & Telemetry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    rolling_accuracy REAL,
                    brier_score REAL,
                    current_bankroll REAL,
                    drawdown_pct REAL,
                    latency_ms REAL
                )
            """)
            conn.commit()

    def _worker_loop(self):
        """
        The continuous background loop. 
        Pops SQL statements from the RAM queue and commits them to the hard drive.
        """
        while self.is_running:
            try:
                # Block for up to 1 second waiting for an item
                task = self.sql_queue.get(timeout=1.0)
                
                if task is None: # Poison pill to shut down gracefully
                    break
                    
                query, parameters = task
                
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, parameters)
                    conn.commit()
                    
                self.sql_queue.task_done()
                
            except queue.Empty:
                continue # No tasks, just loop and wait
            except sqlite3.Error as e:
                logger.error(f"SQLite Write Error in background thread: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in DB worker thread: {e}")

    def execute_async(self, query: str, parameters: tuple = ()):
        """
        Public method used by the Live Engine.
        This function returns INSTANTLY (nanoseconds), passing the heavy lifting 
        to the background worker.
        """
        self.sql_queue.put((query, parameters))

    def log_prediction(self, data: dict):
        """Helper specifically formatted for the Live Engine."""
        query = """
            INSERT INTO live_predictions 
            (issue_id, lstm_prob, trans_prob, xgb_prob, vae_error, meta_calibrated_prob, predicted_size, dqn_action, bet_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            data.get('issue_id'),
            data.get('lstm_prob'),
            data.get('trans_prob'),
            data.get('xgb_prob'),
            data.get('vae_error'),
            data.get('meta_calibrated_prob'),
            data.get('predicted_size'),
            data.get('dqn_action'),
            data.get('bet_amount', 0.0)
        )
        self.execute_async(query, params)

    def update_actual_result(self, issue_id: int, actual_result: int, profit: float):
        """When the draw resolves 30 seconds later, this patches the database record."""
        query = """
            UPDATE live_predictions 
            SET actual_result = ?, profit = ? 
            WHERE issue_id = ?
        """
        self.execute_async(query, (actual_result, profit, issue_id))

    def shutdown(self):
        """Gracefully completes all pending writes and kills the thread."""
        logger.info(f"Shutting down Async DB. {self.sql_queue.qsize()} pending writes remaining...")
        self.sql_queue.put(None) # Send poison pill
        self.worker_thread.join()
        logger.info("Async DB Thread terminated safely.")


if __name__ == "__main__":
    # Test execution
    db = AsyncDatabaseWriter()
    
    print("Sending Async Query...")
    start = time.time()
    db.log_prediction({
        'issue_id': 999999,
        'lstm_prob': 0.55,
        'trans_prob': 0.56,
        'xgb_prob': 0.51,
        'vae_error': 0.001,
        'meta_calibrated_prob': 0.54,
        'predicted_size': 1,
        'dqn_action': 2,
        'bet_amount': 250.0
    })
    # Notice this time is almost 0.0000ms because it doesn't wait for the hard drive
    print(f"Main thread released in {time.time() - start:.6f} seconds!") 
    
    time.sleep(1) # Give background thread time to process
    db.shutdown()