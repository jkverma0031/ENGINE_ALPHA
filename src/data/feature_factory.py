# ==============================================================================
# ENGINE_ALPHA - QUANTITATIVE FEATURE FACTORY
# Core Component: src/data/feature_factory.py
# Description: Ingests raw structured data and executes highly optimized, 
# vectorized transformations to generate temporal, momentum, and entropy tensors.
# ==============================================================================

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FeatureFactory] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add root directory to sys.path to resolve config imports safely
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
except ImportError:
    logger.error("Failed to import feature_config. Ensure the script is run from the project root.")
    sys.exit(1)


class FeatureFactory:
    """
    Enterprise-grade pipeline for mathematical feature engineering.
    Guarantees Zero Train-Serving Skew by applying identical vectorized 
    transformations to both massive historical CSVs and live 60-row data streams.
    """

    def __init__(self, config_path: str = None):
        """
        Initializes the factory and loads hyperparameter horizons from global_config.
        """
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
        
        self.config = self._load_config(config_path)
        
        # Extract operational horizons
        fe_config = self.config.get("data_pipeline", {}).get("feature_engineering", {})
        self.rolling_horizons = fe_config.get("rolling_horizons", [3, 5, 10, 20, 40])
        self.sequence_length = fe_config.get("sequence_length", 60)
        
        logger.info(f"FeatureFactory Initialized. Loaded {len(self.rolling_horizons)} momentum horizons.")

    @staticmethod
    def _load_config(path: str) -> dict:
        """Safely loads the global YAML configuration."""
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Could not load global_config.yaml from {path}: {e}")
            sys.exit(1)

    def _downcast_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes memory allocation by downcasting 64-bit structures to 32-bit or 8-bit.
        Crucial for preventing Kaggle Out-Of-Memory (OOM) crashes on 900k+ rows.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB")
        return df

    def _build_latency_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates hardware and network execution times to find RNG seed stress."""
        logger.info("Building Latency Vectors...")
        df['duration_ms'] = df['end_ts'] - df['start_ts']
        df['lockout_ms'] = df['end_ts'] - df['bet_end_ts']
        return df

    def _build_cyclical_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps temporal linear time into continuous trigonometric matrices."""
        logger.info("Building Cyclical Time Matrices...")
        dt_series = pd.to_datetime(df['start_ts'], unit='ms')
        
        h_norm = 2 * np.pi * dt_series.dt.hour / 24.0
        m_norm = 2 * np.pi * dt_series.dt.minute / 60.0
        s_norm = 2 * np.pi * dt_series.dt.second / 60.0

        df['time_hour_sin'] = np.sin(h_norm)
        df['time_hour_cos'] = np.cos(h_norm)
        df['time_minute_sin'] = np.sin(m_norm)
        df['time_minute_cos'] = np.cos(m_norm)
        df['time_second_sin'] = np.sin(s_norm)
        df['time_second_cos'] = np.cos(s_norm)
        
        return df

    def _build_lag_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the 'Immediate Past' tensors so the LSTM is not blind to recent states.
        Calculates t-1 and t-2 observations.
        """
        logger.info("Building Temporal Lag Tensors...")
        
        # Shift 1 game ago
        df['prev_1_result_number'] = df['result_number'].shift(1)
        df['prev_1_size_target'] = df['Size_Target'].shift(1)
        df['prev_1_is_red'] = df['is_red'].shift(1)
        df['prev_1_is_green'] = df['is_green'].shift(1)
        df['prev_1_is_violet'] = df['is_violet'].shift(1)
        
        # Shift 2 games ago
        df['prev_2_result_number'] = df['result_number'].shift(2)
        df['prev_2_size_target'] = df['Size_Target'].shift(2)
        
        return df

    def _build_volatility_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measures rolling entropy (Standard Deviation). High volatility indicates 
        chaotic RNG seeds; low volatility indicates tight clustering.
        """
        logger.info("Building Entropy & Volatility Matrices...")
        
        df['number_rolling_std_5'] = df['result_number'].rolling(window=5, min_periods=2).std()
        df['number_rolling_std_10'] = df['result_number'].rolling(window=10, min_periods=2).std()
        df['number_rolling_std_20'] = df['result_number'].rolling(window=20, min_periods=2).std()
        
        df['size_rolling_std_10'] = df['Size_Target'].rolling(window=10, min_periods=2).std()
        
        # Measures network jitter and server stress
        df['latency_rolling_std_10'] = df['duration_ms'].rolling(window=10, min_periods=2).std()
        
        return df

    def _build_momentum_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts complex structural momentum and stateful streaks using optimized
        array grouping (No iterative loops used).
        """
        logger.info("Building Momentum & Streak Counters...")
        
        # 1. Standard Rolling Averages
        for window in self.rolling_horizons:
            df[f'size_rolling_mean_{window}'] = df['Size_Target'].rolling(window=window, min_periods=1).mean()

        # 2. Vectorized Streak Logic for Size (+N for Big, -N for Small)
        # Map Big (1) to 1, Small (0) to -1
        size_mapped = df['Size_Target'].map({1: 1, 0: -1})
        # Identify boundaries where the streak changes sign
        size_blocks = size_mapped.diff().ne(0).cumsum()
        # Group by the block and cumulatively sum the 1s and -1s
        df['size_streak_counter'] = size_mapped.groupby(size_blocks).cumsum()

        # 3. Vectorized Streak Logic for Colors (Pure positive consecutive counters)
        def calc_streak(series: pd.Series) -> pd.Series:
            """Calculates consecutive 1s in a boolean/binary series."""
            # Creates a new block ID every time the value is 0
            blocks = (series == 0).cumsum()
            # Groups by block, counts cumulatively, but zeros out the 0 elements
            streak = series.groupby(blocks).cumsum()
            return streak
            
        df['color_red_streak_counter'] = calc_streak(df['is_red'])
        df['color_green_streak_counter'] = calc_streak(df['is_green'])
        df['color_violet_streak_counter'] = calc_streak(df['is_violet'])
        
        return df

    def _build_frequency_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RNG balancing algorithms (Pity-timers) tracking the occurrence rates
        and absolute droughts of specific elements over time.
        """
        logger.info("Building Frequency & Drought Layers...")
        
        # 1. Frequency in last N windows
        df['freq_size_big_last_20'] = df['Size_Target'].rolling(window=20, min_periods=1).sum()
        df['freq_size_small_last_20'] = 20 - df['freq_size_big_last_20']  # Inverse logic
        
        df['freq_color_red_last_20'] = df['is_red'].rolling(window=20, min_periods=1).sum()
        df['freq_color_green_last_20'] = df['is_green'].rolling(window=20, min_periods=1).sum()
        df['freq_color_violet_last_50'] = df['is_violet'].rolling(window=50, min_periods=1).sum()

        # 2. Vectorized Drought Counters (Games since last X)
        def calc_drought(series: pd.Series) -> pd.Series:
            """Calculates number of rows since the last '1' occurred."""
            # Cumsum increases by 1 every time the target hits. 
            # This groups all rows following a hit into the same block.
            blocks = series.cumsum()
            # Cumcount calculates the distance from the start of the block
            drought = blocks.groupby(blocks).cumcount()
            # If the block ID is 0, it means the target hasn't hit yet. We cap it.
            # But cumcount works perfectly for this vectorization.
            return drought

        df['games_since_last_violet'] = calc_drought(df['is_violet'])
        
        # For '0', we create a temporary binary mask
        is_zero_digit = (df['result_number'] == 0).astype(int)
        df['games_since_last_zero'] = calc_drought(is_zero_digit)
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lag and Rolling functions create NaNs in the first N rows of a sequence.
        Since LSTM and XGBoost models fail on NaNs, we must sanitize them.
        """
        logger.info("Sanitizing Matrix NaN values...")
        
        # Standard deviations yield NaN for the first element; fill with 0 (zero volatility initially)
        std_cols = [c for c in df.columns if 'std' in c]
        df[std_cols] = df[std_cols].fillna(0.0)
        
        # For lagged targets in the first 2 rows, we backfill them using the earliest known data
        lag_cols = [c for c in df.columns if 'prev_' in c]
        df[lag_cols] = df[lag_cols].bfill()
        
        # Drop any remaining unresolvable rows (Should be absolutely minimal)
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)
        
        if initial_len != final_len:
            logger.warning(f"Dropped {initial_len - final_len} unresolved NaN rows from sequence boundary.")
            
        return df

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strictly enforces that the final output DataFrame matches the exact column 
        definitions mapped in `feature_config.py`. Prevents Data Leakage.
        """
        logger.info("Enforcing Output Schema Integrity...")
        
        # 1. Extract Target Variables
        required_targets = list(feature_config.TARGETS.values())
        
        # 2. Extract Input Features Matrix
        required_features = feature_config.MODEL_INPUT_FEATURES
        
        # 3. Compile Master List
        # issue_id is retained purely for sequence validation, not for training
        master_columns = ['issue_id'] + required_targets + required_features
        
        # 4. Check for missing columns that failed to generate
        missing_cols = set(master_columns) - set(df.columns)
        if missing_cols:
            logger.critical(f"Schema violation! Failed to generate features: {missing_cols}")
            raise KeyError(f"Missing required engineered features: {missing_cols}")

        # 5. Drop the Burn List & Slice the DataFrame to exact specifications
        final_df = df[master_columns].copy()
        
        logger.info(f"Schema locked. Final Output Shape: {final_df.shape}")
        return final_df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MASTER ORCHESTRATION METHOD.
        Executes the full quantitative pipeline on raw input data.
        """
        logger.info("="*50)
        logger.info("ENGINE_ALPHA FEATURE FACTORY: Processing started...")
        logger.info(f"Input Data Shape: {df.shape}")
        logger.info("="*50)
        
        # 0. Safety Copy
        work_df = df.copy()
        
        # 1. Base Target One-Hot Parsing (Required for Lags and Grouping)
        # Note: If passing live inference data, these must be pre-populated by preprocessor
        if 'is_red' not in work_df.columns:
            work_df['is_red'] = work_df['result_color'].apply(lambda x: 1 if 'red' in str(x).lower() else 0)
            work_df['is_green'] = work_df['result_color'].apply(lambda x: 1 if 'green' in str(x).lower() else 0)
            work_df['is_violet'] = work_df['result_color'].apply(lambda x: 1 if 'violet' in str(x).lower() else 0)

        # 2. Execution Pipeline
        work_df = self._build_latency_layer(work_df)
        work_df = self._build_cyclical_layer(work_df)
        work_df = self._build_lag_layer(work_df)
        work_df = self._build_volatility_layer(work_df)
        work_df = self._build_momentum_layer(work_df)
        work_df = self._build_frequency_layer(work_df)
        
        # 3. Post-Processing
        work_df = self._handle_missing_values(work_df)
        work_df = self._downcast_memory(work_df)
        work_df = self._enforce_schema(work_df)
        
        logger.info("="*50)
        logger.info("FEATURE FACTORY: Processing Complete.")
        logger.info("="*50)
        
        return work_df

# ==============================================================================
# LOCAL EXECUTION TEST SCRIPT
# Run this file directly to verify functionality on your local dataset.
# ==============================================================================
if __name__ == "__main__":
    
    # Simulating the pipeline test
    logger.info("Running standalone Factory Integration Test...")
    
    test_input_path = os.path.join(PROJECT_ROOT, "extras", "WinGo30S_fetched_pages.csv")
    test_output_path = os.path.join(PROJECT_ROOT, "datasets", "WinGo30S_Ready_data.csv")
    
    if os.path.exists(test_input_path):
        # 1. Load Raw Data
        raw_df = pd.read_csv(test_input_path)
        
        # Ensure it is chronologically sorted before feature extraction!
        raw_df = raw_df[raw_df['game_type'] == '30s'].copy()
        raw_df = raw_df.sort_values(by='start_ts').reset_index(drop=True)
        
        # 2. Initialize Factory
        factory = FeatureFactory()
        
        # 3. Process the matrix
        final_dataset = factory.build_features(raw_df)
        
        # 4. Save to Disk
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        final_dataset.to_csv(test_output_path, index=False)
        logger.info(f"Test Successful! Processed data written to: {test_output_path}")
        
    else:
        logger.error(f"Cannot run test. File not found: {test_input_path}")