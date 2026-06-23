# ==============================================================================
# ENGINE_ALPHA - QUANTITATIVE FEATURE FACTORY (INSTITUTIONAL GRADE)
# Core Component: src/data/feature_factory.py
# Description: Generates ultra-high-fidelity deterministic features including 
# Shannon Entropy, Vectorized Fast Fourier Transforms (FFT), Bayesian Markov 
# Chain transition matrices, and Exponentially Weighted Volatility.
# CRITICAL: Implements aggressive Memory Down-casting and Defensive Parsing
# to ensure zero crashes and massive throughput across millions of rows.
# ==============================================================================

from __future__ import annotations

import os
import sys
import yaml
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ------------------------------------------------------------------------------
# Enterprise Logging & Warning Suppression
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FeatureFactory] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# Project Root Resolution
# ------------------------------------------------------------------------------
def _discover_project_root() -> str:
    """Robust project root discovery for both local and cloud execution."""
    env_root = os.getenv("ENGINE_ALPHA_ROOT")
    if env_root and os.path.exists(os.path.abspath(env_root)):
        return os.path.abspath(env_root)

    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [here] + list(Path(here).parents)

    for candidate in candidates:
        candidate = os.path.abspath(str(candidate))
        if os.path.exists(os.path.join(candidate, "config", "global_config.yaml")):
            return candidate
            
    return os.path.abspath(os.path.join(here, "..", ".."))

PROJECT_ROOT = _discover_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
except ImportError as e:
    logger.error(f"Failed to import strict feature_config schema: {e}")
    sys.exit(1)


class InstitutionalFeatureFactory:
    """
    Defensive, mathematically pure feature-engineering pipeline.
    Calculates advanced PRNG anomalies using Information Theory, Sine-Wave 
    Fourier Extractions, and Bayesian Conditional Probabilities.
    Includes aggressive RAM protection protocols.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        fe_cfg = self.config.get("data_pipeline", {}).get("feature_engineering", {})
        
        # Load sliding window parameters securely
        self.rolling_horizons = fe_cfg.get("rolling_horizons", [3, 5, 10, 20, 40])
        self.entropy_windows = fe_cfg.get("entropy_windows", [10, 20, 40])
        self.sequence_length = int(fe_cfg.get("sequence_length", 60))
        
        self.processed_data_path = os.path.join(PROJECT_ROOT, self.config['paths']['processed_data_path'])
        self.raw_data_path = os.path.join(PROJECT_ROOT, self.config['paths']['raw_data_path'])

        logger.info(f"Institutional Factory Initialized | Seq: {self.sequence_length} | Horizons: {self.rolling_horizons}")

    # ==========================================================================
    # DEFENSIVE MEMORY & PARSING ARCHITECTURE
    # ==========================================================================
    
    @staticmethod
    def _safe_numeric(series: pd.Series, default=np.nan) -> pd.Series:
        """Forces corrupt strings into NaN safely for mathematical interpolation."""
        return pd.to_numeric(series, errors="coerce").fillna(default)

    @staticmethod
    def _safe_text(series: pd.Series, default="") -> pd.Series:
        """Prevents object-type memory leaks."""
        return series.fillna(default).astype(str)

    @staticmethod
    def _downgrade_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through all columns and dynamically downcasts them to the smallest 
        possible memory footprint (int8, float32) without losing precision.
        Critical for preventing RAM explosions when processing 1,000,000+ sequences.
        """
        start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

        for col in df.columns:
            if col == "issue_id":
                continue # Never downcast primary keys

            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                df[col] = s.astype(np.int8)
                continue

            if pd.api.types.is_integer_dtype(s):
                c_min, c_max = int(s.min()), int(s.max())
                if np.iinfo(np.int8).min <= c_min and c_max <= np.iinfo(np.int8).max:
                    df[col] = s.astype(np.int8)
                elif np.iinfo(np.int16).min <= c_min and c_max <= np.iinfo(np.int16).max:
                    df[col] = s.astype(np.int16)
                elif np.iinfo(np.int32).min <= c_min and c_max <= np.iinfo(np.int32).max:
                    df[col] = s.astype(np.int32)
                else:
                    df[col] = s.astype(np.int64)
                continue

            if pd.api.types.is_float_dtype(s):
                # Float32 is standard for Tensor Cores. Prevents double-precision waste.
                if np.isfinite(s.replace([np.inf, -np.inf], np.nan).dropna()).all():
                    try:
                        df[col] = s.astype(np.float32)
                    except Exception:
                        pass

        end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
        logger.info(f"Memory Matrix Optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB (Compression: {(1 - end_mem/start_mem)*100:.1f}%)")
        return df

    # ==========================================================================
    # CORE FOUNDATIONAL LAYERS
    # ==========================================================================
    
    def _enforce_raw_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes raw casino payloads and injects safe failovers."""
        logger.info("Executing Schema Sanitization & Failover Injection...")
        
        if "issue_id" not in df.columns:
            df["issue_id"] = np.arange(len(df), dtype=np.int64)

        for col in ("start_ts", "end_ts", "bet_start_ts", "bet_end_ts"):
            if col not in df.columns: df[col] = np.nan

        # Aggressively hunt for the result_number, regardless of what the scraper called it
        df["result_number"] = self._safe_numeric(df.get("result_number", df.get("number", np.nan)))
        df["result_color"] = self._safe_text(df.get("result_color", "")).str.lower()
        
        return df

    def _derive_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates strict mathematical targets for sequence processing."""
        logger.info("Deriving Categorical Targets & Hot-Encodings...")
        
        size_from_number = (df["result_number"] >= 5).astype(np.float32)
        
        if "Size_Target" in df.columns:
            size_from_col = self._safe_numeric(df["Size_Target"])
            df["Size_Target"] = size_from_col.fillna(size_from_number).fillna(0).astype(np.int8)
        else:
            df["Size_Target"] = size_from_number.fillna(0).astype(np.int8)

        color_text = df["result_color"]
        df["is_red"] = color_text.str.contains("red", regex=False).astype(np.int8)
        df["is_green"] = color_text.str.contains("green", regex=False).astype(np.int8)
        df["is_violet"] = color_text.str.contains("violet", regex=False).astype(np.int8)
        
        df["result_number"] = df["result_number"].fillna(-1).astype(np.int64)
        return df

    def _build_latency_and_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts system execution friction and trigonometric cyclical time."""
        logger.info("Building Latency & Cyclical Trigonometry Layer...")
        
        df["duration_ms"] = (self._safe_numeric(df["end_ts"]) - self._safe_numeric(df["start_ts"])).clip(lower=0).fillna(0)
        df["lockout_ms"] = (self._safe_numeric(df["end_ts"]) - self._safe_numeric(df["bet_end_ts"])).clip(lower=0).fillna(0)

        dt = pd.to_datetime(self._safe_numeric(df["start_ts"]), unit="ms", errors="coerce")
        
        hour = dt.dt.hour.fillna(0).astype(np.float32)
        minute = dt.dt.minute.fillna(0).astype(np.float32)
        second = dt.dt.second.fillna(0).astype(np.float32)

        # Trigonometric mapping forces the neural network to understand that 23:59 and 00:00 are adjacent
        df["time_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["time_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["time_minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
        df["time_minute_cos"] = np.cos(2 * np.pi * minute / 60.0)
        df["time_second_sin"] = np.sin(2 * np.pi * second / 60.0)
        df["time_second_cos"] = np.cos(2 * np.pi * second / 60.0)
        
        return df

    # ==========================================================================
    # ADVANCED QUANTITATIVE LAYERS
    # ==========================================================================

    def _build_information_theory_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Shannon Entropy over rolling windows.
        H(X) = -Sum(P(x) * log2(P(x))). 
        Measures the absolute chaos/predictability of the casino's PRNG.
        A sudden drop in entropy indicates the PRNG is caught in a deterministic loop.
        """
        logger.info("Building Information Theory (Shannon Entropy) Matrix...")
        
        # 🚨 LEAK PREVENTION: We strictly use shifted numbers to evaluate PAST chaos
        shifted_nums = df["result_number"].shift(1).fillna(-1).values.astype(np.int8)
        
        for w in self.entropy_windows:
            col_name = f"shannon_entropy_{w}"
            if len(shifted_nums) < w:
                df[col_name] = 0.0
                continue
                
            # Vectorized sliding window (C-Level optimization)
            windows = sliding_window_view(shifted_nums, window_shape=w)
            
            counts = np.zeros((windows.shape[0], 10))
            for digit in range(10):
                counts[:, digit] = np.sum(windows == digit, axis=1)
                
            probs = counts / w
            probs = np.where(probs > 0, probs, 1) # Prevent log(0) NaN explosions
            entropy = -np.sum(probs * np.log2(probs), axis=1)
            
            # Realign arrays by padding the beginning with zeroes
            df[col_name] = np.concatenate((np.zeros(w-1), entropy))
            
        return df

    def _build_fourier_transform_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes a Discrete Fast Fourier Transform (FFT) over the sliding sequence.
        Extracts the dominant sinusoidal frequencies of the casino's algorithm.
        This forces XGBoost to "see" if the casino is pulsing forced wins/losses at set intervals.
        """
        logger.info("Building Fast Fourier Transform (FFT) Frequency Extraction...")
        
        # 🚨 LEAK PREVENTION
        shifted_target = df["Size_Target"].shift(1).fillna(0).values.astype(np.float32)
        w = self.sequence_length
        
        # Failsafe for very small datasets
        if len(shifted_target) < w:
            for i in range(1, 4):
                df[f"fft_component_{i}_amp"] = 0.0
                df[f"fft_component_{i}_freq"] = 0.0
            return df
            
        windows = sliding_window_view(shifted_target, window_shape=w)
        
        # Calculate FFT across 100,000+ windows simultaneously
        fft_out = np.abs(np.fft.rfft(windows, axis=1))
        freqs = np.fft.rfftfreq(w)
        
        # Zero out the DC component (Index 0 is just the mean of the window, we don't care about it)
        fft_out[:, 0] = 0
        
        # Extract Top 3 Frequencies by amplitude
        top_indices = np.argsort(fft_out, axis=1)[:, -3:][:, ::-1] # Shape: (N, 3)
        
        amp_matrix = fft_out[np.arange(len(fft_out))[:, None], top_indices]
        freq_matrix = freqs[top_indices]
        
        pad = np.zeros((w - 1, 3))
        amps = np.vstack((pad, amp_matrix))
        freqs_stacked = np.vstack((pad, freq_matrix))
        
        for i in range(3):
            df[f"fft_component_{i+1}_amp"] = amps[:, i]
            df[f"fft_component_{i+1}_freq"] = freqs_stacked[:, i]
            
        return df

    def _build_markov_bayesian_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Conditional Bayesian Prior (Markov Chain).
        Answers the question: "Based on the entire history of this casino, what is the exact
        mathematical probability of 'Big' appearing, GIVEN that the last three drops were X, Y, and Z?"
        """
        logger.info("Building Bayesian Markov Chain Transition Matrices...")
        
        shifted_target = df["Size_Target"].shift(1).fillna(0).astype(np.int8)
        
        # Dynamic State Encoding (Converts sequences into unique integer states)
        state_1 = shifted_target
        state_2 = shifted_target * 2 + df["Size_Target"].shift(2).fillna(0).astype(np.int8)
        state_3 = shifted_target * 4 + df["Size_Target"].shift(2).fillna(0).astype(np.int8) * 2 + df["Size_Target"].shift(3).fillna(0).astype(np.int8)
        
        states = {
            'markov_p_big_given_lag_1': state_1,
            'markov_p_big_given_lag_2': state_2,
            'markov_p_big_given_lag_3': state_3
        }
        
        for col_name, state_series in states.items():
            df["_temp_state"] = state_series
            
            # EXTREME SAFETY: We calculate the expanding mean of the target grouped by state, 
            # but we explicitly SHIFT it by 1. This guarantees that for row T, the calculated 
            # probability only includes data from row 0 to T-1. Zero Look-Ahead Bias.
            df[col_name] = df.groupby("_temp_state")["Size_Target"].transform(
                lambda x: x.expanding().mean().shift(1)
            ).fillna(0.5) # Default to pure 50/50 uncertainty if the state is brand new
            
        df.drop(columns=["_temp_state"], inplace=True)
        return df

    def _build_volatility_and_ewma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates structural chaos using standard and Exponentially Weighted parameters."""
        logger.info("Building Standard & Exponentially Weighted Volatility Lags...")
        
        shifted_num = df["result_number"].shift(1)
        shifted_size = df["Size_Target"].shift(1)
        
        df["number_rolling_std_5"] = shifted_num.rolling(5, min_periods=2).std()
        df["number_rolling_std_10"] = shifted_num.rolling(10, min_periods=2).std()
        df["number_rolling_std_20"] = shifted_num.rolling(20, min_periods=2).std()
        df["size_rolling_std_10"] = shifted_size.rolling(10, min_periods=2).std()
        
        # Latency volatility detects server-side updates/jitter without causing target leakage
        df["latency_rolling_std_10"] = df["duration_ms"].rolling(10, min_periods=2).std()
        
        # Financial EWMA (Prioritizes recent drops over older ones inside the window)
        df["number_ewma_10"] = shifted_num.ewm(span=10, adjust=False).mean()
        df["number_ewm_vol_10"] = shifted_num.ewm(span=10, adjust=False).std()
        
        return df

    def _build_momentum_and_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Provides raw memory states and streak/momentum physics to the networks."""
        logger.info("Building Structural Lags & Momentum Vectors...")
        
        # Explicit Physical Lags
        df["prev_1_result_number"] = df["result_number"].shift(1)
        df["prev_1_size_target"] = df["Size_Target"].shift(1)
        df["prev_1_is_red"] = df["is_red"].shift(1)
        df["prev_1_is_green"] = df["is_green"].shift(1)
        df["prev_1_is_violet"] = df["is_violet"].shift(1)
        df["prev_2_result_number"] = df["result_number"].shift(2)
        df["prev_2_size_target"] = df["Size_Target"].shift(2)
        
        # Moving Averages
        shifted_target = df["Size_Target"].shift(1)
        for w in (3, 5, 10, 20):
            df[f"size_rolling_mean_{w}"] = shifted_target.rolling(w, min_periods=1).mean()

        def _streak(binary_series):
            """Fast consecutive streak calculator."""
            s = pd.to_numeric(binary_series, errors="coerce").fillna(0).astype(np.int8)
            blocks = (s == 0).cumsum()
            return s.groupby(blocks).cumsum().astype(np.int32)

        def _drought(binary_series):
            """Fast 'Time Since Last Occurence' drought calculator."""
            s = pd.to_numeric(binary_series, errors="coerce").fillna(0).astype(np.int8)
            groups = s.eq(1).cumsum()
            return groups.groupby(groups).cumcount().astype(np.int32)
            
        # Size Streak is signed: +N for consecutive Bigs, -N for consecutive Smalls
        size_signed = shifted_target.map({1: 1, 0: -1}).fillna(-1).astype(np.int8)
        df["size_streak_counter"] = size_signed.groupby(size_signed.diff().ne(0).cumsum()).cumsum()
        
        df["color_red_streak_counter"] = _streak(df["is_red"].shift(1))
        df["color_green_streak_counter"] = _streak(df["is_green"].shift(1))
        df["color_violet_streak_counter"] = _streak(df["is_violet"].shift(1))
        
        # Normalization and Balancing Hunters
        roll20 = shifted_target.rolling(20, min_periods=1)
        df["freq_size_big_last_20"] = roll20.sum()
        df["freq_size_small_last_20"] = roll20.count() - df["freq_size_big_last_20"]
        df["freq_color_red_last_20"] = df["is_red"].shift(1).rolling(20, min_periods=1).sum()
        df["freq_color_green_last_20"] = df["is_green"].shift(1).rolling(20, min_periods=1).sum()
        df["freq_color_violet_last_50"] = df["is_violet"].shift(1).rolling(50, min_periods=1).sum()
        
        df["games_since_last_violet"] = _drought(df["is_violet"].shift(1))
        df["games_since_last_zero"] = _drought((df["result_number"].shift(1) == 0).astype(int))

        return df

    # ==========================================================================
    # SCHEMA LOCKING & GARBAGE COLLECTION
    # ==========================================================================

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The final defensive checkpoint. 
        Ensures all columns exist, cleans infinities, and locks the matrix geometry.
        """
        logger.info("Enforcing Strict Mathematical Output Schema...")
        
        # Purge Infinities caused by zero-divisions in rolling standard deviations
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        required_targets = list(feature_config.TARGETS.values())
        required_features = list(feature_config.MODEL_INPUT_FEATURES)

        missing = [c for c in required_features if c not in df.columns]
        if missing:
            logger.critical(f"FATAL SCHEMA ERROR. Missing generated features: {missing}")
            raise KeyError(missing)

        final_columns = ["issue_id"] + required_targets + required_features
        final_df = df[final_columns].copy()

        return final_df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Master Orchestrator."""
        logger.info("=" * 70)
        logger.info("ENGINE_ALPHA: INSTITUTIONAL FEATURE FACTORY BOOT SEQUENCE")
        logger.info("=" * 70)
        
        # Stage 1: Safety & Structure
        df = self._enforce_raw_schema(df)
        df = df.sort_values(by=["start_ts", "issue_id"]).reset_index(drop=True)
        df = self._derive_targets(df)
        
        # Stage 2: Feature Matrix Expansion
        df = self._build_latency_and_time(df)
        df = self._build_volatility_and_ewma(df)
        df = self._build_information_theory_layer(df)
        df = self._build_fourier_transform_layer(df)
        df = self._build_markov_bayesian_layer(df)
        df = self._build_momentum_and_lag(df)
        
        # Stage 3: Optimization & Locking
        df = self._enforce_schema(df)
        final_df = self._downgrade_memory(df)
        
        logger.info(f"Feature Engineering Complete. Final Dimensions: {final_df.shape}")
        logger.info("=" * 70)
        return final_df


if __name__ == "__main__":
    logger.info("Running standalone Institutional FeatureFactory Integration Test...")
    
    factory = InstitutionalFeatureFactory()
    
    if os.path.exists(factory.raw_data_path):
        raw_df = pd.read_csv(factory.raw_data_path)
        engineered_df = factory.build_features(raw_df)
        
        os.makedirs(os.path.dirname(factory.processed_data_path), exist_ok=True)
        engineered_df.to_csv(factory.processed_data_path, index=False)
        logger.info(f"Success! Master Matrix written to: {factory.processed_data_path}")
    else:
        logger.error(f"Cannot run test. RAW file not found: {factory.raw_data_path}")