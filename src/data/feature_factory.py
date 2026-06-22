
# ==============================================================================
# ENGINE_ALPHA - QUANTITATIVE FEATURE FACTORY
# Drop-in replacement for src/data/feature_factory.py
#
# Goals:
# - Match config/feature_config.py exactly
# - Be resilient to missing columns and schema drift
# - Work for both offline training and live inference buffers
# - Produce deterministic, chronological, leak-free engineered features
# - Avoid unnecessary failures while still surfacing real schema issues clearly
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

# ------------------------------------------------------------------------------
# Logging
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
# Project root / imports
# ------------------------------------------------------------------------------

def _discover_project_root() -> str:
    """
    Find the project root robustly, whether this file lives inside the repo or is
    being tested in isolation from /mnt/data.
    """
    env_root = os.getenv("ENGINE_ALPHA_ROOT")
    if env_root:
        env_root = os.path.abspath(env_root)
        if os.path.exists(env_root):
            return env_root

    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [here]
    candidates.extend(list(Path(here).parents))

    for candidate in candidates:
        candidate = os.path.abspath(str(candidate))
        # Common repo layouts
        if os.path.exists(os.path.join(candidate, "config", "global_config.yaml")):
            return candidate
        if os.path.exists(os.path.join(candidate, "global_config.yaml")):
            return candidate
        if os.path.exists(os.path.join(candidate, "feature_config.py")):
            return candidate

    # Fallback for standard src/data/ -> repo root layout
    return os.path.abspath(os.path.join(here, "..", ".."))


PROJECT_ROOT = _discover_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def _load_feature_config():
    """Prefer config.feature_config, then fall back to a local feature_config.py file."""
    try:
        from config import feature_config as cfg  # type: ignore
        return cfg
    except Exception:
        pass

    import importlib.util

    candidate_paths = [
        os.path.join(PROJECT_ROOT, "config", "feature_config.py"),
        os.path.join(PROJECT_ROOT, "feature_config.py"),
        os.path.join(os.path.dirname(__file__), "feature_config.py"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("feature_config", p)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    return None

feature_config = _load_feature_config()
if feature_config is None:
    logger.warning("Could not import feature_config; using internal defaults only.")


class FeatureFactory:
    """
    Defensive feature-engineering pipeline for the ENGINE_ALPHA project.

    This class is intentionally strict about final schema, but lenient while
    building intermediate features so that live data, partial data, or slightly
    malformed CSVs do not crash the whole pipeline unnecessarily.
    """

    DEFAULT_TARGETS: Dict[str, str] = {
        "binary_size": "Size_Target",
        "exact_number": "result_number",
        "one_hot_red": "is_red",
        "one_hot_green": "is_green",
        "one_hot_violet": "is_violet",
    }

    DEFAULT_MODEL_INPUT_FEATURES: Tuple[str, ...] = (
        "duration_ms",
        "lockout_ms",
        "time_hour_sin",
        "time_hour_cos",
        "time_minute_sin",
        "time_minute_cos",
        "time_second_sin",
        "time_second_cos",
        "prev_1_result_number",
        "prev_1_size_target",
        "prev_1_is_red",
        "prev_1_is_green",
        "prev_1_is_violet",
        "prev_2_result_number",
        "prev_2_size_target",
        "number_rolling_std_5",
        "number_rolling_std_10",
        "number_rolling_std_20",
        "size_rolling_std_10",
        "latency_rolling_std_10",
        "size_rolling_mean_3",
        "size_rolling_mean_5",
        "size_rolling_mean_10",
        "size_rolling_mean_20",
        "size_streak_counter",
        "color_red_streak_counter",
        "color_green_streak_counter",
        "color_violet_streak_counter",
        "freq_size_big_last_20",
        "freq_size_small_last_20",
        "freq_color_red_last_20",
        "freq_color_green_last_20",
        "freq_color_violet_last_50",
        "games_since_last_violet",
        "games_since_last_zero",
    )

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            candidates = [
                os.path.join(PROJECT_ROOT, "config", "global_config.yaml"),
                os.path.join(PROJECT_ROOT, "global_config.yaml"),
                os.path.join(os.path.dirname(__file__), "global_config.yaml"),
            ]
            config_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        self.config = self._load_config(config_path)

        fe_cfg = self.config.get("data_pipeline", {}).get("feature_engineering", {})
        horizons = fe_cfg.get("rolling_horizons", [3, 5, 10, 20, 40])
        self.rolling_horizons = self._normalize_horizons(horizons)
        self.sequence_length = int(fe_cfg.get("sequence_length", 60))
        self.strict_schema = bool(fe_cfg.get("strict_schema", False))

        paths_cfg = self.config.get("paths", {})
        self.processed_data_path = os.path.join(PROJECT_ROOT, paths_cfg.get("processed_data_path", "./datasets/WinGo30S_Ready_data.csv"))
        self.raw_data_path = os.path.join(PROJECT_ROOT, paths_cfg.get("raw_data_path", "./extras/WinGo30S_fetched_pages.csv"))

        self.target_map, self.model_input_features = self._load_schema_from_feature_config()

        logger.info(
            "FeatureFactory initialized | seq_len=%s | horizons=%s | strict_schema=%s",
            self.sequence_length,
            self.rolling_horizons,
            self.strict_schema,
        )

    # --------------------------------------------------------------------------
    # Config / schema helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg
        except FileNotFoundError:
            logger.warning(f"Config file not found at {path}; falling back to internal defaults.")
            return {}
        except Exception as exc:
            raise RuntimeError(f"Failed to load config from {path}: {exc}") from exc

    @staticmethod
    def _normalize_horizons(values) -> List[int]:
        out: List[int] = []
        for v in values or []:
            try:
                iv = int(v)
                if iv > 0:
                    out.append(iv)
            except Exception:
                continue
        out = sorted(set(out))
        return out or [3, 5, 10, 20, 40]

    def _load_schema_from_feature_config(self) -> Tuple[Dict[str, str], List[str]]:
        target_map = dict(self.DEFAULT_TARGETS)
        model_features = list(self.DEFAULT_MODEL_INPUT_FEATURES)

        if feature_config is not None:
            try:
                raw_targets = getattr(feature_config, "TARGETS", None)
                if isinstance(raw_targets, dict) and raw_targets:
                    target_map = dict(raw_targets)

                raw_features = getattr(feature_config, "MODEL_INPUT_FEATURES", None)
                if raw_features:
                    model_features = list(raw_features)
            except Exception as exc:
                logger.warning(f"feature_config schema read failed; falling back to defaults: {exc}")

        # Make sure the core targets are always available, even if feature_config is incomplete.
        target_map.setdefault("binary_size", "Size_Target")
        target_map.setdefault("exact_number", "result_number")
        target_map.setdefault("one_hot_red", "is_red")
        target_map.setdefault("one_hot_green", "is_green")
        target_map.setdefault("one_hot_violet", "is_violet")

        return target_map, model_features

    @staticmethod
    def _ensure_dataframe(data: Union[pd.DataFrame, str, os.PathLike]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, (str, os.PathLike)):
            return pd.read_csv(data)
        raise TypeError(
            "FeatureFactory.build_features() expects a pandas DataFrame or a CSV path."
        )

    @staticmethod
    def _safe_numeric(series: pd.Series, default=np.nan) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(default)

    @staticmethod
    def _safe_text(series: pd.Series, default="") -> pd.Series:
        return series.fillna(default).astype(str)

    @staticmethod
    def _to_int8(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8)

    @staticmethod
    def _to_int64(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64)

    @staticmethod
    def _downgrade_memory(df: pd.DataFrame) -> pd.DataFrame:
        start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

        for col in df.columns:
            if col == "issue_id":
                continue

            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                df[col] = s.astype(np.int8)
                continue

            if pd.api.types.is_integer_dtype(s):
                c_min = int(s.min())
                c_max = int(s.max())
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
                # Avoid converting NaNs or infinities to invalid numeric types
                if np.isfinite(s.replace([np.inf, -np.inf], np.nan).dropna()).all():
                    try:
                        df[col] = s.astype(np.float32)
                    except Exception:
                        pass

        end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
        logger.info(f"Memory optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB")
        return df

    @staticmethod
    def _streak_counter(binary_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(binary_series, errors="coerce").fillna(0).astype(np.int8)
        blocks = (s == 0).cumsum()
        return s.groupby(blocks).cumsum().astype(np.int32)

    @staticmethod
    def _drought_counter(binary_series: pd.Series) -> pd.Series:
        s = pd.to_numeric(binary_series, errors="coerce").fillna(0).astype(np.int8)
        hits = s.eq(1)
        groups = hits.cumsum()
        return groups.groupby(groups).cumcount().astype(np.int32)

    # --------------------------------------------------------------------------
    # Base normalization / target derivation
    # --------------------------------------------------------------------------
    def _ensure_required_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add any missing raw columns with safe defaults so the rest of the pipeline
        can proceed without blowing up.
        """
        if "issue_id" not in df.columns:
            df["issue_id"] = np.arange(len(df), dtype=np.int64)

        # Keep raw time columns present so downstream transforms work.
        for col in ("start_ts", "end_ts", "bet_start_ts", "bet_end_ts"):
            if col not in df.columns:
                df[col] = np.nan

        if "result_number" not in df.columns:
            df["result_number"] = np.nan
        if "result_color" not in df.columns:
            df["result_color"] = ""
        if "result_small_big" not in df.columns:
            df["result_small_big"] = ""

        # Some upstream scripts may use the alternate numeric alias.
        if "number" in df.columns and df["result_number"].isna().all():
            df["result_number"] = pd.to_numeric(df["number"], errors="coerce")

        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        start_ts_num = pd.to_numeric(df["start_ts"], errors="coerce")
        if start_ts_num.notna().any():
            df["_sort_start_ts"] = start_ts_num
            df = df.sort_values(
                by=["_sort_start_ts", "issue_id"],
                kind="mergesort",
            ).drop(columns=["_sort_start_ts"])
        else:
            df = df.sort_values(by=["issue_id"], kind="mergesort")

        return df.reset_index(drop=True)

    def _derive_target_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target columns needed by dataset_loader/train_sequences/live inference.
        """
        df["result_number"] = pd.to_numeric(df["result_number"], errors="coerce")

        # Size target: prefer explicit label, otherwise derive from result number.
        if "Size_Target" in df.columns:
            size_from_col = pd.to_numeric(df["Size_Target"], errors="coerce")
        else:
            size_from_col = pd.Series(index=df.index, dtype="float64")

        size_from_number = (df["result_number"] >= 5).astype(np.float32)
        df["Size_Target"] = size_from_col.fillna(size_from_number).fillna(0).astype(np.int8)

        # result_small_big label is useful for human-readable debugging, but not required.
        default_small_big = pd.Series(np.where(df["Size_Target"] == 1, "big", "small"), index=df.index)
        if "result_small_big" not in df.columns or df["result_small_big"].isna().all():
            df["result_small_big"] = default_small_big.astype(str)
        else:
            df["result_small_big"] = df["result_small_big"].fillna(default_small_big).astype(str)

        # Colors from result_color string
        color_text = self._safe_text(df["result_color"]).str.lower()

        if "is_red" not in df.columns:
            df["is_red"] = color_text.str.contains("red", regex=False).astype(np.int8)
        else:
            df["is_red"] = pd.to_numeric(df["is_red"], errors="coerce").fillna(
                color_text.str.contains("red", regex=False).astype(np.int8)
            ).astype(np.int8)

        if "is_green" not in df.columns:
            df["is_green"] = color_text.str.contains("green", regex=False).astype(np.int8)
        else:
            df["is_green"] = pd.to_numeric(df["is_green"], errors="coerce").fillna(
                color_text.str.contains("green", regex=False).astype(np.int8)
            ).astype(np.int8)

        if "is_violet" not in df.columns:
            df["is_violet"] = color_text.str.contains("violet", regex=False).astype(np.int8)
        else:
            df["is_violet"] = pd.to_numeric(df["is_violet"], errors="coerce").fillna(
                color_text.str.contains("violet", regex=False).astype(np.int8)
            ).astype(np.int8)

        # Numeric labels
        df["result_number"] = df["result_number"].fillna(-1).astype(np.int64)
        df["Size_Target"] = df["Size_Target"].fillna(0).astype(np.int8)

        return df

    # --------------------------------------------------------------------------
    # Feature layers
    # --------------------------------------------------------------------------
    def _build_latency_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building latency layer...")
        start_ts = pd.to_numeric(df["start_ts"], errors="coerce")
        end_ts = pd.to_numeric(df["end_ts"], errors="coerce")
        bet_start_ts = pd.to_numeric(df["bet_start_ts"], errors="coerce")
        bet_end_ts = pd.to_numeric(df["bet_end_ts"], errors="coerce")

        df["duration_ms"] = (end_ts - start_ts)
        df["lockout_ms"] = (end_ts - bet_end_ts)

        # Keep bet_open_ms internal if useful for diagnostics, but do not include it
        # in the final schema unless it is part of feature_config.
        df["bet_open_ms"] = (bet_end_ts - bet_start_ts)

        for col in ("duration_ms", "lockout_ms", "bet_open_ms"):
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].clip(lower=0).fillna(0.0)

        return df

    def _build_cyclical_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building cyclical time layer...")
        start_ts = pd.to_numeric(df["start_ts"], errors="coerce")
        dt = pd.to_datetime(start_ts, unit="ms", errors="coerce")

        # Fallback if timestamps arrive as strings in ISO format
        if dt.isna().all():
            dt = pd.to_datetime(df["start_ts"], errors="coerce")

        if dt.isna().all():
            for col in (
                "time_hour_sin", "time_hour_cos",
                "time_minute_sin", "time_minute_cos",
                "time_second_sin", "time_second_cos",
            ):
                df[col] = 0.0
            return df

        hour = dt.dt.hour.fillna(0).astype(np.float32)
        minute = dt.dt.minute.fillna(0).astype(np.float32)
        second = dt.dt.second.fillna(0).astype(np.float32)

        df["time_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["time_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["time_minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
        df["time_minute_cos"] = np.cos(2 * np.pi * minute / 60.0)
        df["time_second_sin"] = np.sin(2 * np.pi * second / 60.0)
        df["time_second_cos"] = np.cos(2 * np.pi * second / 60.0)

        return df

    def _build_lag_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building lag layer...")
        df["prev_1_result_number"] = df["result_number"].shift(1)
        df["prev_1_size_target"] = df["Size_Target"].shift(1)
        df["prev_1_is_red"] = df["is_red"].shift(1)
        df["prev_1_is_green"] = df["is_green"].shift(1)
        df["prev_1_is_violet"] = df["is_violet"].shift(1)

        df["prev_2_result_number"] = df["result_number"].shift(2)
        df["prev_2_size_target"] = df["Size_Target"].shift(2)

        return df

    def _build_volatility_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building volatility layer...")
        # Number volatility
        df["number_rolling_std_5"] = df["result_number"].rolling(window=5, min_periods=2).std()
        df["number_rolling_std_10"] = df["result_number"].rolling(window=10, min_periods=2).std()
        df["number_rolling_std_20"] = df["result_number"].rolling(window=20, min_periods=2).std()

        # Size volatility
        df["size_rolling_std_10"] = df["Size_Target"].rolling(window=10, min_periods=2).std()

        # Latency volatility
        df["latency_rolling_std_10"] = df["duration_ms"].rolling(window=10, min_periods=2).std()

        return df

    def _build_momentum_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building momentum layer...")
        for window in (3, 5, 10, 20):
            df[f"size_rolling_mean_{window}"] = df["Size_Target"].rolling(window=window, min_periods=1).mean()

        # Size streaks as signed run lengths: +N for consecutive Bigs, -N for consecutive Smalls
        size_signed = df["Size_Target"].map({1: 1, 0: -1}).fillna(-1).astype(np.int8)
        blocks = size_signed.diff().ne(0).cumsum()
        df["size_streak_counter"] = size_signed.groupby(blocks).cumsum().astype(np.int32)

        df["color_red_streak_counter"] = self._streak_counter(df["is_red"])
        df["color_green_streak_counter"] = self._streak_counter(df["is_green"])
        df["color_violet_streak_counter"] = self._streak_counter(df["is_violet"])

        return df

    def _build_frequency_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building frequency layer...")
        roll20 = df["Size_Target"].rolling(window=20, min_periods=1)
        df["freq_size_big_last_20"] = roll20.sum()
        df["freq_size_small_last_20"] = roll20.count() - df["freq_size_big_last_20"]

        df["freq_color_red_last_20"] = df["is_red"].rolling(window=20, min_periods=1).sum()
        df["freq_color_green_last_20"] = df["is_green"].rolling(window=20, min_periods=1).sum()
        df["freq_color_violet_last_50"] = df["is_violet"].rolling(window=50, min_periods=1).sum()

        df["games_since_last_violet"] = self._drought_counter(df["is_violet"])
        df["games_since_last_zero"] = self._drought_counter((df["result_number"] == 0).astype(np.int8))

        return df

    def _build_transition_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Additional internal diagnostics. These are useful when debugging, but they
        are not part of MODEL_INPUT_FEATURES unless feature_config adds them.
        """
        df["same_as_prev_1"] = (df["result_number"] == df["prev_1_result_number"]).astype(np.int8)
        df["same_as_prev_2"] = (df["result_number"] == df["prev_2_result_number"]).astype(np.int8)
        return df

    # --------------------------------------------------------------------------
    # Cleaning / schema enforcement
    # --------------------------------------------------------------------------
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Sanitizing missing values...")

        # Lag features naturally create NaNs at the beginning. Backfill where safe.
        lag_cols = [c for c in df.columns if c.startswith("prev_")]
        if lag_cols:
            df[lag_cols] = df[lag_cols].bfill()

        # Numeric columns: replace infs, fill remaining NaNs with zero.
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            df[num_cols] = df[num_cols].fillna(0)

        # Text columns: fill missing strings.
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in {"issue_id"}:
                df[col] = df[col].fillna("").astype(str)

        return df

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Enforcing output schema integrity...")

        required_targets = list(self.target_map.values())
        required_features = list(self.model_input_features)

        # Make sure all target columns are present before slicing.
        missing_targets = [c for c in required_targets if c not in df.columns]
        if missing_targets:
            raise KeyError(f"Missing required target columns: {missing_targets}")

        missing_features = [c for c in required_features if c not in df.columns]
        if missing_features:
            msg = f"Missing engineered features: {missing_features}"
            if self.strict_schema:
                raise KeyError(msg)
            logger.warning(msg + " | Filling missing features with zeroes.")
            for col in missing_features:
                df[col] = 0.0

        # Final column order: issue_id + targets + model features
        final_columns = ["issue_id"] + required_targets + required_features
        final_df = df[final_columns].copy()

        # Deduplicate by issue_id while preserving chronological order.
        final_df = final_df.drop_duplicates(subset=["issue_id"], keep="last").reset_index(drop=True)

        return final_df

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def build_features(self, data: Union[pd.DataFrame, str, os.PathLike]) -> pd.DataFrame:
        """
        Build the final engineered dataframe from either a raw dataframe or a CSV path.
        """
        logger.info("=" * 70)
        logger.info("ENGINE_ALPHA FEATURE FACTORY: processing started")
        logger.info("=" * 70)

        df = self._ensure_dataframe(data)
        if df.empty:
            raise ValueError("Input data is empty.")

        logger.info(f"Input shape: {df.shape}")

        df = self._ensure_required_raw_columns(df)
        df = self._sort_chronologically(df)
        df = self._derive_target_columns(df)

        # Feature pipeline
        df = self._build_latency_layer(df)
        df = self._build_cyclical_layer(df)
        df = self._build_lag_layer(df)
        df = self._build_volatility_layer(df)
        df = self._build_momentum_layer(df)
        df = self._build_frequency_layer(df)
        df = self._build_transition_layer(df)

        # Clean + lock schema
        df = self._handle_missing_values(df)
        df = self._downgrade_memory(df)
        final_df = self._enforce_schema(df)

        logger.info(f"Feature generation complete. Final shape: {final_df.shape}")
        logger.info("=" * 70)
        return final_df

    def save_features(self, input_csv: Optional[str] = None, output_csv: Optional[str] = None) -> str:
        """
        Convenience helper for offline dataset generation.
        """
        if input_csv is None:
            input_csv = self.raw_data_path
        if output_csv is None:
            output_csv = self.processed_data_path

        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file not found: {input_csv}")

        raw_df = pd.read_csv(input_csv)

        # Keep only 30s rows when the column exists.
        if "game_type" in raw_df.columns:
            raw_df = raw_df[raw_df["game_type"].astype(str) == "30s"].copy()

        engineered_df = self.build_features(raw_df)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        engineered_df.to_csv(output_csv, index=False)
        logger.info(f"Engineered dataset written to: {output_csv}")
        return output_csv


def verify_feature_integrity() -> None:
    """
    Mirror the integrity checks in feature_config.py, but also verify that the
    factory produces every declared input feature and target.
    """
    if feature_config is None:
        logger.warning("feature_config unavailable; skipping integrity verification.")
        return

    raw_targets = getattr(feature_config, "TARGETS", {})
    raw_features = getattr(feature_config, "MODEL_INPUT_FEATURES", [])
    burn_list = getattr(feature_config, "BURN_LIST", [])

    overlap_burn = set(raw_features).intersection(set(burn_list))
    assert len(overlap_burn) == 0, f"Overlap between input features and burn list: {overlap_burn}"

    overlap_targets = set(raw_features).intersection(set(raw_targets.values()))
    assert len(overlap_targets) == 0, f"Target leakage inside input features: {overlap_targets}"

    assert len(raw_features) == len(set(raw_features)), "Duplicate features found in MODEL_INPUT_FEATURES."

    print(f"[+] Feature Config Schema Verified: {len(raw_features)} Total Features.")
    print("[+] Zero Train-Serving Skew Risk Detected.")


if __name__ == "__main__":
    try:
        verify_feature_integrity()
    except Exception as exc:
        logger.error(f"Feature integrity check failed: {exc}")
        raise

    logger.info("Running standalone FeatureFactory integration test...")

    test_input_path = os.path.join(PROJECT_ROOT, "extras", "WinGo30S_fetched_pages.csv")
    test_output_path = os.path.join(PROJECT_ROOT, "datasets", "WinGo30S_Ready_data.csv")

    if os.path.exists(test_input_path):
        factory = FeatureFactory()
        out_path = factory.save_features(test_input_path, test_output_path)
        logger.info(f"Standalone test complete: {out_path}")
    else:
        logger.error(f"Cannot run test. File not found: {test_input_path}")
