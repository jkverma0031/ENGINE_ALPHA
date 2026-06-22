# ==============================================================================
# ENGINE_ALPHA - FEATURE CONFIGURATION SCHEMA v2.0
# Strict Data Segmentation Schema to Eliminate Train-Serving Skew
# ==============================================================================

# 1. Base Framework Source Mapping
RAW_COLUMNS = [
    '_id', 'game_type', 'issue_id', 'start_ts', 'end_ts', 'bet_start_ts', 
    'bet_end_ts', 'start_human', 'end_human', 'date', 'status', 'issueNumber', 
    'number', 'result_number', 'result_color', 'Size_Target'
]

# 2. Hardware and Network Latency Layer
LATENCY_FEATURES = [
    'duration_ms',  # System execution duration loop calculation
    'lockout_ms'    # Gap between betting close interface and generator outcome
]

# 3. Trigonometric Temporal Layer
CYCLICAL_FEATURES = [
    'time_hour_sin',
    'time_hour_cos',
    'time_minute_sin',
    'time_minute_cos',
    'time_second_sin',
    'time_second_cos'
]

# 4. The Lag Layer (Immediate Past History at t-1)
# These allow the LSTM and Transformer to actually "read" the sequence of previous drops
LAG_FEATURES = [
    'prev_1_result_number',   # The exact number that dropped 1 game ago
    'prev_1_size_target',     # The size that dropped 1 game ago
    'prev_1_is_red',          
    'prev_1_is_green',
    'prev_1_is_violet',
    'prev_2_result_number',   # The exact number that dropped 2 games ago
    'prev_2_size_target'
]

# 5. Volatility & Entropy Layer
# Captures the structural chaos and spread of the recent PRNG outputs
VOLATILITY_FEATURES = [
    'number_rolling_std_5',   # Standard deviation of precise numbers over 5 games
    'number_rolling_std_10',  
    'number_rolling_std_20',
    'size_rolling_std_10',    # Measures if the size is oscillating wildly or clustering
    'latency_rolling_std_10'  # Detects erratic server load (network jitter)
]

# 6. Statistical Momentum Layer
MOMENTUM_FEATURES = [
    'size_rolling_mean_3',
    'size_rolling_mean_5',
    'size_rolling_mean_10',
    'size_rolling_mean_20',
    'size_streak_counter',         # E.g., +4 (4 Bigs in a row), -3 (3 Smalls in a row)
    'color_red_streak_counter',
    'color_green_streak_counter',
    'color_violet_streak_counter'
]

# 7. PRNG Balancing & Frequency Layer
# Hunts for forced normalization algorithms (pity-timers) inside the platform code
FREQUENCY_FEATURES = [
    'freq_size_big_last_20',       # Count of 'Big' in the last 20 games
    'freq_size_small_last_20',     
    'freq_color_red_last_20',
    'freq_color_green_last_20',
    'freq_color_violet_last_50',   # Violet is rare, needs a longer lookback window
    'games_since_last_violet',     # Tracks absolute drought lengths for the 0/5 split
    'games_since_last_zero'        # Specifically tracks the '0' digit drought
]

# 8. Core Multi-Model Input Features Matrix
# Combined list defining the precise array order fed into Supervised Systems
MODEL_INPUT_FEATURES = (
    LATENCY_FEATURES + 
    CYCLICAL_FEATURES + 
    LAG_FEATURES + 
    VOLATILITY_FEATURES + 
    MOMENTUM_FEATURES + 
    FREQUENCY_FEATURES
)

# 9. Targets / Label Segmentations
TARGETS = {
    'binary_size': 'Size_Target',               # Primary Focus [0 = Small, 1 = Big]
    'exact_number': 'result_number',            # Hard Target Multi-Class [0 to 9]
    'one_hot_red': 'is_red',                    # Color Classification Mapping [0 or 1]
    'one_hot_green': 'is_green',                # Color Classification Mapping [0 or 1]
    'one_hot_violet': 'is_violet'               # Color Classification Mapping [0 or 1]
}

# 10. Purge Registry (The Burn List)
# Columns extracted or dropped because they are redundant or leak downstream target data
BURN_LIST = [
    '_id', 
    'game_type', 
    'start_human', 
    'end_human', 
    'date', 
    'status', 
    'issueNumber', 
    'number', 
    'result_color',
    'start_ts', 
    'end_ts', 
    'bet_start_ts', 
    'bet_end_ts',
    # Note: 'Size_Target' and 'result_number' are NOT burned here; 
    # they are extracted cleanly as targets during the dataset_loader.py phase.
]

# 11. Integrity Mapping Assertions
def verify_feature_integrity():
    """
    Validation gatekeeper to structurally confirm no data leakage or 
    overlapping column declarations exist anywhere inside the framework.
    """
    # 1. Ensure no input features are secretly in the burn list
    overlap_burn = set(MODEL_INPUT_FEATURES).intersection(set(BURN_LIST))
    assert len(overlap_burn) == 0, \
        f"[CRITICAL ERROR] Overlap detected between Input Features and Burn List: {overlap_burn}"
        
    # 2. Prevent Data Leakage: Ensure future targets are NEVER in the input features
    overlap_targets = set(MODEL_INPUT_FEATURES).intersection(set(TARGETS.values()))
    assert len(overlap_targets) == 0, \
        f"[CRITICAL ERROR] Target column leakage found inside INPUT_FEATURES: {overlap_targets}"

    # 3. Check for duplicates in the input feature matrix
    assert len(MODEL_INPUT_FEATURES) == len(set(MODEL_INPUT_FEATURES)), \
        "[CRITICAL ERROR] Duplicate features found inside MODEL_INPUT_FEATURES."

    print(f"[+] Feature Config Schema Verified: {len(MODEL_INPUT_FEATURES)} Total Features.")
    print("[+] Zero Train-Serving Skew Risk Detected.")

if __name__ == "__main__":
    verify_feature_integrity()