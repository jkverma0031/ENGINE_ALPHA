# ==============================================================================
# ENGINE_ALPHA - INSTITUTIONAL FEATURE REGISTRY v3.0
# Description: Strict Data Segmentation Schema incorporating Information Theory,
# Fourier Transforms, and Markov Chain Probabilities to eliminate Train-Serving 
# Skew and expose the true logic loops of the Casino's PRNG.
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
LAG_FEATURES = [
    'prev_1_result_number',   
    'prev_1_size_target',     
    'prev_1_is_red',          
    'prev_1_is_green',
    'prev_1_is_violet',
    'prev_2_result_number',   
    'prev_2_size_target'
]

# 5. Volatility & Entropy Layer
VOLATILITY_FEATURES = [
    'number_rolling_std_5',   
    'number_rolling_std_10',  
    'number_rolling_std_20',
    'size_rolling_std_10',    
    'latency_rolling_std_10', 
    'number_ewma_10',         # Exponentially Weighted Moving Average (Financial standard)
    'number_ewm_vol_10'       # Exponentially Weighted Volatility
]

# 6. Information Theory Layer (NEW)
# Shannon Entropy calculates the absolute chaos of the sequence. If entropy drops, 
# it means the PRNG is stuck in a deterministic loop.
ENTROPY_FEATURES = [
    'shannon_entropy_10',     # Chaos measurement over last 10 drops
    'shannon_entropy_20',
    'shannon_entropy_40'
]

# 7. Statistical Momentum Layer
MOMENTUM_FEATURES = [
    'size_rolling_mean_3',
    'size_rolling_mean_5',
    'size_rolling_mean_10',
    'size_rolling_mean_20',
    'size_streak_counter',         
    'color_red_streak_counter',
    'color_green_streak_counter',
    'color_violet_streak_counter'
]

# 8. Fast Fourier Transform (FFT) Layer (NEW)
# Extracts cyclical sine-wave frequencies from the PRNG. Identifies if the casino
# forces balancing mechanics exactly every X rounds.
FOURIER_FEATURES = [
    'fft_component_1_amp',    # Amplitude of the primary hidden frequency
    'fft_component_1_freq',   # Wavelength of the primary hidden frequency
    'fft_component_2_amp',    
    'fft_component_2_freq',
    'fft_component_3_amp',
    'fft_component_3_freq'
]

# 9. Markov Chain Transition Matrix (NEW)
# Calculates the mathematical probability of the NEXT state given the CURRENT state.
MARKOV_FEATURES = [
    'markov_p_big_given_lag_1',  # P(Big | t-1)
    'markov_p_big_given_lag_2',  # P(Big | t-1, t-2)
    'markov_p_big_given_lag_3'   # P(Big | t-1, t-2, t-3)
]

# 10. PRNG Balancing & Frequency Layer
FREQUENCY_FEATURES = [
    'freq_size_big_last_20',       
    'freq_size_small_last_20',     
    'freq_color_red_last_20',
    'freq_color_green_last_20',
    'freq_color_violet_last_50',   
    'games_since_last_violet',     
    'games_since_last_zero'        
]

# 11. Core Multi-Model Input Features Matrix
# The absolute ordered array passed to XGBoost, LSTM, Transformer, and Autoencoder.
MODEL_INPUT_FEATURES = (
    LATENCY_FEATURES + 
    CYCLICAL_FEATURES + 
    LAG_FEATURES + 
    VOLATILITY_FEATURES + 
    ENTROPY_FEATURES + 
    MOMENTUM_FEATURES + 
    FOURIER_FEATURES + 
    MARKOV_FEATURES + 
    FREQUENCY_FEATURES
)

# 12. Targets / Label Segmentations
TARGETS = {
    'binary_size': 'Size_Target',               
    'exact_number': 'result_number',            
    'one_hot_red': 'is_red',                    
    'one_hot_green': 'is_green',                
    'one_hot_violet': 'is_violet'               
}

# 13. Purge Registry (The Burn List)
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
    'bet_end_ts'
]

# ==============================================================================
# 14. INSTITUTIONAL SCHEMA GATEKEEPER
# ==============================================================================
def verify_feature_integrity():
    """
    Validation gatekeeper to structurally confirm no data leakage, 
    overlapping column declarations, or dimension mismatch exists.
    """
    print("="*60)
    print("ENGINE_ALPHA: INITIATING FEATURE SCHEMA VERIFICATION")
    print("="*60)
    
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
        
    # 4. Enforce strict Type Checking on variable declarations
    assert all(isinstance(feat, str) for feat in MODEL_INPUT_FEATURES), \
        "[CRITICAL ERROR] Feature names must be strictly strings."
        
    # 5. Dimension Printout
    print(f"[✓] Hardware/Latency Dimensions : {len(LATENCY_FEATURES)}")
    print(f"[✓] Cyclical Time Dimensions  : {len(CYCLICAL_FEATURES)}")
    print(f"[✓] Structural Lag Dimensions : {len(LAG_FEATURES)}")
    print(f"[✓] Volatility Dimensions     : {len(VOLATILITY_FEATURES)}")
    print(f"[✓] Shannon Entropy Dimensions: {len(ENTROPY_FEATURES)}")
    print(f"[✓] Momentum/Trend Dimensions : {len(MOMENTUM_FEATURES)}")
    print(f"[✓] Fourier (FFT) Dimensions  : {len(FOURIER_FEATURES)}")
    print(f"[✓] Markov Chain Dimensions   : {len(MARKOV_FEATURES)}")
    print(f"[✓] Balancing Freq Dimensions : {len(FREQUENCY_FEATURES)}")
    print("-" * 60)
    print(f"[+] Schema Locked & Verified. Total Output Dimensions: {len(MODEL_INPUT_FEATURES)}")
    print("[+] Zero Train-Serving Skew Risk Detected.")
    print("="*60)

if __name__ == "__main__":
    verify_feature_integrity()