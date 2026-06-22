import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(PROJECT_ROOT, "extras", "WinGo30S_fetched_pages.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "datasets", "WinGo30S_Ready_data.csv")

def run_pipeline():
    print("="*50)
    print("[ENGINE_ALPHA] Initializing Data Ingestion Pipeline...")
    print("="*50)

    # 1. Verify File Exists
    if not os.path.exists(INPUT_FILE):
        print(f"[!] ERROR: Input file not found at {INPUT_FILE}")
        sys.exit(1)

    # 2. Load Data
    print(f"[*] Loading raw data from extras/...")
    df = pd.read_csv(INPUT_FILE)
    print(f"[+] Loaded {len(df)} rows.")

    # 3. Filter & Sort (Ensuring Time-Series Integrity)
    print("[*] Filtering for 30s games and sorting chronologically...")
    if 'game_type' in df.columns:
        df = df[df['game_type'] == '30s'].copy()
    
    # Sort by start_ts to ensure the timeline flows forward
    df = df.sort_values(by='start_ts').reset_index(drop=True)

    # 4. Sequence Integrity Check
    print("[*] Scanning for missing sequences...")
    # Convert issue_id to int to check for missing numbers
    df['issue_diff'] = df['issue_id'].astype(np.int64).diff()
    gaps = df[df['issue_diff'] > 1]
    if not gaps.empty:
        print(f"[!] WARNING: Found {len(gaps)} gaps in the sequence timeline.")
        print("[!] Note: Time-series models (LSTMs) may struggle across these gaps.")
    else:
        print("[+] Sequence integrity perfect. No missing gaps detected.")

    # 5. Feature Engineering: Time Deltas
    print("[*] Engineering Time-Delta features...")
    df['duration_ms'] = df['end_ts'] - df['start_ts']
    df['lockout_ms'] = df['end_ts'] - df['bet_end_ts']

    # 6. Target Encoding: Multi-hot Colors
    print("[*] Parsing color classifications...")
    df['is_red'] = df['result_color'].apply(lambda x: 1 if 'red' in str(x).lower() else 0)
    df['is_green'] = df['result_color'].apply(lambda x: 1 if 'green' in str(x).lower() else 0)
    df['is_violet'] = df['result_color'].apply(lambda x: 1 if 'violet' in str(x).lower() else 0)

    # 7. Feature Engineering: Cyclical Time (Trigonometric Transformations)
    print("[*] Applying trigonometric transformations to timestamps...")
    dt_series = pd.to_datetime(df['start_ts'], unit='ms')
    
    # Extract time components
    hours = dt_series.dt.hour
    minutes = dt_series.dt.minute
    seconds = dt_series.dt.second

    # Transform to sine/cosine (Continuous cyclical time)
    df['time_hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['time_hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df['time_minute_sin'] = np.sin(2 * np.pi * minutes / 60.0)
    df['time_minute_cos'] = np.cos(2 * np.pi * minutes / 60.0)
    df['time_second_sin'] = np.sin(2 * np.pi * seconds / 60.0)
    df['time_second_cos'] = np.cos(2 * np.pi * seconds / 60.0)

    # 8. The Burn List (Dropping useless/redundant data)
    print("[*] Purging noisy and redundant columns...")
    cols_to_drop = [
        '_id', 'start_human', 'end_human', 'date', 'status', 
        'issueNumber', 'number', 'game_type', 'result_color',
        'start_ts', 'end_ts', 'bet_start_ts', 'bet_end_ts', 'issue_diff'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 9. Structure Final Output
    # Keep issue_id as an anchor, move targets to the front, followed by features
    front_cols = ['issue_id', 'result_number', 'Size_Target', 'is_red', 'is_green', 'is_violet']
    back_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + back_cols]

    # 10. Save to Datasets Directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("="*50)
    print(f"[+] SUCCESS: Pipeline Complete.")
    print(f"[+] Final Data Shape: {df.shape} (Rows, Columns)")
    print(f"[+] Output saved to: datasets/WinGo30S_Ready_data.csv")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()