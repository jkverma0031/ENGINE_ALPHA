# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE RESILIENT API SCRAPER
# Core Component: extras/web_scraper.py
# Description: High-availability API polling engine. Features connection pooling,
# exponential backoff, automated retries, and strict Schema mapping to ensure
# live API data perfectly matches the required RAW_COLUMNS matrix.
# ==============================================================================

import os
import sys
import time
import json
import yaml
import logging
import random
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List, Dict

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WebScraper] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from config import feature_config
except ImportError:
    logger.error("Failed to import feature_config. Ensure script runs from project root.")
    sys.exit(1)


class WinGoLiveScraper:
    """
    Robust API Interface for WinGo 30s.
    Implements advanced session management to prevent IP blocking and drops.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        platform = self.config.get('platform_settings', {})
        self.base_url = platform.get('base_url', '')
        self.endpoint = platform.get('api_endpoint', '')
        self.game_code = platform.get('game_code', '30s')
        self.token = platform.get('token', '')
        self.timeout = self.config['inference_and_risk']['scraping_timeout_seconds']
        
        self.api_url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        
        # Initialize Resilient Session
        self.session = self._build_resilient_session()

    def _build_resilient_session(self) -> requests.Session:
        """
        Creates a connection-pooled HTTP session with Exponential Backoff.
        If the server fails (e.g., 502 Bad Gateway), it automatically waits 
        and retries without crashing the python script.
        """
        session = requests.Session()
        
        # Configure Retry Strategy (3 total retries, backoff factor of 0.3)
        # Sleep times: 0.3s, 0.6s, 1.2s
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Dynamic User-Agents to prevent automated blocking
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        ]
        
        session.headers.update({
            "User-Agent": random.choice(user_agents),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}" if self.token else ""
        })
        
        return session

    def fetch_latest_history(self, limit: int = 120) -> Optional[pd.DataFrame]:
        """
        Hits the API to grab the most recent game results.
        Translates the raw JSON into the strict RAW_COLUMNS schema needed by ENGINE_ALPHA.
        """
        payload = {
            "game_type": self.game_code,
            "limit": limit,
            "offset": 0
        }
        
        try:
            start_time = time.time()
            # Post request assuming a standard JSON API structure
            response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            latency = time.time() - start_time
            
            if data.get('code') != 200 and data.get('status') != 'success':
                logger.warning(f"API Returned non-success code: {data}")
                # We do not crash, we return None to let the Live Engine decide what to do
                return None
                
            records = data.get('data', [])
            if not records:
                logger.warning("API returned empty data array.")
                return None
                
            # Parse into DataFrame and enforce Schema Integrity
            df = self._parse_json_to_dataframe(records)
            logger.debug(f"API Fetch Success | Latency: {latency:.3f}s | Rows: {len(df)}")
            
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"API Request TIMEOUT after {self.timeout}s.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API Network Failure: {e}")
            return None
        except Exception as e:
            logger.critical(f"Critical Parsing Error during Scrape: {e}")
            return None

    def _parse_json_to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """
        Crucial translation layer.
        Ensures the JSON payload exactly matches the Phase 2 Feature Config.
        """
        df = pd.DataFrame(records)
        
        # 1. Fallback mapping if API keys differ slightly from expected CSV keys
        # Example: API returns 'issue' instead of 'issue_id'
        column_mapping = {
            'issue': 'issue_id',
            'result': 'result_number',
            'color': 'result_color',
            'size': 'result_small_big'
        }
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # 2. Re-create Size_Target if missing
        if 'Size_Target' not in df.columns and 'result_small_big' in df.columns:
            df['Size_Target'] = df['result_small_big'].apply(lambda x: 1 if str(x).lower() == 'big' else 0)
        
        # 3. Ensure timestamps exist (crucial for latency features)
        # If the API doesn't provide them, we synthesize them safely using the issue_id sequence
        if 'start_ts' not in df.columns:
            logger.debug("API missing 'start_ts'. Synthesizing from local clock and sequence logic.")
            # This is a fallback; ideally, the API returns precise epoch timestamps
            current_time = int(time.time() * 1000)
            df['end_ts'] = current_time
            df['start_ts'] = current_time - 30000
            df['bet_end_ts'] = current_time - 5000
            
        # 4. Strict Column Filtering
        # Add any missing RAW_COLUMNS as NaN to prevent KeyError in pipeline, 
        # then slice to exact list.
        for col in feature_config.RAW_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[feature_config.RAW_COLUMNS].copy()
        
        # 5. Type Enforcement
        df['issue_id'] = pd.to_numeric(df['issue_id'], errors='coerce')
        df['result_number'] = pd.to_numeric(df['result_number'], errors='coerce')
        
        # Sort Chronologically and clean up
        df = df.sort_values(by='issue_id').reset_index(drop=True)
        
        return df

if __name__ == "__main__":
    # Test execution
    scraper = WinGoLiveScraper()
    test_df = scraper.fetch_latest_history(limit=5)
    if test_df is not None:
        print("\n--- SCRAPER TEST SUCCESS ---")
        print(test_df[['issue_id', 'result_number', 'result_color', 'Size_Target']])