# ==============================================================================
# ENGINE_ALPHA - ENTERPRISE RESILIENT API SCRAPER & DATA MINER
# Core Component: extras/web_scraper.py
# Description: High-availability API polling engine. Features connection pooling,
# exponential backoff, and deep-history pagination mining. Includes the 
# SmartDatasetSynchronizer to seamlessly append live data without duplication.
# ==============================================================================

import os
import sys
import time
import json
import yaml
import logging
import random
import threading
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List, Dict

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ScraperCore] %(message)s",
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
    Implements advanced session management and Deep-Mining capabilities.
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
        self.session = self._build_resilient_session()

    def _build_resilient_session(self) -> requests.Session:
        """Creates a connection-pooled HTTP session with Exponential Backoff."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=4,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
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

    def fetch_latest_history(self, limit: int = 150) -> Optional[pd.DataFrame]:
        """Hits the API to grab the most recent game results for live inference."""
        return self._execute_fetch(limit=limit, offset=0)

    def mine_historical_data(self, pages: int = 100, page_size: int = 100) -> Optional[pd.DataFrame]:
        """
        The Catch-Up Protocol. 
        Mines deep history by paginating through the API backwards in time.
        """
        logger.info("="*60)
        logger.info(f"INITIATING DEEP DATA MINING: {pages} Pages requested.")
        logger.info("="*60)
        
        all_dfs = []
        for page in range(pages):
            offset = page * page_size
            df = self._execute_fetch(limit=page_size, offset=offset)
            
            if df is not None and not df.empty:
                all_dfs.append(df)
            else:
                logger.warning(f"Mining hit an empty page at offset {offset}. Stopping.")
                break
                
            # Anti-ban heuristic delay
            time.sleep(random.uniform(0.2, 0.6))
            
            if (page + 1) % 10 == 0:
                logger.info(f"Mined {page + 1}/{pages} pages...")

        if all_dfs:
            master_df = pd.concat(all_dfs, ignore_index=True)
            # Ensure chronological sorting and drop cross-page duplicates
            master_df = master_df.drop_duplicates(subset=['issue_id']).sort_values(by='issue_id').reset_index(drop=True)
            logger.info(f"Deep Mining Complete. Extracted {len(master_df)} unique historical rows.")
            return master_df
        return None

    def _execute_fetch(self, limit: int, offset: int) -> Optional[pd.DataFrame]:
        """Internal worker to hit the API endpoint."""
        payload = {
            "game_type": self.game_code,
            "limit": limit,
            "offset": offset
        }
        
        try:
            response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') != 200 and data.get('status') != 'success':
                return None
                
            records = data.get('data', [])
            if not records:
                return None
                
            return self._parse_json_to_dataframe(records)
            
        except Exception as e:
            logger.error(f"API Fetch Failure at offset {offset}: {e}")
            return None

    def _parse_json_to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """Translates the raw JSON into the strict RAW_COLUMNS schema."""
        df = pd.DataFrame(records)
        
        column_mapping = {
            'issue': 'issue_id', 'result': 'result_number', 
            'color': 'result_color', 'size': 'result_small_big'
        }
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        if 'Size_Target' not in df.columns and 'result_small_big' in df.columns:
            df['Size_Target'] = df['result_small_big'].apply(lambda x: 1 if str(x).lower() == 'big' else 0)
        
        if 'start_ts' not in df.columns:
            current_time = int(time.time() * 1000)
            df['end_ts'] = current_time
            df['start_ts'] = current_time - 30000
            df['bet_end_ts'] = current_time - 5000
            
        for col in feature_config.RAW_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[feature_config.RAW_COLUMNS].copy()
        
        df['issue_id'] = pd.to_numeric(df['issue_id'], errors='coerce')
        df['result_number'] = pd.to_numeric(df['result_number'], errors='coerce')
        
        df = df.dropna(subset=['issue_id']).sort_values(by='issue_id').reset_index(drop=True)
        return df


class SmartDatasetSynchronizer:
    """
    O(1) Data Appender & Deep RAM Buffer.
    Handles appending scraped data to the massive CSV without rewriting the file.
    Maintains a 2,000-row RAM buffer so the Feature Factory can calculate perfect 
    historical momentum features during live execution.
    """
    def __init__(self, config_path: str = None, buffer_size: int = 2000):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "global_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # The user's primary raw dataset
        self.csv_path = os.path.join(PROJECT_ROOT, self.config['paths']['raw_data_path'])
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        
        self.highest_issue_id = 0
        self.buffer_df = pd.DataFrame(columns=feature_config.RAW_COLUMNS)
        
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Loads the end of the CSV into RAM and finds the highest known issue."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                # We don't want to load 50MB into RAM if we don't have to.
                # But to guarantee safety, we load it once, grab the tail, and free memory.
                logger.info("Mounting Master Dataset into RAM Buffer...")
                full_df = pd.read_csv(self.csv_path)
                
                # Sanitize and ensure numeric
                full_df['issue_id'] = pd.to_numeric(full_df['issue_id'], errors='coerce')
                full_df = full_df.dropna(subset=['issue_id']).sort_values(by='issue_id')
                
                if not full_df.empty:
                    self.highest_issue_id = full_df['issue_id'].max()
                    self.buffer_df = full_df.tail(self.buffer_size).copy().reset_index(drop=True)
                    logger.info(f"Master Dataset synced. Current Highest Issue ID: {self.highest_issue_id}")
            except Exception as e:
                logger.error(f"Failed to mount Master Dataset: {e}")
        else:
            logger.warning("Master Dataset not found. A new one will be created.")

    def sync_new_data(self, incoming_df: pd.DataFrame) -> int:
        """
        Takes a dataframe (either from Catch-up or Live loop).
        Filters out rows we already have.
        Appends only the truly new rows to the CSV instantly.
        """
        if incoming_df is None or incoming_df.empty:
            return 0
            
        with self.lock:
            # Filter strictly for issues newer than what we have on disk
            new_rows = incoming_df[incoming_df['issue_id'] > self.highest_issue_id].copy()
            
            if new_rows.empty:
                return 0
                
            new_rows = new_rows.sort_values(by='issue_id')
            
            # 1. Append directly to Hard Drive (O(1) time complexity)
            file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
            new_rows.to_csv(self.csv_path, mode='a', header=not file_exists, index=False)
            
            # 2. Update RAM Buffer
            self.highest_issue_id = new_rows['issue_id'].max()
            self.buffer_df = pd.concat([self.buffer_df, new_rows], ignore_index=True)
            self.buffer_df = self.buffer_df.tail(self.buffer_size).reset_index(drop=True)
            
            added_count = len(new_rows)
            logger.info(f"💾 [Dataset Synchronizer] Appended {added_count} new rows to raw dataset.")
            return added_count

    def get_inference_buffer(self) -> pd.DataFrame:
        """Returns the deep 2,000-row history for the Live Engine's Feature Factory."""
        with self.lock:
            return self.buffer_df.copy()

if __name__ == "__main__":
    # Integration test for the Miner and Synchronizer
    scraper = WinGoLiveScraper()
    sync = SmartDatasetSynchronizer()
    
    test_df = scraper.mine_historical_data(pages=2, page_size=10)
    sync.sync_new_data(test_df)
    
    print("\n[+] Buffer Ready:")
    print(sync.get_inference_buffer().tail(2)[['issue_id', 'result_number', 'Size_Target']])