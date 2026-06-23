import os
import csv
import time
import requests
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

# ==========================================
# 0. CONFIGURATION & TARGET FILE
# ==========================================
# 🚨 UPDATE THIS WITH YOUR CURRENT 32-CHAR HEX TOKEN
TOKEN = "f1e658069f4d4bdabf2338eadb8bb9a6"

GAME_CODE = "30s"
BASE_URL = "https://damangameworld.org"
DATA_FILE = "WinGo30S_fetched_pages.csv"

# Exact structural mapping to the Casino's JSON
ALL_COLUMNS = [
    "_id", "start_ts", "start_human", "end_human", "end_ts", 
    "bet_start_ts", "bet_end_ts", "date", "issue_id", "status", 
    "result_color", "result_number", "result_small_big", "game_type", 
    "issueNumber", "number", "Size_Target" 
]

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,hi;q=0.6",
    "authorization": f"Bearer {TOKEN}",
    "cache-control": "no-cache",
    "cookie": "_ga=GA1.1.375558395.1782054417; is_splash_screen_shown=true; show_daman_recharge_bonus=true; show_fake_site_notice=true; show_daman_welcome=true; _clck=1o7np1k%5E2%5Eg74%5E0%5E2363; _clsk=1ycadox%5E1782086696377%5E3%5E1%5Ea.clarity.ms%2Fcollect; _ga_ZB406B1P12=GS2.1.s1782085508$o4$g1$t1782086705$j49$l0$h0",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://damangameworld.org/",
    "sec-ch-ua": "\"Google Chrome\";v=\"149\", \"Chromium\";v=\"149\", \"Not)A;Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
}

# ==========================================
# 1. BULLETPROOF MULTI-THREADED SCRAPER
# ==========================================
class BulletproofScraper:
    def __init__(self, max_workers=50): 
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        
        adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=3)
        self.session.mount("https://", adapter)
        
        self.fatal_error_event = threading.Event()
        self.end_of_database = threading.Event() 
        
        self.memory_buffer = []
        self.buffer_lock = threading.Lock()
        self.total_saved_rows = 0

    def safe_get(self, endpoint, params, max_retries=3):
        for attempt in range(max_retries):
            if self.fatal_error_event.is_set() or self.end_of_database.is_set():
                return None
                
            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                
                if response.status_code in [403, 429]:
                    time.sleep(3 + (attempt * 2)) 
                    continue
                    
                if response.status_code == 401:
                    print("\n❌ [FATAL] Token Expired (HTTP 401). Shutting down...")
                    self.fatal_error_event.set()
                    return None
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException:
                time.sleep(2 + attempt) 
                if attempt == max_retries - 1:
                    return None

    def fetch_page(self, page_no):
        if self.fatal_error_event.is_set() or self.end_of_database.is_set():
            return []
            
        endpoint = f"{BASE_URL}/api/wingo/get-issues"
        params = {"game_type": GAME_CODE, "type": "historical", "page": page_no, "per_page": 10}
        
        data = self.safe_get(endpoint, params)
        
        if not data:
            return []
            
        records = data.get("data", [])
        
        if not records:
            print(f"\n🟢 [END OF HISTORY DETECTED AT PAGE {page_no}]. Stopping all remaining threads...")
            self.end_of_database.set()
            return []
            
        rows = []
        for row in records:
            # Drop the live pending row so it doesn't break the CSV math
            if row.get("status") != "completed" or row.get("result_number") == "na":
                continue
                
            # Assuming perfect casino format as requested
            r_num = int(row["result_number"])
            
            rows.append({
                "_id": row["_id"],
                "start_ts": row["start_ts"],
                "start_human": row["start_human"],
                "end_human": row["end_human"],
                "end_ts": row["end_ts"],
                "bet_start_ts": row["bet_start_ts"],
                "bet_end_ts": row["bet_end_ts"],
                "date": row["date"],
                "issue_id": row["issue_id"],
                "status": row["status"],
                "result_color": row["result_color"],
                "result_number": row["result_number"],
                "result_small_big": row["result_small_big"],
                "game_type": row["game_type"],
                
                # ML specific duplicates
                "issueNumber": row["issue_id"],
                "number": r_num,
                "Size_Target": 1 if r_num >= 5 else 0
            })
                
        return rows

    def flush_buffer_to_disk(self):
        """Uses native csv.DictWriter to prevent string injection/comma corruption."""
        if not self.memory_buffer:
            return
            
        file_exists = os.path.exists(DATA_FILE)
        
        with open(DATA_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerows(self.memory_buffer)
        
        self.total_saved_rows += len(self.memory_buffer)
        self.memory_buffer.clear()

    def execute_massive_scrape(self, start_page=1, target_pages=90000):
        print(f"\n🚀 Initiating MASSIVE SCRAPE from page {start_page:,} to {target_pages:,} using {self.max_workers} Workers.")
        print(f"💾 Utilizing RAM Batching. Native OS disk writes occur every 5,000 rows.\n")
        
        # Initialize header safely to prevent missing keys on first run
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                writer.writeheader()

        pages_processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_page, p): p for p in range(start_page, target_pages + 1)}
            
            for future in as_completed(futures):
                if self.fatal_error_event.is_set() or self.end_of_database.is_set():
                    continue
                    
                try:
                    result = future.result()
                    if result: 
                        with self.buffer_lock:
                            self.memory_buffer.extend(result)
                            
                            if len(self.memory_buffer) >= 5000:
                                self.flush_buffer_to_disk()
                                pages_processed += 500  # 5000 rows / 10 per page
                                print(f"🟢 [DISK WRITE] Total Rows Documented in Session: {self.total_saved_rows:,} | (Approx Pages Scanned: {pages_processed:,})")
                                
                except Exception:
                    pass

        # Final flush for remaining stragglers
        with self.buffer_lock:
            self.flush_buffer_to_disk()

        print(f"\n✅ Network operations complete. Proceeding to final data cleanup...")
        return self.total_saved_rows

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Resume safety switch
    START_PAGE = 0  
    TARGET_PAGES = 0  
    WORKERS = 50          
    
    scraper = BulletproofScraper(max_workers=WORKERS)
    
    try:
        total_fetched = scraper.execute_massive_scrape(start_page=START_PAGE, target_pages=TARGET_PAGES)
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard Interrupt detected! Saving progress...")
        scraper.fatal_error_event.set()
        with scraper.buffer_lock:
            scraper.flush_buffer_to_disk()

    # ---------------------------------------------------------
    # MEMORY-SAFE FINAL CLEANUP (Zero Data Corruption Guarantee)
    # ---------------------------------------------------------
    if os.path.exists(DATA_FILE):
        print(f"\n🧹 Performing final deduplication and chronological sort...")
        try:
            # 🚨 CRITICAL FIX: Force Pandas to treat ALL columns as strings.
            # This prevents massive Unix timestamps from being corrupted into scientific notation (1.78e+12)
            safe_dtypes = {col: str for col in ALL_COLUMNS}
            
            # Explicitly specifying utf-8 encoding to prevent Windows cp1252 charmap crashes
            final_df = pd.read_csv(DATA_FILE, dtype=safe_dtypes, encoding='utf-8')
            final_df = final_df.drop_duplicates(subset=["issue_id"])
            final_df = final_df.sort_values(by="issue_id").reset_index(drop=True)
            
            final_df.to_csv(DATA_FILE, index=False, encoding='utf-8')
            
            print(f"🏆 MASTER FILE READY: {len(final_df):,} total chronological rows saved securely.\n")
        except Exception as e:
            print(f"❌ Error during final cleanup: {e}")