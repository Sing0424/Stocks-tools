# run_all_stages.py
import os
import sys
import time
from datetime import datetime

# Ensure screener directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import Config

from stage_01_download_symbols import download_symbols
from stage_02_filter_symbols import filter_symbols
from stage_03_download_price_data import download_price_data
from stage_04_technical_analysis_and_ranking import analyze_and_rank
from stage_05_market_breadth import perform_reporting_analysis
from stage_06_screenResultUpload import upload_results

def init_paths():
    paths = [Config.data_folder, Config.GoogleAPI_folder]
    if Config.DOWNLOAD_FOR_WEBAPP:
        paths.append(Config.WEBAPP_DATA_FOLDER)
    
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")
        else:
            print(f"Folder exists: {path}")

def run_stage(name, func, delay=3):
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")
    t0 = time.time()
    try:
        success = func()
        duration = time.time() - t0
        if success:
            print(f"[SUCCESS] {name} (completed in {duration:.1f}s)")
            if delay > 0:
                print(f"Waiting {delay}s before next stage...")
                time.sleep(delay)
        else:
            print(f"[FAIL] {name} (failed after {duration:.1f}s)")
            sys.exit(1)
    except Exception as e:
        duration = time.time() - t0
        print(f"[ERROR] {name}: {e} (after {duration:.1f}s)")
        sys.exit(1)

if __name__ == "__main__":
    init_paths()
    start_time = time.time()
    print(f"Pipeline start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_stage("Download Symbols", download_symbols)
    run_stage("Filter Symbols", filter_symbols)
    run_stage("Download Price Data", download_price_data)
    run_stage("Technical Analysis and RS Ranking", analyze_and_rank)
    run_stage("Reporting Analysis (Market Breadth & Sector)", perform_reporting_analysis)
    run_stage("Upload Results", upload_results)
    
    total_elapsed = time.time() - start_time
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"\nPipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Total time: {mins}m {secs}s)")