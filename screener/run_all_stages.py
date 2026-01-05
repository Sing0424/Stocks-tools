# run_all_stages.py
import os
import sys
import tracemalloc
from datetime import datetime
from config import Config

tracemalloc.start()

def init_paths():
    if Config.DOWNLOAD_FOR_WEBAPP:
        paths = [Config.data_folder, Config.GoogleAPI_folder, Config.WEBAPP_DATA_FOLDER]
    else:
        paths = [Config.data_folder, Config.GoogleAPI_folder]
    
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        else:
            print(f"Folder exists: {path}")
    return

def run_stage(name, func):
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")
    try:
        if func():
            print(f"[SUCCESS] {name}")
        else:
            print(f"[FAIL] {name}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_paths()
    from stage_01_download_symbols import download_symbols
    from stage_02_filter_symbols import filter_symbols
    from stage_03_download_price_data import download_price_data
    from stage_04_technical_analysis_and_ranking import analyze_and_rank
    from stage_05_screenResultUpload import upload_results

    print(f"Pipeline start: {datetime.now()}")
    run_stage("Download Symbols", download_symbols)
    run_stage("Filter Symbols", filter_symbols)
    run_stage("Download Price Data", download_price_data)
    run_stage("Technical Analysis and RS Ranking", analyze_and_rank)
    run_stage("Upload Results", upload_results)
    print(f"Pipeline completed: {datetime.now()}")