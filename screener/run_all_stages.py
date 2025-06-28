# run_all_stages.py

import sys
import tracemalloc
from datetime import datetime

tracemalloc.start()
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
    from stage_01_download_symbols import download_symbols
    from stage_02_filter_symbols import filter_symbols
    from stage_03_download_price_data import download_price_data
    from stage_04_technical_analysis import analyze_all
    from stage_05_rs_ranking import rank_stocks
    from stage_06_screenResultUpload import upload_results

    print(f"Pipeline start: {datetime.now()}")
    run_stage("Download Symbols", download_symbols)
    run_stage("Filter Symbols", filter_symbols)
    run_stage("Download Price Data", download_price_data)
    run_stage("Technical Analysis", analyze_all)
    run_stage("RS Ranking", rank_stocks)
    run_stage("Upload Results", upload_results)
    print(f"Pipeline completed: {datetime.now()}")