# stage_01_download_symbols.py
import os
import sys
import csv
import requests
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import Config

def download_symbols():
    print(f"Using CPU threads: {Config.WORKERS}")
    print(f"[{datetime.now()}] Stage 1: Downloading symbols...")
    if os.path.exists(Config.LISTING_STATUS_FILE) and not Config.FORCE_REFRESH_SYMBOLS:
        print("Symbols file exists. Set FORCE_REFRESH_SYMBOLS=True to re-download.")
        return True
    try:
        url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={Config.ALPHA_VANTAGE_API_KEY}'
        with requests.Session() as s:
            response = s.get(url)
            response.raise_for_status()
            decoded = response.content.decode('utf-8')
            cr = csv.reader(decoded.splitlines(), delimiter=',')
            all_rows = list(cr)
            header = all_rows[0]
            if len(all_rows) < 2 or 'symbol' not in [c.strip().lower() for c in header]:
                print(f"Downloaded data is invalid or API returned an error: {header}")
                return False
            data_rows = all_rows[1:]
            data_rows.sort(key=lambda r: r[0])
            with open(Config.LISTING_STATUS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data_rows)
        print(f"Downloaded {len(data_rows)} symbols.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    download_symbols()
