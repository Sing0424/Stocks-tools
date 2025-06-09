# stage_01_download_symbols.py
import os
import csv
import requests
from datetime import datetime
from config import Config

def download_symbols():
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
            if len(all_rows) < 2:
                print("Downloaded data is empty or malformed.")
                return False
            header = all_rows[0]
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
