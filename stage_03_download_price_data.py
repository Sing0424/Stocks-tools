# stage_03_download_price_data.py

import pandas as pd
import yfinance as yf
import os
import warnings
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from config import Config
from contextlib import redirect_stderr
from io import StringIO

def download_single_stock(symbol):
    try:
        with redirect_stderr(StringIO()):
            stock = yf.Ticker(symbol)
            data = stock.history(period=Config.PRICE_DATA_PERIOD)
            
        if data.empty:
            return None
        data = data.reset_index()
        data['Symbol'] = symbol
        cols = ['Symbol', 'Date'] + [c for c in data.columns if c not in ['Symbol', 'Date']]
        data = data[cols]
        return data
    except:
        return None

def init_worker():
    import warnings
    warnings.filterwarnings("ignore", module="yfinance")

def download_price_data():
    print(f"[{datetime.now()}] Stage 3: Downloading to consolidated file...")
    if not os.path.exists(Config.FILTERED_SYMBOLS_FILE):
        print("Run stage 2 first.")
        return False
    if os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE) and not Config.FORCE_REFRESH_PRICE_DATA:
        print("Consolidated file exists. Set FORCE_REFRESH_PRICE_DATA=True to re-download.")
        return True
    df = pd.read_csv(Config.FILTERED_SYMBOLS_FILE)
    symbols = df['symbol'].tolist()
    with Pool(processes=Config.MAX_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(download_single_stock, symbols), total=len(symbols)))
    warnings.resetwarnings()
    successful = [d for d in results if d is not None]
    if not successful:
        print("No data downloaded.")
        return False
    concat_df = pd.concat(successful, ignore_index=True)
    os.makedirs(os.path.dirname(Config.CONSOLIDATED_PRICE_DATA_FILE), exist_ok=True)
    concat_df.to_csv(Config.CONSOLIDATED_PRICE_DATA_FILE, index=False)
    print(f"Saved {len(concat_df):,} records.")
    return True

if __name__ == "__main__":
    download_price_data()
