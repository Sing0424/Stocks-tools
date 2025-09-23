# stage_03_download_price_data.py

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from config import Config
from contextlib import redirect_stderr
from io import StringIO
from curl_cffi import requests as cffi_requests
import shutil

def download_price_data():
    print(f"CPU Threads: {Config.DOWNLOAD_WORKERS}")
    print(f"Batch_Size: {Config.BATCH_SIZE}")
    print(f"[{datetime.now()}] Stage 3: Downloading to consolidated file...")
    if not os.path.exists(Config.FILTERED_SYMBOLS_FILE):
        print("Run stage 2 first.")
        return False
    if os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE) and not Config.FORCE_REFRESH_PRICE_DATA:
        print("Consolidated file exists. Set FORCE_REFRESH_PRICE_DATA=True to re-download.")
        return True
    df = pd.read_csv(Config.FILTERED_SYMBOLS_FILE)
    symbols = df['symbol'].tolist()
    
    # Split symbols into batches
    symbol_batches = [symbols[i:i + Config.BATCH_SIZE] for i in range(0, len(symbols), Config.BATCH_SIZE)]

    # Create the output file and write the header
    pd.DataFrame(columns=['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']).to_csv(Config.CONSOLIDATED_PRICE_DATA_FILE, index=False)

    total_records = 0
    with Pool(processes=Config.DOWNLOAD_WORKERS) as pool:
        for batch_result in tqdm(pool.imap(download_multiple_stocks, symbol_batches), total=len(symbol_batches)):
            successful_downloads = [result for result in batch_result if result is not None]
            if successful_downloads:
                concat_df = pd.concat(successful_downloads, ignore_index=True)
                # Append to the CSV file without writing headers
                concat_df.to_csv(Config.CONSOLIDATED_PRICE_DATA_FILE, mode='a', header=False, index=False)
                total_records += len(concat_df)

    if total_records > 0:
        print(f"Saved {total_records:,} records.")
        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(Config.CONSOLIDATED_PRICE_DATA_FILE_WEBAPP), exist_ok=True)
        shutil.copy(Config.CONSOLIDATED_PRICE_DATA_FILE, Config.CONSOLIDATED_PRICE_DATA_FILE_WEBAPP)
        print(f"Saved {total_records:,} records for webapp.")
    else:
        print("No data was successfully downloaded.")
        # If no data, remove the created file (it will only have headers)
        os.remove(Config.CONSOLIDATED_PRICE_DATA_FILE)
        return False
    return True

def download_multiple_stocks(symbols):
    # Create a new session for each worker process to avoid pickling issues.
    # impersonate="chrome110" makes requests look like they're from a real browser.
    session = cffi_requests.Session(impersonate="chrome110")
    try:
        with redirect_stderr(StringIO()):
            data = yf.download(
                tickers=symbols,
                period=Config.PRICE_DATA_PERIOD,
                group_by='ticker',
                session=session)
        
        if data.empty:
            return [None] * len(symbols)  # Return None for each symbol if download failed
        
        results = []
        for symbol in symbols:
            if symbol in data.columns:  # Check if the symbol's data is in the downloaded data
                symbol_data = data[symbol].reset_index()  # Reset index to make 'Date' a column
                if not symbol_data.empty:
                    symbol_data['Symbol'] = symbol
                    cols = ['Symbol', 'Date'] + [col for col in symbol_data.columns if col not in ['Symbol', 'Date']]
                    symbol_data = symbol_data[cols]
                    results.append(symbol_data)
                else:
                    results.append(None)  # Append None if the DataFrame is empty for this symbol
            else:
                results.append(None)  # Append None if symbol's data is not present in the downloaded data
        return results
    except Exception as e:
        print(f"Failed to download data for symbols {symbols}: {e}")
        return [None] * len(symbols)

if __name__ == "__main__":
    download_price_data()