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


def download_price_data():
    print(f"CPU Threads: {Config.WORKERS}")
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

    # To avoid accumulating all data in memory, we'll write to the CSV file in chunks.
    # The first batch will write the header, subsequent batches will append.
    is_first_batch = True
    total_records_saved = 0

    with Pool(processes=Config.WORKERS) as pool:
        for batch_result in tqdm(pool.imap(download_multiple_stocks, symbol_batches), total=len(symbol_batches)):
            # Filter out None results for the current batch
            successful_batch_downloads = [result for result in batch_result if result is not None]

            if successful_batch_downloads:
                # Concatenate DataFrames from the current batch only
                batch_df = pd.concat(successful_batch_downloads, ignore_index=True)
                
                if is_first_batch:
                    # For the first batch, write with header
                    batch_df.to_csv(Config.CONSOLIDATED_PRICE_DATA_FILE, index=False, mode='w')
                    is_first_batch = False
                else:
                    # For subsequent batches, append without header
                    batch_df.to_csv(Config.CONSOLIDATED_PRICE_DATA_FILE, index=False, mode='a', header=False)
                
                total_records_saved += len(batch_df)

    if total_records_saved == 0:
        print("No data was successfully downloaded.")
        return False

    print(f"Saved {total_records_saved:,} records to {Config.CONSOLIDATED_PRICE_DATA_FILE}")
    return True

def download_multiple_stocks(symbols):
    # Create a new session for each worker process to avoid pickling issues.
    # impersonate="chrome110" makes requests look like they're from a real browser.
    session = cffi_requests.Session(impersonate="chrome")
    try:
        with redirect_stderr(StringIO()):
            data = yf.download(
                tickers=symbols,
                period=Config.PRICE_DATA_PERIOD,
                group_by='ticker',
                session=session)
            # print(data)
        
        if data.empty:
            return [None] * len(symbols)  # Return None for each symbol if download failed
        
        results = []
        for symbol in symbols:
            if symbol in data.columns:  # Check if the symbol's data is in the downloaded data
                symbol_data = data[symbol].reset_index()  # Reset index to make 'Date' a column
                if not symbol_data.empty:
                    symbol_data['Symbol'] = symbol
                    cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
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