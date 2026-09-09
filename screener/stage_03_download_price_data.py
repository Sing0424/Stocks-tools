# stage_03_download_price_data.py

import os
import sys
from datetime import datetime
from io import StringIO
from contextlib import redirect_stderr
from multiprocessing import Pool
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import Config

PRICE_DATA_SCHEMA = pa.schema([
    ('Symbol', pa.string()),
    ('Date', pa.timestamp('ms')),
    ('Open', pa.float64()),
    ('High', pa.float64()),
    ('Low', pa.float64()),
    ('Close', pa.float64()),
    ('Volume', pa.float64()),
])

def download_price_data():
    print(f"CPU Threads: {Config.WORKERS}")
    print(f"Batch_Size: {Config.BATCH_SIZE}")
    print(f"[{datetime.now()}] Stage 3: Downloading to consolidated file...")

    if not os.path.exists(Config.FILTERED_SYMBOLS_FILE):
        print("Run stage 2 first.")
        return False

    target_file = Config.CONSOLIDATED_PRICE_DATA_PARQUET if Config.USE_PARQUET else Config.CONSOLIDATED_PRICE_DATA_FILE
    if os.path.exists(target_file) and not Config.FORCE_REFRESH_PRICE_DATA:
        print(f"Consolidated file exists at {target_file}. Set FORCE_REFRESH_PRICE_DATA=True to re-download.")
        return True

    df = pd.read_csv(Config.FILTERED_SYMBOLS_FILE)
    symbols = df['symbol'].dropna().unique().tolist()
    
    # Split symbols into batches
    symbol_batches = [symbols[i:i + Config.BATCH_SIZE] for i in range(0, len(symbols), Config.BATCH_SIZE)]

    total_records_saved = 0
    parquet_writer = None
    is_first_csv_batch = True

    try:
        with Pool(processes=Config.WORKERS) as pool:
            for batch_result in tqdm(pool.imap(download_multiple_stocks, symbol_batches), total=len(symbol_batches)):
                successful_batch_downloads = [result for result in batch_result if result is not None and not result.empty]

                if successful_batch_downloads:
                    batch_df = pd.concat(successful_batch_downloads, ignore_index=True)
                    batch_df['Date'] = pd.to_datetime(batch_df['Date']).dt.tz_localize(None)
                    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_cols:
                        batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce').astype('float64')
                    batch_df['Symbol'] = batch_df['Symbol'].astype(str)
                    batch_df = batch_df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

                    # Write to Parquet
                    if Config.USE_PARQUET:
                        table = pa.Table.from_pandas(batch_df, schema=PRICE_DATA_SCHEMA, preserve_index=False)
                        if parquet_writer is None:
                            parquet_writer = pq.ParquetWriter(
                                Config.CONSOLIDATED_PRICE_DATA_PARQUET,
                                PRICE_DATA_SCHEMA,
                                compression='snappy'
                            )
                        parquet_writer.write_table(table)

                    # Write to CSV if Parquet is not used or if webapp sync is requested
                    if not Config.USE_PARQUET or Config.DOWNLOAD_FOR_WEBAPP:
                        csv_targets = []
                        if not Config.USE_PARQUET:
                            csv_targets.append(Config.CONSOLIDATED_PRICE_DATA_FILE)
                        if Config.DOWNLOAD_FOR_WEBAPP:
                            csv_targets.append(Config.CONSOLIDATED_PRICE_DATA_FILE_WEBAPP)

                        for target_csv in csv_targets:
                            os.makedirs(os.path.dirname(target_csv), exist_ok=True)
                            if is_first_csv_batch:
                                batch_df.to_csv(target_csv, index=False, mode='w')
                            else:
                                batch_df.to_csv(target_csv, index=False, mode='a', header=False)
                        is_first_csv_batch = False
                    
                    total_records_saved += len(batch_df)
    finally:
        if parquet_writer:
            parquet_writer.close()

    if total_records_saved == 0:
        print("No data was successfully downloaded.")
        if Config.USE_PARQUET and os.path.exists(Config.CONSOLIDATED_PRICE_DATA_PARQUET):
            os.remove(Config.CONSOLIDATED_PRICE_DATA_PARQUET)
        if os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE):
            os.remove(Config.CONSOLIDATED_PRICE_DATA_FILE)
        return False

    output_path = Config.CONSOLIDATED_PRICE_DATA_PARQUET if Config.USE_PARQUET else Config.CONSOLIDATED_PRICE_DATA_FILE
    print(f"Saved {total_records_saved:,} records to {output_path}")
    return True

def download_multiple_stocks(symbols):
    try:
        with redirect_stderr(StringIO()):
            data = yf.download(
                tickers=symbols,
                period=Config.PRICE_DATA_PERIOD,
                group_by='ticker',
                progress=False
            )
        
        if data.empty:
            return [None] * len(symbols)
        
        results = []
        is_multi_index = isinstance(data.columns, pd.MultiIndex)
        
        for symbol in symbols:
            try:
                if is_multi_index:
                    if symbol in data.columns.levels[0]:
                        symbol_data = data[symbol].dropna(how='all').reset_index()
                    else:
                        results.append(None)
                        continue
                else:
                    symbol_data = data.dropna(how='all').reset_index()

                if not symbol_data.empty and 'Close' in symbol_data.columns:
                    symbol_data['Symbol'] = symbol
                    cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    symbol_data = symbol_data[[c for c in cols if c in symbol_data.columns]]
                    results.append(symbol_data)
                else:
                    results.append(None)
            except Exception:
                results.append(None)
        return results
    except Exception as e:
        print(f"Failed to download data for symbols {symbols}: {e}")
        return [None] * len(symbols)

if __name__ == "__main__":
    download_price_data()