# stage_02_filter_symbols.py
import os
import pandas as pd
from datetime import datetime
from config import Config

def filter_symbols():
    print(f"[{datetime.now()}] Stage 2: Filtering symbols...")
    if not os.path.exists(Config.LISTING_STATUS_FILE):
        print("Run stage 1 first.")
        return False
    if os.path.exists(Config.FILTERED_SYMBOLS_FILE) and not Config.FORCE_REFRESH_FILTERS:
        print("Filtered symbols file exists. Set FORCE_REFRESH_FILTERS=True to re-filter.")
        return True
    try:
        df = pd.read_csv(Config.LISTING_STATUS_FILE, on_bad_lines='skip')
        initial = len(df)
        # Ensure essential columns have no missing values and filter out unwanted security types.
        df.dropna(subset=['symbol', 'name'], inplace=True)
        df = df[df['assetType'] != 'ETF']
        df = df[~df['name'].str.contains('Warrants|Units', case=False, na=False)]
        df = df[~df['symbol'].str.contains(r'[\.\+\$\^\-=()]', na=False, regex=True)] # Exclude symbols with special characters
        df[['symbol']].to_csv(Config.FILTERED_SYMBOLS_FILE, index=False)
        print(f"Filtered {initial - len(df)} ETFs, non-standard symbols, Warrants, and Units.")
        print(f"{len(df)} symbols remain.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    filter_symbols()
