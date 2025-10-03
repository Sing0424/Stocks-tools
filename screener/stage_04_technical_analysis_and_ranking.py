# stage_04_technical_analysis.py

import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from config import Config
from io import StringIO
from curl_cffi import requests as cffi_requests
from contextlib import redirect_stderr
import yfinance as yf


def analyze_stock(args):
    symbol, df = args
    try:
        df = df.sort_values('Date').set_index('Date')
        if len(df) < 252:
            return None
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA150'] = df['Close'].rolling(150).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        latest = df.iloc[-1]
        p = latest['Close']
        s50, s150, s200 = latest['SMA50'], latest['SMA150'], latest['SMA200']
        if pd.isna([s50, s150, s200]).any():
            return None
        high52w = df['Close'][-252:].max()
        low52w = df['Close'][-252:].min()
        if len(df) < 30:
            return None
        close_volume_30d = (df['Close'][-30:] * df['Volume'][-30:]).mean()
        conds = [
            p > 12,
            p > s150 and p > s200,
            s150 > s200,
            s200 > df['SMA200'].iloc[-21],
            s50 > s150 and s50 > s200,
            p > s50,
            p >= low52w * 1.25,
            p >= high52w * 0.75,
            close_volume_30d > 10000000
        ]
        if all(conds):
            p_ = df['Close'].iloc[-1]
            p_3m = df['Close'].iloc[-63]
            p_6m = df['Close'].iloc[-126]
            p_9m = df['Close'].iloc[-189]
            p_12m = df['Close'].iloc[-252]
            rs_score = ((p_ / p_3m)*0.4 + (p_ / p_6m)*0.2 + (p_ / p_9m)*0.2 + (p_ / p_12m)*0.2) * 100
            session = cffi_requests.Session(impersonate="chrome110")
            try:
                with redirect_stderr(StringIO()):
                    info = yf.Ticker(symbol, session=session).info
                    industry = info.get('industry', 'N/A')
                    # print(industry)
                    sector = info.get('sector', 'N/A')
                    # print(sector)
            except Exception:
                industry = 'N/A'
                sector = 'N/A'
            return {
                'symbol': symbol,
                'industry': industry,
                'sector': sector,
                'price': p_,
                'high_52w': high52w,
                'low_52w': low52w,
                'rs_score': rs_score,
                'avg_close_volume_30d': close_volume_30d
            }
        else:
            return None
    except:
        return None

def analyze_and_rank():
    print(f"[{datetime.now()}] Stage 4: Analyzing consolidated data and calculating RS rank...")
    if not os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE):
        print("Run stage 3 first.")
        return False
    df_all = pd.read_csv(Config.CONSOLIDATED_PRICE_DATA_FILE)
    df_all['Date'] = pd.to_datetime(df_all['Date'], utc=True)
    grouped = df_all.groupby('Symbol')
    args = [(sym, group) for sym, group in grouped]
    with Pool(processes=Config.MAX_WORKERS) as pool:
        results = list(tqdm(pool.imap(analyze_stock, args), total=len(args)))
    # warnings.resetwarnings()
    filtered = [r for r in results if r]
    if not filtered:
        print("No stocks passed the initial analysis.")
        return False
    
    df = pd.DataFrame(filtered)
    df['rs_rank'] = df['rs_score'].rank(pct=True) * 100
    final = df[df['rs_rank'] >= Config.MIN_RS_RANK].sort_values('rs_rank', ascending=False)

    # Reorder columns for final output
    cols_order = [
        'symbol', 'industry', 'sector', 'price', 'rs_rank', 'rs_score',
        'high_52w', 'low_52w', 'avg_close_volume_30d'
    ]
    # Filter to columns that exist in the dataframe to avoid errors
    final_cols = [col for col in cols_order if col in final.columns]
    final = final[final_cols]

    final.to_csv(Config.FINAL_RESULTS_FILE, index=False)
    # final.to_csv(Config.FINAL_RESULTS_FILE_WEBAPP, index=False)
    print(f"{len(final)} stocks meet RS criteria.")
    return True

if __name__ == "__main__":
    analyze_and_rank()
