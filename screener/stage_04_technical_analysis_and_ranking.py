# stage_04_technical_analysis_and_ranking.py

import os
import sys
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def send_telegram_alert(message):
    """Sends a Telegram alert if configured, with exception handling."""
    if not Config.ENABLE_TELEGRAM:
        logging.info("Telegram alerts disabled (TG_BOT_TOKEN or TG_CHAT_ID not configured).")
        return
    try:
        import telegram
        bot = telegram.Bot(Config.TG_BOT_TOKEN)
        async with bot:
            await bot.send_message(text=message, chat_id=Config.TG_CHAT_ID)
        logging.info("Telegram alert sent successfully.")
    except Exception as e:
        logging.warning(f"Failed to send Telegram alert: {e}")

def get_stock_metadata(symbol, max_retries=3, sleep_time=5):
    """
    Fetches industry, sector, and quarterly EPS change for a single stock.
    Isolates .info and quarterly_financials so EPS failures do not wipe out sector/industry.
    """
    industry = 'N/A'
    sector = 'N/A'
    eps_growth = 'N/A'

    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            # 1. Fetch sector & industry
            try:
                info = ticker.info or {}
                industry = str(info.get('industry', 'N/A'))
                sector = str(info.get('sector', 'N/A'))
            except Exception as e_info:
                logging.debug(f"Could not fetch info for {symbol}: {e_info}")

            # 2. Separately fetch quarterly EPS growth
            try:
                qf = ticker.quarterly_financials
                if qf is not None and not qf.empty and 'Diluted EPS' in qf.index:
                    eps_series = qf.loc['Diluted EPS'].dropna()
                    if len(eps_series) >= 2:
                        eps_curr = float(eps_series.iloc[0])
                        eps_prev = float(eps_series.iloc[1])
                        if eps_prev != 0:
                            eps_growth = round(((eps_curr - eps_prev) / abs(eps_prev)) * 100, 2)
            except Exception as e_eps:
                logging.debug(f"EPS calculation error for {symbol}: {e_eps}")

            return {
                'symbol': symbol,
                'industry': industry,
                'sector': sector,
                'EPS_%': eps_growth
            }
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(sleep_time * (attempt + 1))
                    continue
            logging.debug(f"Could not fetch metadata for {symbol}: {e}")
            break

    return {
        'symbol': symbol,
        'industry': industry,
        'sector': sector,
        'EPS_%': eps_growth
    }

def analyze_and_rank():
    """
    Performs vectorized technical analysis, calculates True IBD CAN SLIM RS Ranking
    across the entire market universe, fetches metadata concurrently, and writes
    a clean Excel workbook with dedicated summary information.
    """
    logging.info("Stage 4: Starting technical analysis and market-wide RS ranking...")

    # Step 1: Load consolidated price data (prefer Parquet, fallback to CSV)
    data_file = None
    if Config.USE_PARQUET and os.path.exists(Config.CONSOLIDATED_PRICE_DATA_PARQUET):
        data_file = Config.CONSOLIDATED_PRICE_DATA_PARQUET
        logging.info(f"Loading price data from Parquet: {data_file}")
        df_all = pd.read_parquet(data_file)
    elif os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE):
        data_file = Config.CONSOLIDATED_PRICE_DATA_FILE
        logging.info(f"Loading price data from CSV: {data_file}")
        df_all = pd.read_csv(data_file)
    else:
        logging.error("Consolidated price data file not found. Run Stage 3 first.")
        return False

    df_all['Date'] = pd.to_datetime(df_all['Date']).dt.tz_localize(None)

    # Step 2: Pivot tables for vectorized calculations
    logging.info("Pivoting price and volume tables for vectorized evaluation...")
    close_wide = df_all.pivot(index='Date', columns='Symbol', values='Close').sort_index()
    high_wide = df_all.pivot(index='Date', columns='Symbol', values='High').sort_index()
    low_wide = df_all.pivot(index='Date', columns='Symbol', values='Low').sort_index()
    vol_wide = df_all.pivot(index='Date', columns='Symbol', values='Volume').sort_index()

    total_universe_count = len(close_wide.columns)
    logging.info(f"Loaded {total_universe_count} stocks in market universe across {len(close_wide)} dates.")

    if len(close_wide) < 252:
        logging.error(f"Insufficient historical data: {len(close_wide)} dates found, minimum 252 required.")
        return False

    # Step 3: Vectorized True IBD CAN SLIM Relative Strength Calculation
    logging.info("Calculating True IBD CAN SLIM RS scores and percentiles across all stocks...")
    p_now = close_wide.iloc[-1]
    p_3m = close_wide.iloc[-63]
    p_6m = close_wide.iloc[-126]
    p_9m = close_wide.iloc[-189]
    p_12m = close_wide.iloc[-252]

    # Validate price points
    valid_rs_mask = (p_now.notna()) & (p_3m > 0) & (p_6m > 0) & (p_9m > 0) & (p_12m > 0)

    rs_score = pd.Series(np.nan, index=close_wide.columns)
    rs_score[valid_rs_mask] = (
        ((p_now[valid_rs_mask] / p_3m[valid_rs_mask]) - 1) * Config.RS_WEIGHT_3M +
        ((p_now[valid_rs_mask] / p_6m[valid_rs_mask]) - 1) * Config.RS_WEIGHT_6M +
        ((p_now[valid_rs_mask] / p_9m[valid_rs_mask]) - 1) * Config.RS_WEIGHT_9M +
        ((p_now[valid_rs_mask] / p_12m[valid_rs_mask]) - 1) * Config.RS_WEIGHT_12M
    ) * 100

    # True IBD CAN SLIM RS Rank: Percentile rank across ALL stocks in universe (1-100)
    rs_rank = rs_score.rank(pct=True) * 100

    # Step 4: Vectorized Mark Minervini Trend Template & Liquidity Criteria
    logging.info("Applying Minervini Trend Template and liquidity criteria vectorially...")
    sma50 = close_wide.rolling(50).mean().iloc[-1]
    sma150 = close_wide.rolling(150).mean().iloc[-1]
    sma200 = close_wide.rolling(200).mean().iloc[-1]
    sma200_trend = close_wide.rolling(200).mean().iloc[-Config.SMA200_TREND_DAYS]

    high_52w = high_wide.iloc[-252:].max()
    low_52w = low_wide.iloc[-252:].min()
    avg_dollar_vol_30d = (close_wide.iloc[-30:] * vol_wide.iloc[-30:]).mean()

    # Criteria evaluation
    cond_price = p_now > Config.MIN_PRICE
    cond_above_150_200 = (p_now > sma150) & (p_now > sma200)
    cond_150_above_200 = sma150 > sma200
    cond_200_trend = sma200 > sma200_trend
    cond_50_above_both = (sma50 > sma150) & (sma50 > sma200)
    cond_above_50 = p_now > sma50
    cond_above_low = p_now >= (low_52w * Config.MIN_52W_LOW_INCREASE_FACTOR)
    cond_near_high = p_now >= (high_52w * Config.MIN_52W_HIGH_DECREASE_FACTOR)
    cond_volume = avg_dollar_vol_30d > Config.MIN_CLOSE_VOLUME_30D
    cond_rs = rs_rank >= Config.MIN_RS_RANK

    passed_mask = (
        valid_rs_mask &
        cond_price &
        cond_above_150_200 &
        cond_150_above_200 &
        cond_200_trend &
        cond_50_above_both &
        cond_above_50 &
        cond_above_low &
        cond_near_high &
        cond_volume &
        cond_rs
    )

    passed_symbols = close_wide.columns[passed_mask].tolist()
    logging.info(f"{len(passed_symbols)} stocks passed all technical criteria and RS Rank >= {Config.MIN_RS_RANK}.")

    if not passed_symbols:
        logging.warning("No stocks passed the technical and RS screening criteria.")
        return False

    # Step 5: Build candidates DataFrame
    candidates_df = pd.DataFrame({
        'symbol': passed_symbols,
        'price': p_now[passed_symbols].round(2).values,
        'rs_rank': rs_rank[passed_symbols].round(2).values,
        'rs_score': rs_score[passed_symbols].round(2).values,
        'high_52w': high_52w[passed_symbols].round(2).values,
        'low_52w': low_52w[passed_symbols].round(2).values,
        'avg_close_volume_30d': avg_dollar_vol_30d[passed_symbols].round(0).values
    }).sort_values('rs_rank', ascending=False)

    # Step 6: Fetch metadata concurrently with ThreadPoolExecutor
    logging.info(f"Fetching metadata for {len(candidates_df)} qualifying stocks concurrently...")
    metadata_records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_sym = {executor.submit(get_stock_metadata, sym): sym for sym in candidates_df['symbol']}
        for future in tqdm(as_completed(future_to_sym), total=len(future_to_sym), desc="Metadata"):
            metadata_records.append(future.result())

    metadata_df = pd.DataFrame(metadata_records)
    final_df = pd.merge(candidates_df, metadata_df, on='symbol')

    # Filter out Biotechnology industry
    before_bio = len(final_df)
    final_df = final_df[final_df['industry'] != 'Biotechnology'].copy()
    logging.info(f"Filtered out {before_bio - len(final_df)} Biotechnology stocks. {len(final_df)} stocks remaining.")

    cols_order = [
        'symbol', 'industry', 'sector', 'price', 'rs_rank', 'rs_score',
        'high_52w', 'low_52w', 'avg_close_volume_30d', 'EPS_%'
    ]
    final_df = final_df[[c for c in cols_order if c in final_df.columns]].sort_values('rs_rank', ascending=False)

    # Step 7: Clean Excel Output with Dedicated Summary Sheet
    logging.info(f"Saving {len(final_df)} screened stocks to {Config.EXCEL_REPORT_FILE}...")
    symbols_list = final_df['symbol'].tolist()
    finviz_url = f"https://finviz.com/screener.ashx?v=211&t={','.join(symbols_list)}&o=tickersfilter&p=w"

    summary_df = pd.DataFrame([
        {"Metric": "Execution Date", "Value": str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))},
        {"Metric": "Market Universe Evaluated", "Value": total_universe_count},
        {"Metric": "Screened Stocks Count", "Value": len(final_df)},
        {"Metric": "Minimum RS Rank Threshold", "Value": Config.MIN_RS_RANK},
        {"Metric": "Minimum Price ($)", "Value": Config.MIN_PRICE},
        {"Metric": "Minimum 30D Dollar Volume ($)", "Value": f"${Config.MIN_CLOSE_VOLUME_30D:,.0f}"},
        {"Metric": "Finviz Screener URL", "Value": finviz_url}
    ])

    with pd.ExcelWriter(Config.EXCEL_REPORT_FILE, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Screening Results')
        summary_df.to_excel(writer, index=False, sheet_name='Summary')

    # Step 8: Send Telegram Notification
    if Config.ENABLE_TELEGRAM:
        top_symbols = ", ".join(final_df['symbol'].head(10).tolist())
        tg_message = (
            f"📈 Stock Screener Report ({pd.Timestamp.now().strftime('%Y-%m-%d')})\n"
            f"Passed: {len(final_df)} stocks (RS Rank >= {Config.MIN_RS_RANK})\n"
            f"Top 10: {top_symbols}\n\n"
            f"Finviz: {finviz_url}"
        )
        asyncio.run(send_telegram_alert(tg_message))

    logging.info(f"[SUCCESS] Stage 4 complete: {len(final_df)} stocks saved to {Config.EXCEL_REPORT_FILE}")
    logging.info(f"Finviz URL: {finviz_url}")
    return True

if __name__ == "__main__":
    analyze_and_rank()
