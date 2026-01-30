import pandas as pd
import os
import logging
import time
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from config import Config
import csv
import yfinance as yf
import asyncio
import telegram

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def send_telegram_message(message):
    """
    Sends a message to a specified Telegram chat.
    """
    bot = telegram.Bot(token=Config.TG_BOT_TOKEN)
    async with bot:
        await bot.send_message(text=message, chat_id=Config.TG_CHAT_ID)

def analyze_stock(args):
    """
    Analyzes a single stock based on technical indicators and predefined criteria.
    This function is designed to be run in parallel and does not perform network requests.

    Args:
        args (tuple): A tuple containing the stock symbol and its price history DataFrame.

    Returns:
        dict: A dictionary with analysis results if the stock passes the criteria, otherwise None.
    """
    symbol, df = args
    try:
        df = df.sort_values('Date').set_index('Date')

        # Ensure there is enough data for the analysis
        if len(df) < 252:
            return None

        # Calculate SMAs
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA150'] = df['Close'].rolling(150).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()

        latest_data = df.iloc[-1]
        latest_price = latest_data['Close']
        sma50, sma150, sma200 = latest_data['SMA50'], latest_data['SMA150'], latest_data['SMA200']

        if pd.isna([sma50, sma150, sma200]).any():
            return None

        high_52w = df['High'][-252:].max()
        low_52w = df['Low'][-252:].min()
        
        if len(df) < 30:
            return None
        avg_close_volume_30d = (df['Close'][-30:] * df['Volume'][-30:]).mean()

        # Mark Minervini's Trend Template conditions
        conditions = [
            latest_price > Config.MIN_PRICE,
            latest_price > sma150 and latest_price > sma200,
            sma150 > sma200,
            sma200 > df['SMA200'].iloc[-Config.SMA200_TREND_DAYS],
            sma50 > sma150 and sma50 > sma200,
            latest_price > sma50,
            latest_price >= low_52w * Config.MIN_52W_LOW_INCREASE_FACTOR,
            latest_price >= high_52w * Config.MIN_52W_HIGH_DECREASE_FACTOR,
            avg_close_volume_30d > Config.MIN_CLOSE_VOLUME_30D
        ]

        if all(conditions):
            # Calculate custom RS score with bounds checking
            if len(df) < 252:
                return None
                
            price_now = df['Close'].iloc[-1]
            price_3m = df['Close'].iloc[-63] if len(df) >= 63 else None
            price_6m = df['Close'].iloc[-126] if len(df) >= 126 else None
            price_9m = df['Close'].iloc[-189] if len(df) >= 189 else None
            price_12m = df['Close'].iloc[-252] if len(df) >= 252 else None

            if None in [price_3m, price_6m, price_9m, price_12m] or 0 in [price_3m, price_6m, price_9m, price_12m]:
                return None

            rs_score = (
                ((price_now / price_3m) - 1) * Config.RS_WEIGHT_3M +
                ((price_now / price_6m) - 1) * Config.RS_WEIGHT_6M +
                ((price_now / price_9m) - 1) * Config.RS_WEIGHT_9M +
                ((price_now / price_12m) - 1) * Config.RS_WEIGHT_12M
            ) * 100
            
            return {
                'symbol': symbol,
                'price': latest_price,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'rs_score': rs_score,
                'avg_close_volume_30d': avg_close_volume_30d
            }
        else:
            return None
    except Exception as e:
        # Log error but don't include stock symbol to avoid confusion in parallel logs
        logging.error(f"Error analyzing a stock: {e}")
        return None

def get_stock_metadata(symbol, max_retries=5, sleep_time=2):
    """
    Fetches industry and sector for a single stock, with exponential backoff for rate limiting.
    """
    for i in range(max_retries):
        try:
            stock_info = yf.Ticker(symbol).info
            if stock_info:
                return {
                    'industry': str(stock_info.get('industry', 'N/A')),
                    'sector': str(stock_info.get('sector', 'N/A'))
                }
            else:
                logging.warning(f"No info returned for {symbol}")
                return {'industry': 'N/A', 'sector': 'N/A'}
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                if i < max_retries - 1:
                    logging.warning(f"Rate limited on {symbol}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Could not fetch info for {symbol} after {max_retries} retries: {e}")
            else:
                logging.error(f"Could not fetch info for {symbol}: {e}")
                break  # Non-rate-limit error, break loop

    return {
        'industry': 'N/A',
        'sector': 'N/A'
    }

def analyze_and_rank():
    """
    Performs the main analysis and ranking process.
    Reads consolidated data, filters stocks, ranks them, and saves the results.
    """
    logging.info("Stage 4: Analyzing consolidated data and calculating RS rank...")
    if not os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE):
        logging.error("Consolidated price data file not found. Run stage 3 first.")
        return False

    df_all = pd.read_csv(Config.CONSOLIDATED_PRICE_DATA_FILE)
    df_all['Date'] = pd.to_datetime(df_all['Date'], utc=True)
    grouped = df_all.groupby('Symbol')
    args = [(sym, group) for sym, group in grouped]

    # Step 1: Perform CPU-bound analysis in parallel
    logging.info(f"Performing technical analysis on {len(args)} stocks...")
    with Pool(processes=Config.WORKERS) as pool:
        results = list(tqdm(pool.imap(analyze_stock, args), total=len(args)))
    
    filtered_results = [r for r in results if r]
    if not filtered_results:
        logging.warning("No stocks passed the initial technical analysis.")
        return False
    
    df = pd.DataFrame(filtered_results)
    
    # Step 2: Rank and finalize
    df['rs_rank'] = df['rs_score'].rank(pct=True) * 100
    final_df = df[df['rs_rank'] >= Config.MIN_RS_RANK].sort_values('rs_rank', ascending=False)

    # Step 3: Fetch metadata (industry, sector) sequentially to avoid rate limiting
    logging.info(f"Fetching metadata for {len(final_df)} filtered stocks...")
    metadata_list = []
    for symbol in tqdm(final_df['symbol'], total=len(final_df)):
        metadata = get_stock_metadata(symbol)
        metadata['symbol'] = symbol
        metadata_list.append(metadata)
        
    metadata_df = pd.DataFrame(metadata_list)
    final_df = pd.merge(final_df, metadata_df, on='symbol')

    cols_order = [
        'symbol', 'industry', 'sector', 'price', 'rs_rank', 'rs_score',
        'high_52w', 'low_52w', 'avg_close_volume_30d'
    ]
    final_df = final_df[[col for col in cols_order if col in final_df.columns]]

    final_df.to_csv(Config.FINAL_RESULTS_FILE, index=False)
    logging.info(f"{len(final_df)} stocks meet RS criteria.")
    
    if not final_df.empty:
        symbols = ",".join(final_df['symbol'].tolist())
        finviz_url = f"https://finviz.com/screener.ashx?v=211&t={symbols}&o=tickersfilter&p=w"
        
        if len(finviz_url) > 2000:
            logging.warning(f"Finviz URL is very long ({len(finviz_url)} chars). It may not work in all browsers.")

        logging.info("\n--- Finviz URL for Quick View ---")
        logging.info(finviz_url)

        try:
            asyncio.run(send_telegram_message(finviz_url))
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")

        try:
            with open(Config.FINAL_RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow([finviz_url])
            logging.info(f"Finviz URL also saved to {Config.FINAL_RESULTS_FILE}")
        except Exception as e:
            logging.error(f"Could not append URL to CSV file: {e}")
        
    return True

if __name__ == "__main__":
    analyze_and_rank()
