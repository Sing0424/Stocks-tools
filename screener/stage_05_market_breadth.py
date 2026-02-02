# stage_06_market_breadth.py

import pandas as pd
import os
from datetime import datetime
from config import Config

def calculate_market_breadth():
    """
    Calculates the percentage of stocks trading above their 50-day moving average.
    """
    print(f"[{datetime.now()}] Stage 6: Calculating Percentage of Stocks Above 50-Day SMA...")
    
    # 1. Read consolidated price data
    if not os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE):
        print(f"Consolidated price data file not found at {Config.CONSOLIDATED_PRICE_DATA_FILE}. Run stage 3 first.")
        return False
        
    try:
        price_df_long = pd.read_csv(Config.CONSOLIDATED_PRICE_DATA_FILE)
    except Exception as e:
        print(f"Error reading {Config.CONSOLIDATED_PRICE_DATA_FILE}: {e}")
        return False

    # 2. Pivot the data
    price_df_wide = price_df_long.pivot(index='Date', columns='Symbol', values='Close')
    price_df_wide.ffill(inplace=True)

    # 3. Calculate 50-day SMA for each stock
    sma50_df = price_df_wide.rolling(window=50).mean()

    # 4. Count stocks above their 50-day SMA
    above_sma_df = price_df_wide > sma50_df
    above_sma_count = above_sma_df.sum(axis=1)
    
    # 5. Calculate the percentage of stocks above SMA50
    total_stocks = price_df_wide.notna().sum(axis=1)
    above_sma_percentage = (above_sma_count / total_stocks) * 100
    
    # 6. Create the result DataFrame
    market_breadth_df = pd.DataFrame({
        'Date': price_df_wide.index,
        'Above_SMA50_Count': above_sma_count,
        'Above_SMA50_Percentage': above_sma_percentage
    })

    # 7. Save the results to Excel
    print(f"Number of stocks above SMA50: {above_sma_count.iloc[-1]}")
    print(f"Percentage of stocks above SMA50: {above_sma_percentage.iloc[-1]}")
    
    try:
        with pd.ExcelWriter(Config.EXCEL_REPORT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            market_breadth_df.to_excel(writer, sheet_name='Market Breadth (SMA50)', index=False)
        print(f"Market breadth data appended to {Config.EXCEL_REPORT_FILE}")
    except FileNotFoundError:
        print(f"Error: Excel file not found at {Config.EXCEL_REPORT_FILE}. Run stage 4 first to create it.")
        return False
    except Exception as e:
        print(f"Error appending to Excel file: {e}")
        return False
        
    return True

if __name__ == "__main__":
    calculate_market_breadth()
