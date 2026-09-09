# stage_05_market_breadth.py

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from openpyxl.chart import LineChart, Reference

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import Config

def calculate_market_breadth():
    """
    Calculates the percentage of stocks trading above their 50-day moving average.
    Uses eligible stocks (those with at least 50 days of history) as denominator.
    """
    print(f"[{datetime.now()}] Stage 5: Calculating Market Breadth (Stocks > 50-Day SMA)...")
    
    # 1. Read consolidated price data (prefer Parquet)
    if Config.USE_PARQUET and os.path.exists(Config.CONSOLIDATED_PRICE_DATA_PARQUET):
        try:
            price_df_long = pd.read_parquet(Config.CONSOLIDATED_PRICE_DATA_PARQUET)
        except Exception as e:
            print(f"Error reading {Config.CONSOLIDATED_PRICE_DATA_PARQUET}: {e}")
            return False
    elif os.path.exists(Config.CONSOLIDATED_PRICE_DATA_FILE):
        try:
            price_df_long = pd.read_csv(Config.CONSOLIDATED_PRICE_DATA_FILE)
        except Exception as e:
            print(f"Error reading {Config.CONSOLIDATED_PRICE_DATA_FILE}: {e}")
            return False
    else:
        print("Consolidated price data file not found. Run Stage 3 first.")
        return False

    price_df_long['Date'] = pd.to_datetime(price_df_long['Date']).dt.tz_localize(None)

    # 2. Pivot the data
    price_df_wide = price_df_long.pivot(index='Date', columns='Symbol', values='Close').sort_index()
    # Forward-fill limited to 5 trading days to prevent dead/halted tickers from persisting forever
    price_df_wide.ffill(limit=5, inplace=True)

    # 3. Calculate 50-day SMA for each stock
    sma50_df = price_df_wide.rolling(window=50, min_periods=50).mean()

    # 4. Count stocks above their 50-day SMA
    above_sma_df = price_df_wide > sma50_df
    above_sma_count = above_sma_df.sum(axis=1)
    
    # 5. True denominator: only stocks that actually have a 50-day SMA on that date
    total_eligible_stocks = sma50_df.notna().sum(axis=1)
    above_sma_percentage = (above_sma_count / total_eligible_stocks.replace(0, np.nan)) * 100
    
    # 6. Create the result DataFrame
    market_breadth_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in price_df_wide.index],
        'Above_SMA50_Count': above_sma_count.values,
        'Eligible_Universe_Count': total_eligible_stocks.values,
        'Above_SMA50_Percentage': above_sma_percentage.round(2).values
    })

    # Skip first 49 days where SMA50 is not yet formed
    market_breadth_df = market_breadth_df.iloc[49:].copy()

    # 7. Save the results to Excel
    latest_count = market_breadth_df['Above_SMA50_Count'].iloc[-1]
    latest_pct = market_breadth_df['Above_SMA50_Percentage'].iloc[-1]
    latest_eligible = market_breadth_df['Eligible_Universe_Count'].iloc[-1]
    print(f"Latest Market Breadth: {latest_count}/{latest_eligible} stocks ({latest_pct}%) above 50-day SMA.")
    
    try:
        with pd.ExcelWriter(Config.EXCEL_REPORT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            market_breadth_df.to_excel(writer, sheet_name='Market Breadth (SMA50)', index=False)

            worksheet = writer.sheets['Market Breadth (SMA50)']

            # Create a line chart
            chart = LineChart()
            chart.title = "Market Breadth: % of Stocks Above 50-Day SMA"
            chart.style = 13
            chart.y_axis.title = "Percentage (%)"
            chart.x_axis.title = "Date"
            chart.height = 12
            chart.width = 22

            # Data range: Col 4 is Above_SMA50_Percentage
            data = Reference(worksheet, min_col=4, min_row=1, max_col=4, max_row=len(market_breadth_df) + 1)
            chart.add_data(data, titles_from_data=True)

            # Category range: Col 1 is Date
            cats = Reference(worksheet, min_col=1, min_row=2, max_col=1, max_row=len(market_breadth_df) + 1)
            chart.set_categories(cats)

            # Place chart at cell F2
            worksheet.add_chart(chart, "F2")
        print(f"Market breadth data and chart appended to {Config.EXCEL_REPORT_FILE}")
    except FileNotFoundError:
        print(f"Error: Excel file not found at {Config.EXCEL_REPORT_FILE}. Run Stage 4 first.")
        return False
    except Exception as e:
        print(f"Error appending to Excel file: {e}")
        return False
        
    return True

def analyze_sector_performance():
    """
    Analyzes sector performance based on the True IBD RS ranks of qualified stocks.
    """
    print(f"[{datetime.now()}] Analyzing Sector Performance...")
    
    if not os.path.exists(Config.EXCEL_REPORT_FILE):
        print(f"Excel report file not found at {Config.EXCEL_REPORT_FILE}. Run Stage 4 first.")
        return False
        
    try:
        results_df = pd.read_excel(Config.EXCEL_REPORT_FILE, sheet_name='Screening Results')
    except Exception as e:
        print(f"Error reading 'Screening Results' sheet from {Config.EXCEL_REPORT_FILE}: {e}")
        return False

    if 'sector' not in results_df.columns or 'rs_rank' not in results_df.columns:
        print("The 'Screening Results' sheet must contain 'sector' and 'rs_rank' columns.")
        return False

    # Filter out empty or N/A sectors if any
    valid_sectors = results_df[results_df['sector'].notna() & (results_df['sector'] != 'N/A')].copy()
    if valid_sectors.empty:
        valid_sectors = results_df.copy()

    sector_performance = valid_sectors.groupby('sector')['rs_rank'].agg(['mean', 'count']).round(2)
    sector_performance.rename(columns={'mean': 'average_rs_rank', 'count': 'stock_count'}, inplace=True)
    sector_performance.sort_values(['average_rs_rank', 'stock_count'], ascending=[False, False], inplace=True)
    sector_performance.reset_index(inplace=True)
    
    try:
        with pd.ExcelWriter(Config.EXCEL_REPORT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            sector_performance.to_excel(writer, sheet_name='Sector Performance', index=False)
        print(f"Sector performance analysis appended to {Config.EXCEL_REPORT_FILE}")
    except Exception as e:
        print(f"Error appending to Excel file: {e}")
        return False

    print("\n--- Top Performing Sectors ---")
    print(sector_performance.to_string(index=False))
    
    return True

def perform_reporting_analysis():
    """Runs both market breadth and sector performance analysis."""
    breadth_ok = calculate_market_breadth()
    sector_ok = analyze_sector_performance()
    return breadth_ok and sector_ok

if __name__ == "__main__":
    perform_reporting_analysis()
