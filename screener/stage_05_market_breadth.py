# stage_05_market_breadth.py

import pandas as pd
import os
from datetime import datetime
from config import Config
from openpyxl.chart import LineChart, Reference

def calculate_market_breadth():
    """
    Calculates the percentage of stocks trading above their 50-day moving average.
    """
    print(f"[{datetime.now()}] Stage 5: Calculating Market Breadth (Stocks > 50-Day SMA)...")
    
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

    # Hide the initial rows which are zero due to the rolling window calculation period.
    # The first 49 rows are artifacts of the 50-day rolling window calculation.
    market_breadth_df = market_breadth_df.iloc[49:]

    # 7. Save the results to Excel
    print(f"Number of stocks above SMA50: {above_sma_count.iloc[-1]}")
    print(f"Percentage of stocks above SMA50: {above_sma_percentage.iloc[-1]}")
    
    try:
        with pd.ExcelWriter(Config.EXCEL_REPORT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            market_breadth_df.to_excel(writer, sheet_name='Market Breadth (SMA50)', index=False)

            # Get workbook and worksheet objects
            worksheet = writer.sheets['Market Breadth (SMA50)']

            # Create a line chart
            chart = LineChart()
            chart.title = "Market Breadth: % of Stocks Above 50-Day SMA"
            chart.style = 13
            chart.y_axis.title = "Percentage (%)"
            chart.x_axis.title = "Date"
            chart.height = 10
            chart.width = 20

            # Define data range for the chart (the percentage column)
            data = Reference(worksheet, min_col=3, min_row=1, max_col=3, max_row=len(market_breadth_df) + 1)
            chart.add_data(data, titles_from_data=True)

            # Define category range (the date column for the x-axis)
            cats = Reference(worksheet, min_col=1, min_row=2, max_col=1, max_row=len(market_breadth_df) + 1)
            chart.set_categories(cats)

            # Add chart to the worksheet, placing it at cell E2
            worksheet.add_chart(chart, "G2")
        print(f"Market breadth data and chart appended to {Config.EXCEL_REPORT_FILE}")
    except FileNotFoundError:
        print(f"Error: Excel file not found at {Config.EXCEL_REPORT_FILE}. Run stage 4 first to create it.")
        return False
    except Exception as e:
        print(f"Error appending to Excel file: {e}")
        return False
        
    return True

def analyze_sector_performance():
    """
    Analyzes the performance of different sectors based on the RS ranks of their stocks.
    """
    print(f"[{datetime.now()}] Analyzing Sector Performance...")
    
    # 1. Check if the Excel report file exists
    if not os.path.exists(Config.EXCEL_REPORT_FILE):
        print(f"Excel report file not found at {Config.EXCEL_REPORT_FILE}. Run stage 4 first.")
        return False
        
    # 2. Read the ranked stock data from the Excel file
    try:
        results_df = pd.read_excel(Config.EXCEL_REPORT_FILE, sheet_name='Screening Results')
    except Exception as e:
        print(f"Error reading 'Screening Results' sheet from {Config.EXCEL_REPORT_FILE}: {e}")
        return False

    # 3. Check for required columns
    if 'sector' not in results_df.columns or 'rs_rank' not in results_df.columns:
        print("The 'Screening Results' sheet must contain 'sector' and 'rs_rank' columns.")
        return False

    # 4. Group by sector and calculate performance metrics
    sector_performance = results_df.groupby('sector')['rs_rank'].agg(['mean', 'count'])
    sector_performance.rename(columns={'mean': 'average_rs_rank', 'count': 'stock_count'}, inplace=True)
    
    # 5. Sort sectors by average RS rank
    sector_performance.sort_values('average_rs_rank', ascending=False, inplace=True)
    
    # 6. Save the results to the Excel file
    sector_performance.reset_index(inplace=True)
    
    try:
        with pd.ExcelWriter(Config.EXCEL_REPORT_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            sector_performance.to_excel(writer, sheet_name='Sector Performance', index=False)
        print(f"Sector performance analysis appended to {Config.EXCEL_REPORT_FILE}")
    except Exception as e:
        print(f"Error appending to Excel file: {e}")
        return False

    print("\n--- Top Performing Sectors ---")
    print(sector_performance.head())
    
    return True

def perform_reporting_analysis():
    """Runs both market breadth and sector performance analysis."""
    breadth_ok = calculate_market_breadth()
    sector_ok = analyze_sector_performance()
    return breadth_ok and sector_ok

if __name__ == "__main__":
    perform_reporting_analysis()
