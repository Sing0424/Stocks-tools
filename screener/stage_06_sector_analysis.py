# stage_07_sector_analysis.py

import pandas as pd
import os
from datetime import datetime
from config import Config

def analyze_sector_performance():
    """
    Analyzes the performance of different sectors based on the RS ranks of their stocks.
    """
    print(f"[{datetime.now()}] Stage 7: Analyzing Sector Performance...")
    
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
    except FileNotFoundError:
        print(f"Error: Excel file not found at {Config.EXCEL_REPORT_FILE}. Make sure it was created in stage 4.")
        return False
    except Exception as e:
        print(f"Error appending to Excel file: {e}")
        return False

    print("\n--- Top Performing Sectors ---")
    print(sector_performance.head())
    
    return True

if __name__ == "__main__":
    analyze_sector_performance()
