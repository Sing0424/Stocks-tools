# stage_05_rs_ranking.py
import os
import pandas as pd
from datetime import datetime
from config import Config

def rank_stocks():
    print(f"[{datetime.now()}] Stage 5: Calculating RS rank...")
    if not os.path.exists(Config.TECHNICAL_RESULTS_FILE):
        print("Run stage 4 first.")
        return False
    df = pd.read_csv(Config.TECHNICAL_RESULTS_FILE)
    if df.empty:
        print("No data to rank.")
        return False
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

if __name__ == "__main__": # Added this if condition to prevent execution on import
    rank_stocks()
