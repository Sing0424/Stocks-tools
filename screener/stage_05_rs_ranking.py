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
    final.to_csv(Config.FINAL_RESULTS_FILE, index=False)
    final.to_csv(Config.FINAL_RESULTS_FILE_WEBAPP, index=False)
    print(f"{len(final)} stocks meet RS criteria.")
    return True

if __name__ == "__main__":
    rank_stocks()
