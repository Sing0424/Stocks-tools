import os
import pandas as pd
import yfinance as yf
from yahooquery import Ticker
from config import daily_rs_rating_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time
import logging

# Screen_result_list = []
# import_data = pd.read_excel(daily_rs_rating_path)
# symbols = import_data["Symbol"]
symbols = ['SMCI']
for symbol in symbols:
    ticker_data = yf.Ticker(symbol)
    stock_data = ticker_data.history(period="max")
    #print(stock_data)
    print("======================================================================================")

    inc_stat_q = ticker_data.quarterly_income_stmt

    if pd.notna(inc_stat_q.loc['Diluted EPS'][0]) and pd.notna(inc_stat_q.loc['Diluted EPS'][4]):
        EPS_Q = round(((inc_stat_q.loc['Diluted EPS'][0]  / inc_stat_q.loc['Diluted EPS'][4]) - 1) * 100)
    elif pd.isna(inc_stat_q.loc['Diluted EPS'][0]) and pd.notna(inc_stat_q.loc['Diluted EPS'][4]):
        logging.basicConfig(level=logging.CRITICAL)
        EPS_Q = round(((ticker_data.get_earnings_dates(limit=20).reset_index(drop=True).dropna()['Reported EPS'][0]  / inc_stat_q.loc['Diluted EPS'][4]) - 1) * 100)
        logging.basicConfig(level=logging.WARNING)
    else:
        EPS_Q = 0
        
    print(EPS_Q)


