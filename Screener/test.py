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
symbols = ['VRT']
for symbol in symbols:
    ticker_data = yf.Ticker(symbol)
    yq_data = Ticker(symbol)
    stock_data = ticker_data.history(period="max")
    #print(stock_data)
    print("======================================================================================")

    inc_stat_q = ticker_data.quarterly_income_stmt
    inc_stat_a = ticker_data.income_stmt

    REV_Q_LIST = inc_stat_q.loc['Operating Revenue'].dropna()
    if pd.notna(REV_Q_LIST[0]) and pd.notna(REV_Q_LIST[1]):
        REV_Q = round(((REV_Q_LIST.iloc[0] - REV_Q_LIST.iloc[4]) / abs(REV_Q_LIST.iloc[4])) * 100)
    
    print(REV_Q_LIST.iloc[4])

    #EPS Q/Q


