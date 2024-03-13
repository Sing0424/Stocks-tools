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
symbols = ['MSTR']
for symbol in symbols:
    ticker_data = yf.Ticker(symbol)
    stock_data = ticker_data.history(period="max")
    #print(stock_data)
    print("======================================================================================")

    inc_stat_q = ticker_data.quarterly_income_stmt

    #EPS Q/Q
    
    EPS_Q = ticker_data.get_earnings_dates(limit=20)['Reported EPS'].dropna().reset_index(drop=True)[0]
        
    print(EPS_Q)


