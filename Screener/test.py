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
    
    print(yq_data.income_statement(frequency="a", trailing=False)["DilutedEPS"])

    #EPS Q/Q


