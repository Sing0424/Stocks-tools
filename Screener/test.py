import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_Top_30_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time

Screen_result_list = []
import_data = pd.read_excel(daily_rs_rating_Top_30_path)

symbols = import_data["Symbol"]
symbols = ['ASPAU']

# stock_data = yf.download(tickers = symbol, period='max', progress=False)
for symbol in symbols:
    ticker_data = yf.Ticker(symbol)
    #sector = info.get('sector')
    #industry = info.get('industry')

    inc_stat = ticker_data.quarterly_income_stmt
    print(inc_stat)
    try:
        eps_list = inc_stat.loc['Diluted EPS']
            # lenth_eps_list = len(eps_list)
        first_qtr_eps = eps_list.iloc[3]
        second_qtr_eps = eps_list.iloc[2]
        third_qtr_eps = eps_list.iloc[1]
        current_qtr_eps = eps_list.iloc[0]
    except:
        first_qtr_eps = 0
        second_qtr_eps = 0
        third_qtr_eps = 0
        current_qtr_eps = 0