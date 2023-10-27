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
# import_data = pd.read_excel(daily_rs_rating_Top_30_path)

# symbols = import_data["Symbol"]
symbols = ['GD']

# stock_data = yf.download(tickers = symbol, period='max', progress=False)
for symbol in symbols:
    print(symbol)
    ticker_data = yf.Ticker(symbol)
    #sector = info.get('sector')
    #industry = info.get('industry')

    inc_stat = ticker_data.income_stmt
    #Total Revenue
    rev_list = inc_stat.loc['Total Revenue'].dropna()
    #print(inc_list)
    lenth_rev_list = len(rev_list)
    if lenth_rev_list >= 4:
        first_qtr_rev = rev_list.iloc[3]
        second_qtr_rev = rev_list.iloc[2]
        third_qtr_rev = rev_list.iloc[1]
        current_qtr_rev = rev_list.iloc[0]
    elif lenth_rev_list == 3:
        first_qtr_rev = 0
        second_qtr_rev = rev_list.iloc[2]
        third_qtr_rev = rev_list.iloc[1]
        current_qtr_rev = rev_list.iloc[0]
    elif lenth_rev_list == 2:
        first_qtr_rev = 0
        second_qtr_rev = 0
        third_qtr_rev = rev_list.iloc[1]
        current_qtr_rev = rev_list.iloc[0]
    elif lenth_rev_list == 1:
        first_qtr_rev = 0
        second_qtr_rev = 0
        third_qtr_rev = 0
        current_qtr_rev = rev_list.iloc[0]

    print(first_qtr_rev)
    print(second_qtr_rev)
    print(third_qtr_rev)
    print(current_qtr_rev)
