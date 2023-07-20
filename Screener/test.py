import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_Top_30_path, screen_result_path
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from tqdm import tqdm
import datetime

symbol = 'NVDA'

yq_stock_data = yq.Ticker(symbol)
inc_stat = yq_stock_data.income_statement('q')

rev_list = pd.DataFrame(inc_stat['TotalRevenue']).dropna()
lenth_rev_list = len(rev_list)
first_qtr_rev = rev_list.iloc[lenth_rev_list-5,0]
second_qtr_rev = rev_list.iloc[lenth_rev_list-4,0]
third_qtr_rev = rev_list.iloc[lenth_rev_list-2,0]
current_qtr_rev = rev_list.iloc[lenth_rev_list-3,0]

print(rev_list)
print('---------------------------------------------------------------------------------------------------')
print(first_qtr_rev)
print(second_qtr_rev)
print(third_qtr_rev)
print(current_qtr_rev)