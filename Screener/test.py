import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time

eps_list = []
lenth_eps_list = len(eps_list)
print(f'lenth_eps_list: {lenth_eps_list}')
# Screen_result_list = []
# import_data = pd.read_excel(daily_rs_rating_path)
# symbols = import_data["Symbol"]
# symbol = 'NVDA'
# for symbol in symbols:
#     ticker_data = yf.Ticker(symbol)
    # try:
    #     eps_list = ticker_data.earnings_dates.reset_index(drop=True).dropna()['Reported EPS']
    # except:
    #     pass
    # lenth_eps_list = len(eps_list)
    # if lenth_eps_list >= 5:
    #     YoY_eps = eps_list.iloc[4]
    #     print(f'YoY_eps:{YoY_eps}')
    #     before_last_qtr_eps = eps_list.iloc[2]
    #     print(f'before_last_qtr_eps:{before_last_qtr_eps}')
    #     last_qtr_eps = eps_list.iloc[1]
    #     print(f'last_qtr_eps:{last_qtr_eps}')
    #     current_qtr_eps = eps_list.iloc[0]
    #     print(f'current_qtr_eps:{current_qtr_eps}')
    #     eps_growth_perc_last_qtr = ((current_qtr_eps - last_qtr_eps) / last_qtr_eps) * 100
    #     print(f'eps_growth_perc_last_YoY:{eps_growth_perc_last_qtr}')
    #     eps_growth_perc_yester_qtr = ((last_qtr_eps - before_last_qtr_eps) / before_last_qtr_eps) * 100
    #     print(f'eps_growth_perc_yester_YoY:{eps_growth_perc_yester_qtr}')
    #     eps_growth_perc_current_YoY = ((current_qtr_eps - YoY_eps) / YoY_eps) * 100
    #     print(f'eps_growth_perc_current_YoY:{eps_growth_perc_current_YoY}')



# eps_list = inc_stat_q.loc['Diluted EPS'].dropna()
# last_qtr_eps = eps_list.iloc[1]
# current_qtr_eps = eps_list.iloc[0]
# qtr_eps_growth_perc = (last_qtr_eps / current_qtr_eps) * 100

# print(last_qtr_eps)
# print(current_qtr_eps)
# print(qtr_eps_growth_perc)
# print(stock_data.summary_profile[symbol]['industry'])
# print(stock_data.summary_profile[symbol]['sector'])
# # stock_data = yf.download(tickers = symbol, period='max', progress=False)

# def test_block(symbol):
#     rs_ratings = []
#     start_date = datetime.datetime.now() - datetime.timedelta(weeks=52)
#     end_date = datetime.datetime.now()
#     logging.basicConfig(level=logging.CRITICAL)
#     stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads = True)
#     print(stock_data)
#     logging.basicConfig(level=logging.WARNING)
#     if stock_data.empty:
#         return None
#     now_price = stock_data["Adj Close"][-1]
#     # try:
#     if now_price < 12 or now_price == None:
#         return None
#     else:
#         q1_p = (now_price / stock_data["Adj Close"][-(days_per_month)]) * (rs_month_weight * 2)
#         q2_p = now_price / stock_data["Adj Close"][-(days_per_month*2)] * rs_month_weight
#         q3_p = now_price / stock_data["Adj Close"][-(days_per_month*3)] * rs_month_weight
#         q4_p = now_price / stock_data["Adj Close"][-(days_per_month*4)] * rs_month_weight
#         rs_rating = q1_p + q2_p + q3_p + q4_p
#         print(rs_rating)
#         rs_ratings.append((symbol, rs_rating))
#         return rs_ratings

# test_block(symbols)


