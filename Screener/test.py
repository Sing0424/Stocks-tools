import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time

# Screen_result_list = []
# import_data = pd.read_excel(daily_rs_rating_path)
# symbols = import_data["Symbol"]
symbols = ['NVDA']
for symbol in symbols:
    ticker_data = yf.Ticker(symbol)
    print(ticker_data.info)
    # try:
    #     eps_list = ticker_data.get_earnings_dates().reset_index(drop=True).dropna()['Reported EPS']
    # except:
    #     pass
    # lenth_eps_list = len(eps_list)
    # print(lenth_eps_list)
    # if lenth_eps_list >= 7:
    #     EPS_QoQ_C = round(((eps_list.iloc[0] - eps_list.iloc[4]) / eps_list.iloc[4]) * 100)
    #     EPS_QoQ_LQ = round(((eps_list.iloc[1] - eps_list.iloc[5]) / eps_list.iloc[5]) * 100)
    #     EPS_QoQ_L2Q = round(((eps_list.iloc[2] - eps_list.iloc[6]) / eps_list.iloc[6]) * 100)

    # inc_stat_a = ticker_data.income_stmt
    # EPS_list_A = inc_stat_a.loc['Diluted EPS']
    # lenth_EPS_list_A = len(EPS_list_A)
    # if lenth_EPS_list_A >=2:
    #     EPS_A = round(((EPS_list_A.iloc[0] - EPS_list_A.iloc[1]) / EPS_list_A.iloc[1]) * 100)
    #     EPS_A1 = round(((EPS_list_A.iloc[1] - EPS_list_A.iloc[2]) / EPS_list_A.iloc[2]) * 100)
    # elif lenth_EPS_list_A >=1:
    #     EPS_A = round(((EPS_list_A.iloc[0] - EPS_list_A.iloc[1]) / EPS_list_A.iloc[1]) * 100)
    #     EPS_A1 = 0

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


