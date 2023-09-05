import yfinance as yf
import datetime
from config import *
import logging

def calculate_rs_rating(symbol):
    rs_ratings = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=98)
    end_date = datetime.datetime.now()
    logging.basicConfig(level=logging.CRITICAL)
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads = True)
    logging.basicConfig(level=logging.WARNING)
    if stock_data.empty:
        return None
    now_price = stock_data["Adj Close"][-1]
    try:
        if now_price < 12 or now_price == None:
            return None
        else:
            c_1m = stock_data["Adj Close"].div(stock_data["Adj Close"].shift(days_per_month-1))
            c_2m = stock_data["Adj Close"].div(stock_data["Adj Close"].shift((days_per_month*2)-1))
            c_3m = stock_data["Adj Close"].div(stock_data["Adj Close"].shift((days_per_month*3)-1))
            rs_rating = (c_1m*rs_month_weight[0] + c_2m*rs_month_weight[1] + c_3m*rs_month_weight[2])[-1] * 100
            rs_ratings.append((symbol, rs_rating))
            return rs_ratings
    except:
        return None

with open(stocks_csv_path, "r") as f: 
        symbols = [line.strip() for line in f]
for symbol in symbols:
    print(calculate_rs_rating(symbol))