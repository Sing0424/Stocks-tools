import yfinance as yf
import datetime
from config import *
import logging

def calculate_rs_rating(symbol):
    rs_ratings = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=3 * days_per_month)
    end_date = datetime.datetime.now()
    logging.basicConfig(level=logging.CRITICAL)
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads = True)
    logging.basicConfig(level=logging.WARNING)
    now_price = stock_data["Adj Close"][-1]
    past_price = stock_data["Adj Close"][days_per_month]
    if now_price < 12 or now_price == None:
        return None
    else:
        try:
            c_p1 = (stock_data["Adj Close"][-1]/stock_data["Adj Close"][days_per_month])*0.4
            print(c_p1)
            c_p2 = (stock_data["Adj Close"][-1]/stock_data["Adj Close"][2*days_per_month])*0.3
            print(c_p2)
            c_p3 = (stock_data["Adj Close"][-1]/stock_data["Adj Close"][3*days_per_month])*0.3
            print(c_p3)
            rs_rating = c_p1 + c_p2 + c_p3
            rs_ratings.append((symbol, rs_rating))
            return rs_ratings
        except:
            return None

a = calculate_rs_rating('VRT')
print(a)