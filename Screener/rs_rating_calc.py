import yfinance as yf
import datetime
from config import *
from multiprocessing import Pool
from tqdm import tqdm
import xlsxwriter
import logging
import time

def calculate_rs_rating(symbol):
    rs_ratings = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=98)
    end_date = datetime.datetime.now()
    logging.basicConfig(level=logging.CRITICAL)
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads = True)
    logging.basicConfig(level=logging.WARNING)
    try:
        now_price = stock_data["Adj Close"][-1]
        if now_price > 12:
            c_1m = (stock_data["Adj Close"].div(stock_data["Adj Close"].shift(days_per_month)))
            c_2m = (stock_data["Adj Close"].div(stock_data["Adj Close"].shift(days_per_month*2)))
            c_3m = (stock_data["Adj Close"].div(stock_data["Adj Close"].shift(days_per_month*3)))
            rs_rating = (c_1m*rs_month_weight[0] + c_2m*rs_month_weight[1] + c_3m*rs_month_weight[2])[-1] * 100
            rs_ratings.append((symbol, rs_rating))
            return rs_ratings
        else:
            rs_rating = 0
            return rs_ratings
    except:
        rs_rating = 0
        rs_ratings.append((symbol, rs_rating))
        return rs_ratings

def run_rs_data_program():
    if __name__ == '__main__':
        # Get the RS ratings using multiprocessing
        with open(stocks_csv_path, "r") as f: 
            symbols = [line.strip() for line in f]

        num_cpus = os.cpu_count()
        pool = Pool(processes=int(num_cpus/2), maxtasksperchild=1)
        rs_rating_list = []
        process_bar = tqdm(desc='Calculating RS', unit=' stocks', total=len(symbols), ncols=80, smoothing=1, miniters=1)

        for result in pool.imap_unordered(calculate_rs_rating, symbols, chunksize=chunksize):
            if result is not None:
                rs_rating_list.extend(result)
            process_bar.update()
        process_bar.close()

        # Sort the list of RS ratings by ascending order
        rs_rating_list.sort(key=lambda x: x[1], reverse=True)

        # Get the top 30% of RS ratings
        num_top_ratings = int(len(rs_rating_list) * top_rating)
        top_ratings = rs_rating_list[:num_top_ratings]

        workbook = xlsxwriter.Workbook(daily_rs_rating_Top_30_path)
        worksheet = workbook.add_worksheet()
        float_pt_round = workbook.add_format({'num_format': '#,#####0.00000', 'border': 1})
        worksheet.write('A1', 'Symbol')
        worksheet.write('B1', 'RS Rating')
        row = 1
        col = 0
        
        for symbol, rs_rating in (top_ratings):
            worksheet.write(row, col, symbol)
            worksheet.write(row, col + 1, rs_rating, float_pt_round)
            row += 1
            
        workbook.close()

run_rs_data_program()