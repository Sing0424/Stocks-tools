import yfinance as yf
import datetime
import time
from config import *
from multiprocessing import Pool
from tqdm import tqdm
import xlsxwriter
import logging

def calculate_rs_rating(symbol):
    rs_ratings = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=120)
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
            c_1m = now_price / stock_data["Adj Close"][-(days_per_month)]
            c_2m = now_price / stock_data["Adj Close"][-(days_per_month*2)]
            c_3m = now_price / stock_data["Adj Close"][-(days_per_month*3)]
            rs_rating = (c_1m*rs_month_weight[0] + c_2m*rs_month_weight[1] + c_3m*rs_month_weight[2]) * 100
            rs_ratings.append((symbol, rs_rating))
            return rs_ratings
    except:
        return None

def run_rs_data_program():
    # Get the RS ratings using multiprocessing
    with open(stocks_csv_path, "r") as f: 
        symbols = [line.strip() for line in f]

    cpu_count = os.cpu_count() / 2
    pool = Pool(processes=int(cpu_count), maxtasksperchild=1)
    rs_rating_list = []
    process_bar = tqdm(desc='Calculating RS', unit=' stocks', total=len(symbols), ncols=80, smoothing=1, miniters=cpu_count)

    for result in pool.imap_unordered(calculate_rs_rating, symbols):
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

if __name__ == '__main__':
    run_rs_data_program()