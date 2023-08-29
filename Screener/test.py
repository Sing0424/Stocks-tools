import yfinance as yf
import datetime
from config import *
from concurrent.futures import ProcessPoolExecutor
import xlsxwriter
from tqdm import tqdm
import logging

def calculate_rs_rating(symbol):
    rs_ratings = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=3 * days_per_month)
    end_date = datetime.datetime.now()
    logging.basicConfig(level=logging.CRITICAL)
    stock_data = yf.download(symbols, start=start_date, end=end_date, progress=False, threads = True)
    logging.basicConfig(level=logging.WARNING)
    now_price = stock_data["Adj Close"][-1]
    past_price = stock_data["Adj Close"][0]
    if now_price < 12 or now_price == None:
        return None
    else:
        try:
            c_p1 = stock_data["Adj Close"][-1].div(stock_data["Adj Close"].shift(days_per_month))
            c_p2 = stock_data["Adj Close"].div(stock_data["Adj Close"].shift(2*days_per_month))
            c_p3 = stock_data["Adj Close"].div(dstock_dataf["Adj Close"].shift(3*days_per_month))
            rs_rating = ((0.4 * c_p1) + (0.3 * c_p2) + (0.3 * c_p3)) * 100
            rs_ratings.append((symbol, rs_rating))
            return rs_ratings
        except:
            return None

def run_rs_data_program():
    # Get the RS ratings using multiprocessing
    if __name__ == '__main__':
        with open(stocks_csv_path, "r") as f:
            symbols = [line.strip() for line in f]

        rs_rating_list = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = tqdm(executor.map(calculate_rs_rating, symbols, chunksize=chunksize), desc='Calculating RS', unit=' stocks', total=len(symbols), ncols=80, miniters=1)
            for result in results:
                if result is not None:
                    rs_rating_list.extend(result)

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