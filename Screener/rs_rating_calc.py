from functools import cache
import yfinance as yf
import datetime
from config import *
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer
import xlsxwriter

@cache
def calculate_price_change(symbol, quarters):
    start_date = datetime.datetime.now() - datetime.timedelta(
        days=quarters * trading_days_per_quarter
    )
    end_date = datetime.datetime.now()
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    now_price = stock_data["Adj Close"][-1]
    past_price = stock_data["Adj Close"][0]
    price_percentage_change = (now_price / past_price) * 100
    return price_percentage_change, now_price

def calculate_rs_rating(symbol):
    rs_ratings = []
    print(f"{symbol}\n")
    try:
        c_q1, now_price = calculate_price_change(symbol, 1)
        if now_price < 10:
            # print(f"Price under 10\n")
            return None
        c_q2 = calculate_price_change(symbol, 2)[0]
        c_q3 = calculate_price_change(symbol, 3)[0]
        c_q4 = calculate_price_change(symbol, 4)[0]
        rs_rating = ((0.4 * c_q1) + (0.2 * c_q2) + (0.2 * c_q3) + (0.2 * c_q4))
        rs_ratings.append((symbol, rs_rating))
        return rs_ratings
    except:
        return None

def run_rs_data_program():
    # Get the RS ratings using multiprocessing
    with open(stocks_csv_path, "r") as f:
        symbols = [line.strip() for line in f]

    rs_rating_list = []
    if __name__ == '__main__':
        with ProcessPoolExecutor(max_workers=None) as executor:
            results = executor.map(calculate_rs_rating, symbols, chunksize=chunksize) #chunksize = 60 >> runtime: 266.5sec
            for result in results:
                if result is not None:
                    rs_rating_list.extend(result)

    # Sort the list of RS ratings by ascending order
    rs_rating_list.sort(key=lambda x: x[1])

    # Get the top 30% of RS ratings
    num_top_ratings = int(len(rs_rating_list) * top_rating)
    top_ratings = rs_rating_list[-num_top_ratings:]

    workbook = xlsxwriter.Workbook(daily_rs_rating_Top_30_path)
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'Symbol')
    worksheet.write('B1', 'RS Rating')
    row = 1
    col = 0

    for symbol, rs_rating in (top_ratings):
        worksheet.write(row, col, symbol)
        worksheet.write(row, col + 1, rs_rating)
        row += 1
        
    workbook.close()

program_start_time = timer()
run_rs_data_program()
program_end_time = timer()

print("Program runtime: --- %.2f seconds ---" % (program_end_time - program_start_time))