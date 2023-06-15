import yfinance as yf
import pandas as pd
import datetime
from config import stocks_csv_path, daily_rs_rating_Top_30_path
from concurrent.futures import ProcessPoolExecutor

# Load the stock symbols from the CSV file into a list
with open(stocks_csv_path, "r") as f:
    symbols = [line.strip() for line in f]

# Define the number of trading days in a quarter
trading_days_per_quarter = 63

# Define a function to calculate the price percentage change for a given stock over a given number of quarters
def calculate_price_change(symbol, quarters):
    start_date = datetime.datetime.now() - datetime.timedelta(
        days=quarters * trading_days_per_quarter
    )
    end_date = datetime.datetime.now()
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    now_price = stock_data["Adj Close"][-1]
    past_price = stock_data["Adj Close"][0]
    price_percentage_change = (now_price / past_price) * 100
    return price_percentage_change, now_price

# Calculate the RS rating for each stock
def calculate_rs_rating(symbol):
    rs_ratings = []
    print(symbol)
    try:
        c_q1, now_price = calculate_price_change(symbol, 1)
        if now_price < 10:
            print(f"Price under 10")
            return None
        c_q2 = calculate_price_change(symbol, 2)[0]
        c_q3 = calculate_price_change(symbol, 3)[0]
        c_q4 = calculate_price_change(symbol, 4)[0]
        rs_rating = ((0.4 * c_q1) + (0.2 * c_q2) + (0.2 * c_q3) + (0.2 * c_q4))
        rs_ratings.append((symbol, rs_rating))
        return rs_ratings
    except:
        return None

# Calculate the RS rating for each stock using multiprocessing
def calculate_rs_ratings_multiprocessing(symbols):
    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(calculate_rs_rating, symbols, chunksize=20)
        for result in results:
            if result is not None:
                rs_rating_list.extend(result)

# Get the RS ratings using multiprocessing
rs_rating_list = []
if __name__ == '__main__':
    calculate_rs_ratings_multiprocessing(symbols)

# Sort the list of RS ratings by ascending order
rs_rating_list.sort(key=lambda x: x[1])

# Get the top 30% of RS ratings
num_top_ratings = int(len(rs_rating_list) * 0.3)
top_ratings = rs_rating_list[-num_top_ratings:]

# Write the top ratings to a CSV file
with open(daily_rs_rating_Top_30_path, "w") as f:
    f.write("Symbol,RS Rating\n")
    for rating in top_ratings:
        f.write(f"{rating[0]},{rating[1]}\n")