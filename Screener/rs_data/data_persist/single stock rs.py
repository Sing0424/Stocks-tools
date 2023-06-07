import yfinance as yf
import pandas as pd
import datetime
from config import stocks_csv_path

# Load the stock symbols from the CSV file into a list
# with open(stocks_csv_path, "r") as f:
#     symbols = [line.strip() for line in f]

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
    return price_percentage_change


# Calculate the RS rating for each stock
symbol = 'NVDA'
rs_ratings = []
# for symbol in symbols:
try:
    c_q1 = calculate_price_change(symbol, 1)
    c_q2 = calculate_price_change(symbol, 2)
    c_q3 = calculate_price_change(symbol, 3)
    c_q4 = calculate_price_change(symbol, 4)
    rs_rating = ((0.4 * c_q1) + (0.2 * c_q2) + (0.2 * c_q3) + (0.2 * c_q4))
    rs_ratings.append((symbol, rs_rating))
except:
    pass

# Sort the list of RS ratings by ascending order
# rs_ratings.sort(key=lambda x: x[1])

# # Get the top 30% of RS ratings
# num_top_ratings = int(len(rs_ratings) * 0.3)
# top_ratings = rs_ratings[-num_top_ratings:]

# # Write the top ratings to a CSV file
# date_string = datetime.datetime.now().strftime("%Y-%m-%d")
# filename = f"top_rs_ratings_{date_string}.csv"
# with open(filename, "w") as f:
#     f.write("Symbol,RS Rating\n")
#     for rating in top_ratings:
#         f.write(f"{rating[0]},{rating[1]}\n")
