import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from config import stocks_csv_path

# Read the stock symbols from the CSV file
stocks_df = pd.read_csv(stocks_csv_path)
symbols = stocks_df["symbol"].tolist()

# Calculate the date range for each quarter
today = datetime.date.today()
one_quarter_ago = today - datetime.timedelta(days=63)
two_quarters_ago = today - datetime.timedelta(days=2 * 63)
three_quarters_ago = today - datetime.timedelta(days=3 * 63)
four_quarters_ago = today - datetime.timedelta(days=4 * 63)

# Define a function to calculate the price percentage change for a given stock and time range
def price_change_percentage(stock, start, end):
    stock_data = yf.download(stock, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    initial_price = stock_data["Close"].iloc[0]
    final_price = stock_data["Close"].iloc[-1]
    return ((final_price - initial_price) / initial_price) * 100

# Calculate the RS ratings for each stock
rs_ratings = []
for symbol in symbols:
    try:
        pc_1 = price_change_percentage(symbol, one_quarter_ago, today)
        pc_2 = price_change_percentage(symbol, two_quarters_ago, one_quarter_ago)
        pc_3 = price_change_percentage(symbol, three_quarters_ago, two_quarters_ago)
        pc_4 = price_change_percentage(symbol, four_quarters_ago, three_quarters_ago)

        rs_rating = (0.4 * pc_1) + (0.2 * pc_2) + (0.2 * pc_3) + (0.2 * pc_4)
        rs_ratings.append((symbol, rs_rating))
    except Exception as e:
        print(f"Error calculating RS rating for {symbol}: {str(e)}")

# Sort the list by RS rating in descending order and get the top 30%
rs_ratings.sort(key=lambda x: x[1], reverse=True)
top_30_percent = rs_ratings[:int(len(rs_ratings) * 0.3)]

# Save the top 30% of stocks with the highest RS rating into a new CSV file
top_rs_df = pd.DataFrame(top_30_percent, columns=["symbol", "RS_rating"])
top_rs_df.to_csv("top_30_percent_RS_rating.csv", index=False)