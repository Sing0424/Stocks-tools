import pandas as pd
import numpy as np
import yfinance as yf
from config import stocks_csv_path

# Load the stock symbols into a pandas dataframe
stocks_df = pd.read_csv(stocks_csv_path)

# Define the start and end dates for the RS rating calculation
start_date = pd.Timestamp.today() - pd.DateOffset(years=1)
end_date = pd.Timestamp.today()

# Calculate the RS rating for each stock symbol
ratings = []
for symbol in stocks_df["symbol"]:
    try:
        # Get the historical stock prices from Yahoo Finance
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        # Calculate the percentage change in price for the past year
        price_changes = stock_data["Adj Close"].pct_change()

        # Calculate the percentage change in price for the S&P 500 index for the past year
        sp500_data = yf.download("^GSPC", start=start_date, end=end_date)
        sp500_changes = sp500_data["Adj Close"].pct_change()

        # Calculate the RS rating as the ratio of the stock's price change to the S&P 500's price change
        rs_rating = (price_changes / sp500_changes).mean() * 100

        # Append the symbol and RS rating to the list of ratings
        ratings.append((symbol, rs_rating))

    except:
        # If there is an error getting the stock data, skip this symbol
        continue

# Convert the list of ratings to a pandas dataframe
ratings_df = pd.DataFrame(ratings, columns=["symbol", "RS Rating"])

# Filter the results to only include symbols with an RS rating greater than 70
filtered_ratings_df = ratings_df[ratings_df["RS Rating"] > 70]

# Save the results to a CSV file
filtered_ratings_df.to_csv("rs_ratings.csv", index=False)