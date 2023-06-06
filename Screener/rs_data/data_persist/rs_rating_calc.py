import pandas as pd
import yfinance as yf
from config import stocks_csv_path

def get_rs_rating(stock):
    # Download historical data for the stock and the S&P 500 index
    data = yf.download(stock, start="2020-01-01", end="2023-06-06")
    sp500_data = yf.download("^GSPC", start="2020-01-01", end="2023-06-06")

    # Calculate the percentage change in price over the past 6 months
    pct_change_6m = data['Adj Close'].pct_change(periods=126)
    sp500_pct_change_6m = sp500_data['Adj Close'].pct_change(periods=126)

    # Calculate the RS rating as the ratio of the stock's average percentage change to the S&P 500's average percentage change
    rs_rating = (pct_change_6m.mean() / sp500_pct_change_6m.mean()) * 100
    return rs_rating

# Load the stock symbols into a Pandas DataFrame
stocks_df = pd.read_csv(stocks_csv_path)

# Calculate the RS rating for each stock and add it to the DataFrame
rs_ratings = []
for symbol in stocks_df['symbol']:
    try:
        rs_rating = get_rs_rating(symbol)
        rs_ratings.append(rs_rating)
    except:
        rs_ratings.append(None)
stocks_df['RS Rating'] = rs_ratings

# Scale the RS rating between 0 and 100
min_rating = stocks_df['RS Rating'].min()
max_rating = stocks_df['RS Rating'].max()
stocks_df['Scaled RS Rating'] = ((stocks_df['RS Rating'] - min_rating) / (max_rating - min_rating)) * 100
print(stocks_df['Scaled RS Rating'])

# Filter the DataFrame to include only stocks with a scaled RS rating greater than 70
filtered_stocks_df = stocks_df[stocks_df['Scaled RS Rating'] > 70]

# Write the filtered DataFrame to a new CSV file
filtered_stocks_df.to_csv(f'RS_Ratings_{pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)