import os
import datetime

#Path config
stocks_csv_path = 'csv/symbols.csv'
daily_rs_rating_Top_30_path = 'csv/rs_rating_Top_30.csv'
# stocks_csv_path = os.path.join("Screener", "rs_data", "data_persist", "symbols.csv")
# daily_rs_rating_Top_30_path = os.path.join("Screener", "rs_data", "data_persist", "rs_rating_Top_30.csv")

#Data config
alphavantage_api_key = '7D80AAZF1EFC0TZJ'
symbols_url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={alphavantage_api_key}'
with open(stocks_csv_path, "r") as f:
    symbols = [line.strip() for line in f]

#Program config
trading_days_per_quarter = 63 #define days in a quarter
top_rating = 0.3 #percentage of top ratings, range: 0.0 ~ 1.0
chunksize = 64 #optimizing the multiprocessing