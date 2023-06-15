import os
import datetime

#Date/time config
date_string = datetime.datetime.now().strftime("%Y-%m-%d")

#Path config
stocks_csv_path = os.path.join("Screener", "rs_data", "data_persist", "symbols.csv")
daily_rs_rating_filename = f"rs_rating_Top_30_{date_string}.csv"
daily_rs_rating_Top_30_path = os.path.join("Screener", "rs_data", "data_persist", daily_rs_rating_filename)

#Data config
alphavantage_api_key = '7D80AAZF1EFC0TZJ'
symbols_url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={alphavantage_api_key}'
with open(stocks_csv_path, "r") as f:
    symbols = [line.strip() for line in f]

#Program config
trading_days_per_quarter = 63 #define days in a quarter
top_rating = 0.3 #percentage of top ratings, range: 0.0 ~ 1.0
chunksize = 64 #for optimizing the multiprocessing, higher = faster 