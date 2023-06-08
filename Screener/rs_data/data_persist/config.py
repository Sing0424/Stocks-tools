import os
import datetime

date_string = datetime.datetime.now().strftime("%Y-%m-%d")

stocks_csv_path = os.path.join("Screener", "rs_data", "data_persist", "symbols.csv")

daily_rs_rating_filename = f"rs_rating_Top_30_{date_string}.csv"
daily_rs_rating_Top_30_path = os.path.join("Screener", "rs_data", "data_persist", daily_rs_rating_filename)

alphavantage_api_key = '7D80AAZF1EFC0TZJ'
symbols_url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={alphavantage_api_key}'