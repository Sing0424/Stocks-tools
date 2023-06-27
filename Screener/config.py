import os

#Path config for command promt
stocks_csv_path = 'data/symbols.csv'
daily_rs_rating_Top_30_path = 'data/rs_rating_top_30.xlsx'
screen_result_path = 'ScreenResult/ScreenResult.xlsx'

#Path config for vscode terminal
# stocks_csv_path = os.path.join("Screener", "data", "symbols.csv")
# daily_rs_rating_Top_30_path = os.path.join("Screener", "data", "rs_rating_top_30.xlsx")
# screen_result_path = os.path.join("Screener", "ScreenResult", "ScreenResult.xlsx")

#Data config
#Get symbol list from https://www.alphavantage.co/
stock_symbol_api_key = '7D80AAZF1EFC0TZJ'
symbols_url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={stock_symbol_api_key}'

#Program config
trading_days_per_quarter = 63 #define days in a quarter
top_rating = 0.3 #percentage of top ratings, range: 0.0 ~ 1.0
chunksize = 64 #optimizing the multiprocessing