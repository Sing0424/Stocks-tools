import os

#Path config for command promt
stocks_csv_path = 'data/symbols.csv'
daily_rs_rating_path = 'data/rs_rating.xlsx'
screen_result_path = 'ScreenResult/ScreenResult.xlsx'

#Data config
#Get symbol list from https://stock-symbol.herokuapp.com/
stock_symbol_api_key = '2b96bf82-acb9-40be-8cc1-73659e7fbafb'

#Program config
days_per_month = 62 #define days in a quarter
rs_month_weight = 0.2 #weight for calculate_rs_rating
top_rating = 0.2  #percentage of top ratings, range: 0.0 ~ 1.0