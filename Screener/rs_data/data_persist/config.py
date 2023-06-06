import os

stocks_csv_path = os.path.join("Screener", "rs_data", "data_persist", "symbols.csv")

apikey = 'demo'
symbols_url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={apikey}'