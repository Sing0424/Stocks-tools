import pandas as pd
import requests
from config import symbols_url, stocks_csv_path

with requests.Session() as s:
    download = s.get(symbols_url)
    decoded_content = download.content.decode('utf-8')
    cr = pd.read_csv(pd.compat.StringIO(decoded_content))
    symbols = cr['symbol'].tolist()

    # Write the symbols to a CSV file
    cr['symbol'].to_csv(stocks_csv_path, index=False, header=True)
    print('Symbols written to', stocks_csv_path)