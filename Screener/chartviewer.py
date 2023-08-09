import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import datetime
import pandas as pd
from config import screen_result_path

import_data = pd.read_excel(screen_result_path , usecols=['Symbol', 'Code 33'])
code33_filter = import_data[(import_data['Code 33'] == 'T')]['Symbol']

# List of stock symbols
print(f"Filter with code 33? (Y/N)")
x = input()

if x.upper() == "Y":
    symbols = import_data[(import_data['Code 33'] == 'T')]['Symbol']
elif x.upper() == "N":
    symbols = import_data['Symbol']
else:
    print("Input Error")

def dailyChart():
    for symbol in symbols:
        # Get stock data 
        data = yf.Ticker(symbol)
        df = data.history(start = datetime.datetime.now() - datetime.timedelta(days = 504), end = datetime.datetime.now())

        kwargs = dict(type='candle', mav=(10,20,30,50,150,200), volume=True, figratio=(16,9), figscale=1, title=symbol, style='yahoo') 

        mpf.plot(df, **kwargs)

def weeklyChart():
    for symbol in symbols:
        
        # Get stock data 
        data = yf.Ticker(symbol)
        df = data.history(start = datetime.datetime.now() - datetime.timedelta(weeks= 104), end = datetime.datetime.now(),interval='1wk')

        df = df.resample('W').agg({'Open': 'first', 'High': 'max',
                            'Low': 'min','Close': 'last','Volume': 'sum'})

        kwargs = dict(type='candle', mav=(10,20,30,50,150,200), volume=True, figratio=(16,9), figscale=1, title=symbol, style='yahoo') 

        mpf.plot(df, **kwargs)

print(f"View as daily or weekly? (D/W)")
x = input()

if x.upper() == "D":
    dailyChart()
elif x.upper() == "W":
    weeklyChart()
else:
    print("Input Error")