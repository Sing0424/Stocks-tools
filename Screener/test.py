import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import datetime
import pandas as pd
from config import screen_result_path

import_data = pd.read_excel(screen_result_path , usecols=['Symbol'])

# List of stock symbols
symbols = import_data['Symbol']

for symbol in symbols:
    
    # Get stock data 
    data = yf.Ticker(symbol)
    df = data.history(start = datetime.datetime.now() - datetime.timedelta(weeks= 104), end = datetime.datetime.now(),interval='1wk')

    df = df.resample('W').agg({'Open': 'first', 'High': 'max',
                           'Low': 'min','Close': 'last','Volume': 'sum'})

    #針對線圖的外觀微調，將上漲設定為紅色，下跌設定為綠色，符合台股表示習慣
    #接著把自訂的marketcolors放到自訂的style中，而這個改動是基於預設的yahoo外觀

    kwargs = dict(type='candle', mav=(10,20,30,50,150,200), volume=True, figratio=(16,9), figscale=0.8, title=symbol, style='yahoo') 
    #設定可變參數kwargs，並在變數中填上繪圖時會用到的設定值

    mpf.plot(df, **kwargs)
    #選擇df資料表為資料來源，帶入kwargs參數，畫出目標股票的走勢圖