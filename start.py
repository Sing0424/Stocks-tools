import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dtpyt
from pandas_datareader import data as pdr

yf.pdr_override()

stock = input("Enter stock symbol:")
print(stock)

startyear = 2022
startmonth = 12
startday = 1

start = dt.datetime(startyear, startmonth, startday)
now = dt.datetime.now()

df=pdr.get_data_yahoo(stock, start, now)

ma = 50

smaString = "Sma_" + str(ma)
df[smaString] = df.iloc[:,4].rolling(window=ma).mean()

df = df.iloc[ma:]

for i in df.index:
    print(df[smaString][i])