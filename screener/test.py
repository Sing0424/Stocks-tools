from io import StringIO
from curl_cffi import requests as cffi_requests
from contextlib import redirect_stderr
import yfinance as yf
import json

def download_single_stock(symbol):
    with redirect_stderr(StringIO()):
        info = yf.Ticker(symbol).quarterly_financials.loc['Diluted EPS'].iloc[1]
        print(info)

download_single_stock("WDC")