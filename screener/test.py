from io import StringIO
from curl_cffi import requests as cffi_requests
from contextlib import redirect_stderr
import yfinance as yf
import json

def download_single_stock(symbol):
    with redirect_stderr(StringIO()):
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        industry = str(info.get('industry', 'N/A'))
        print(f"Symbol: {symbol}")
        print(f"Industry: {industry}")

download_single_stock("MU")