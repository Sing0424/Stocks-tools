from io import StringIO
from curl_cffi import requests as cffi_requests
from contextlib import redirect_stderr
import yfinance as yf

def download_single_stock(symbol):
    session = cffi_requests.Session(impersonate="chrome")
    with redirect_stderr(StringIO()):
        info = yf.Ticker(symbol, session=session).info
        industry = info.get('industry')
        print(industry)
        sector = info.get('sector')
        print(sector)

download_single_stock("NVDA")