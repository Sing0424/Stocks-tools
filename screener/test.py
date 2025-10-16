from io import StringIO
from curl_cffi import requests as cffi_requests
from contextlib import redirect_stderr
import yfinance as yf

def download_single_stock(symbol):
    session = cffi_requests.Session(impersonate="chrome110")
    try:
        with redirect_stderr(StringIO()):
            # info = yf.Ticker(symbol, session=session).info
            # industry = info.get('industry', 'N/A')
            # sector = info.get('sector', 'N/A')
            data = yf.download(
                tickers=symbol,
                period='1mo',
                group_by='ticker',
                repair=True,
                session=session)
            print(data)
    except Exception as e:
        print(e)
    return data

download_single_stock(["NVDA","AAPL"])