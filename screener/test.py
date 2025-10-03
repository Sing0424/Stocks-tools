from io import StringIO
from curl_cffi import requests as cffi_requests
from contextlib import redirect_stderr
import yfinance as yf

def download_single_stock(symbol):
    session = cffi_requests.Session(impersonate="chrome110")
    try:
        with redirect_stderr(StringIO()):
            info = yf.Ticker(symbol, session=session).info
            industry = info.get('industry', 'N/A')
            sector = info.get('sector', 'N/A')
    except Exception as e:
        print(e)
        industry = 'N/A'
        sector = 'N/A'
    return {
        'symbol': symbol,
        'industry': industry,
        'sector': sector,
    }

print(download_single_stock("NVDA"))