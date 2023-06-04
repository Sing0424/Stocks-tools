# import library
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from yahooquery import Ticker
import json
import hashlib
import logging
from multiprocessing import Pool
from functools import partial

def fetch_data(ticker, symbol):
    try:
        industry = ticker.summary_profile.get(symbol,{}).get("industry")
        sector = ticker.summary_profile.get(symbol,{}).get("sector")
    except Exception as e:
        print(f"{symbol}: data fetch failed: {e}")
        return None
    return {"info": {"industry":industry, "sector":sector}}

def update_data_for_symbol(symbol, fetched_data, exist_data):
    if fetched_data is None:
        return
    if symbol in exist_data and exist_data[symbol] == fetched_data:
        logging.info(f"{symbol}: data was up to date")
    else:
        exist_data[symbol] = fetched_data
        logging.info(f"{symbol}: data updated successfully")

def update_data(latest_data):
    try:
      with open('/test_tickers_info.json', 'r') as exist_json:
          exist_data = json.load(exist_json)
    except (FileNotFoundError, json.JSONDecodeError):
        exist_data = {}
    
    if not isinstance(exist_data, dict):
        exist_data = {}

    symbols = set(key.strip() for key in latest_data)

    retries = Retry(total=5, backoff_factor=0.1, status_forcelist = [ 500, 502, 503, 504 ])
    session = requests.Session()
    adapter = HTTPAdapter(max_retries= retries)
    adapter.init_poolmanager(connections = 10, maxsize = 10, max_retries = retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.timeout = (30, 30)

    pool = Pool()
    fetch_data_func = partial(fetch_data, Ticker(symbols=symbols, session = session))
    fetch_data_list = pool.map(fetch_data_func, symbols)
    pool.close()
    pool.join()

    fetch_data_dict = {}

    for symbol, fetched_data in zip(symbols, fetch_data_list):
        fetch_data_dict[symbol] = fetched_data
    
    for symbol, fetched_data in fetch_data_dict.items():
        update_data_for_symbol(symbol, fetched_data, exist_data)

    with open("/test_tickers_info.json", "w") as outfile:
        json.dump(exist_data, outfile, indent=2)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    latest_data = requests.get("https://dumbstockapi.com/stock?format=tickers-only&countries=US").json()
    update_data(latest_data)