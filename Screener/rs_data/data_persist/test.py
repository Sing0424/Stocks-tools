import os
import re
import requests
from yahooquery import Ticker
import json


def get_latest_data():
    url = "https://dumbstockapi.com/stock?format=tickers-only&countries=US"
    with requests.Session() as session:
        response = session.get(url)
        data_json = response.json()
    return data_json


def fetch_data(symbol):
    ticker = Ticker(symbol)
    try:
        profile = ticker.get_modules("assetProfile")[symbol]
        industry = profile.get("industry")
        sector = profile.get("sector")
    except Exception as e:
        print(f"{symbol}: data fetch failed, {e}")
        return None
    return {"info": {"industry": re.sub("[^a-zA-Z]", ' ', str(industry)), "sector": re.sub(r"[^a-zA-Z]", ' ', str(sector))}}


def update_data(data_json):
    json_dir_path = os.path.join("Screener", "rs_data", "data_persist", "test_tickers_info.json")
    with open(json_dir_path, "r") as exist_json:
        exist_data = json.load(exist_json)

    symbols = {key.strip() for key in data_json}
    for symbol in symbols:
        symbol = symbol.split('^', 1)[0]
        fetched_data = fetch_data(symbol)

        if fetched_data is None:
            continue

        if symbol in exist_data:
            if exist_data[symbol] != fetched_data:
                exist_data[symbol] = fetched_data
                print(f"{symbol}: data updated successfully")
            else:
                print(f"{symbol}: data was up to date")
        else:
            exist_data[symbol] = fetched_data
            print(f"{symbol}: data add successfully")

    with open(json_dir_path, "w") as outfile:
        outfile.write(json.dumps(exist_data, indent=2))


latest_data = get_latest_data()
update_data(latest_data)
