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
    return {"info": {"industry": industry, "sector": sector}}


def update_data(data_json):
    with open(
        r"C:\Ivan\Repo\Stocks-tools\Screener\rs_data\data_persist\test_tickers_info.json",
        "r",
    ) as exist_json:
        exist_data = json.load(exist_json)

    symbols = {key.strip() for key in data_json}
    for symbol in symbols:
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

    with open(
        r"C:\Ivan\Repo\Stocks-tools\Screener\rs_data\data_persist\test_tickers_info.json",
        "w",
    ) as outfile:
        json.dump(exist_data, outfile, indent=2)


latest_data = get_latest_data()
update_data(latest_data)
