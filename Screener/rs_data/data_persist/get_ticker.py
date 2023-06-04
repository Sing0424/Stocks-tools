# import library
from urllib.request import urlopen
from yahooquery import Ticker
import json

def get_latest_data():
    url = "https://dumbstockapi.com/stock?format=tickers-only&countries=US"
    response = urlopen(url)
    data_json = json.loads(response.read())
    return data_json

def update_data(data_json):
    with open('/test_tickers_info.json', 'r') as exist_json:
        json_str = exist_json.read()
        exist_data = json.loads(json_str)

        for key in data_json:
            symbol = key.strip() 
            ticker = Ticker(symbol)

            try:
                industry = ticker.summary_profile[symbol]["industry"]
                sector = ticker.summary_profile[symbol]["sector"]
            except:
                print(symbol + ": data fetch failed")
    
            if symbol in exist_data:
                try:
                    exist_industry = exist_data[symbol]["info"]["industry"]
                    exist_sector = exist_data[symbol]["info"]["sector"]

                    if industry != exist_industry or sector != exist_sector:
                        exist_data[symbol] = {
                                                "info":{
                                                        "industry":industry,
                                                        "sector":sector
                                                        }
                                            }
                        print(symbol + ": data fetch sucessful by update")
                    else:
                        print(symbol + ": data up to date")
                except:
                    print(symbol + ": No symbol in exist_data")
    
            elif symbol not in exist_data:
                try:
                    append_data = {
                            "info":{
                                    "industry":industry,
                                    "sector":sector
                                    }
                            }
                    exist_data[symbol] = append_data
                    print(symbol + ": data fetch sucessfully by add")
                except:
                    print(symbol + ": data add failed")

    with open("/test_tickers_info.json", "w") as outfile:
        outfile.write(json.dumps(exist_data,indent=2))

    with open('/test_tickers_info.json', 'r') as exist_json:
        json_str = exist_json.read()
        exist_data = json.loads(json_str)
        non_exist_list = []

        for key in exist_data:
            if key not in data_json:
                non_exist_list.append(key)

        for key in non_exist_list:
            del exist_data[key]
            print(key + ": not existing, has been deleted")

    with open("/test_tickers_info.json", "w") as outfile:
        outfile.write(json.dumps(exist_data,indent=2))

latest_data = get_latest_data()
update_data(latest_data)