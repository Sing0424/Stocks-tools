# import urllib library
from urllib.request import urlopen
from yahooquery import Ticker
import json

# store the URL in url as 
# parameter for urlopen
url = "https://dumbstockapi.com/stock?format=tickers-only&countries=US"
  
# store the response of URL
response = urlopen(url)
  
# storing the JSON response 
# from url in data
data_json = json.loads(response.read())
num_of_tickers = 1#len(data_json) - 1
formatted_json = {}

with open('Screener/rs_data/data_persist/test_tickers_info.json', 'r') as exist_json:
    json_str = exist_json.read()
    exist_data = json.loads(json_str)

    i = 0 
    while i < num_of_tickers: 

        symbol = "A"#data_json[i].strip()
        print(symbol)
        ticker = Ticker(symbol)

        if symbol in exist_data:
            print("exist ticker")
            try:
                industry = ticker.summary_profile[symbol]["industry"]
                sector = ticker.summary_profile[symbol]["sector"]

                exist_data[symbol] = {
                                        "info":{
                                                "industry":str(industry),
                                                "sector":str(sector)
                                                }
                                    }
                i += 1
            except:
                pass
        elif symbol not in exist_data:
            print("not exist ticker")
            print(exist_data)
            industry = ticker.summary_profile[symbol]["industry"]
            sector = ticker.summary_profile[symbol]["sector"]

            append_data = {"AA": {
                                    "info": {
                                    "industry": "Diagnostics & Research",
                                    "sector": "Healthcare"
                                    }
                                }}
                            
                
            exist_data.append(append_data)
            i += 1

with open("Screener/rs_data/data_persist/test_tickers_info.json", "w") as outfile:
    outfile.write(json.dumps(exist_data,indent=2))