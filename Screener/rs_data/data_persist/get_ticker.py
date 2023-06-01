# import library
from urllib.request import urlopen
from yahooquery import Ticker
import json
import time

start_time = time.time()

# store the URL in url as 
# parameter for urlopen
url = "https://dumbstockapi.com/stock?format=tickers-only&countries=US"
  
# store the response of URL
response = urlopen(url)
  
# storing the JSON response 
# from url in data
data_json = json.loads(response.read())

with open('test_tickers_info.json', 'r') as exist_json:
    json_str = exist_json.read()
    exist_data = json.loads(json_str)

    for key in data_json:
    
        symbol = key.strip()
        ticker = Ticker(symbol)
    
        if symbol in exist_data:
            try:
                industry = ticker.summary_profile[symbol]["industry"]
                sector = ticker.summary_profile[symbol]["sector"]
    
                exist_data[symbol] = {
                                        "info":{
                                                "industry":str(industry),
                                                "sector":str(sector)
                                                }
                                    }
                print(symbol + ": data fetch sucessfully by update")
            except:
                print(symbol + ": data fetch failed")
    
        elif symbol not in exist_data:
            try:
                industry = ticker.summary_profile[symbol]["industry"]
                sector = ticker.summary_profile[symbol]["sector"]
    
                append_data = {
                                "info":{
                                        "industry":str(industry),
                                        "sector":str(sector)
                                        }
                                }
                                            
                exist_data[symbol] = append_data
                print(symbol + ": data fetch sucessfully by add")
            except:
                print(symbol + ": data fetch failed")

with open("test_tickers_info.json", "w") as outfile:
    outfile.write(json.dumps(exist_data,indent=2))

with open('test_tickers_info.json', 'r') as exist_json:
    json_str = exist_json.read()
    exist_data = json.loads(json_str)
    non_exist_list = []

    for key in exist_data:
        if key not in data_json:
            non_exist_list.append(key)

    for key in non_exist_list:
        del exist_data[key]
        print(key + ": not existing, has been deleted")

with open("test_tickers_info.json", "w") as outfile:
    outfile.write(json.dumps(exist_data,indent=2))

end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 