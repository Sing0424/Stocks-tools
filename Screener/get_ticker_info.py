from stocksymbol import StockSymbol
from config import stocks_csv_path, stock_symbol_api_key
import csv

symbols = StockSymbol(stock_symbol_api_key)

# get symbol list based on market
symbol_list_US = symbols.get_symbol_list(market="US") # "us" or "america" will also work
symbol_list_US.sort(key=lambda x: x['symbol'])

with open(stocks_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['symbol'], extrasaction='ignore')
        writer.writerows(symbol_list_US)

print(f'Symbols written to {stocks_csv_path}')

# get symbol list based on index
# symbol_list_spx = ss.get_symbol_list(index="SPX")

# # show a list of available market
# market_list = ss.market_list

# # show a list of available index
# index_list = ss.index_list