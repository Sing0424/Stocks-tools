from stocksymbol import StockSymbol
from config import stocks_csv_path, stock_symbol_api_key
import csv

symbols = StockSymbol(stock_symbol_api_key)

# get symbol list based on market
symbol_list_US = symbols.get_symbol_list(market="US",symbols_only=True) # "us" or "america" will also work
symbol_list_US.sort(key=lambda x: x[0])

with open(stocks_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for symbol in symbol_list_US:
        if '.' not in symbol:
            writer.writerow([symbol])

print(f'Symbols written to {stocks_csv_path}')