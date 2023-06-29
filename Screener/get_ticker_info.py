import csv
import requests
from config import symbols_url, stocks_csv_path

def format_symbol(symbol):
    # Count the number of '-' characters in the symbol
    num_dashes = symbol.count('-')
    # If there's only one '-', leave the symbol as is
    if num_dashes <= 1:
        return symbol
    # Otherwise, remove the last '-' character
    else:
        last_dash_index = symbol.rfind('-')
        formatted_symbol = symbol[:last_dash_index] + symbol[last_dash_index+1:]
        return formatted_symbol

with requests.Session() as s:
    download = s.get(symbols_url)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    symbol_list = list(cr)
    
    # Extract the symbol column from the list and format the symbols
    symbols = []
    for row in symbol_list[1:]:
        symbol = row[0]
        formatted_symbol = format_symbol(symbol)
        symbols.append(formatted_symbol.strip())
    
# Write the formatted symbols to a CSV file
with open(stocks_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for symbol in symbols:
        writer.writerow([symbol])
        
print(f'Symbols written to {stocks_csv_path}')