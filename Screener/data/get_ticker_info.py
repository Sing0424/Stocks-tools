import csv
import requests
from config import symbols_url, stocks_csv_path

with requests.Session() as s:
    download = s.get(symbols_url)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    
    # Extract the symbol column from the list and format the symbols
    symbols = []
    for row in my_list[1:]:
        symbol = row[0]
        if symbol.count('-') >= 2:
            symbol = symbol.rsplit('-', 1)[0]
        symbols.append(symbol.strip())
    
    # Write the formatted symbols to a CSV file
    with open(stocks_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for symbol in symbols:
            writer.writerow([symbol])
        
    print(f'Symbols written to {stocks_csv_path}')