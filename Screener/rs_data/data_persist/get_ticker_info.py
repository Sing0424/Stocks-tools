import csv
import requests
from config import symbols_url, stocks_csv_path

with requests.Session() as s:
    download = s.get(symbols_url)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    
    # Extract the symbol column from the list
    symbols = [row[0] for row in my_list[1:]]
    
    # Write the symbols to a CSV file
    with open(stocks_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for symbol in symbols:
            writer.writerow([symbol])
        
    print(f'Symbols written to {stocks_csv_path}')