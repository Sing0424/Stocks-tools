import csv
import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
CSV_URL = 'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo'

with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    
    # Extract the symbol column from the list
    symbols = [row[1] for row in my_list[1:]]
    
    # Write the symbols to a CSV file
    with open('symbols.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for symbol in symbols:
            writer.writerow([symbol])
        
    print('Symbols written to symbols.csv')