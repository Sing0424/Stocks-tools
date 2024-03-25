import ftplib
import pandas as pd
from config import stocks_csv_path, stock_symbol_api_key, symbol_list_nq_path, symbol_list_other_path
import csv

# symbols = StockSymbol(stock_symbol_api_key)

# # get symbol list based on market
# symbol_list_US = symbols.get_symbol_list(market="US",symbols_only=True) # "us" or "america" will also work
# symbol_list_US.sort(key=lambda x: x[0])

# with open(stocks_csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for symbol in symbol_list_US:
#         if '.' not in symbol:
#             writer.writerow([symbol])

# print(f'Symbols written to {stocks_csv_path}')

ftp_nasdaq = ftplib.FTP('ftp.nasdaqtrader.com')
ftp_nasdaq.login()
ftp_nasdaq.encoding = "utf-8"

all_symbol = []
ftp_nasdaq.cwd('Symboldirectory')

with open(symbol_list_nq_path, "wb") as file:
    ftp_nasdaq.retrbinary(f"RETR nasdaqlisted.txt", file.write)
    data_nasdaq = pd.read_csv(symbol_list_nq_path, sep="|")
    data_nasdaq_filtered = data_nasdaq[(data_nasdaq['Test Issue'] == 'N') & (data_nasdaq['Financial Status'] == 'N') & (data_nasdaq["Symbol"].str.contains(f"[.+=$^-]") != True) & (data_nasdaq['ETF'] == 'N')]
    # print(data_nasdaq_filtered[(data_nasdaq_filtered['Symbol'].str.contains("nan") == True)])

with open(symbol_list_other_path, "wb") as file:
    ftp_nasdaq.retrbinary(f"RETR otherlisted.txt", file.write)
    data_nasdaq_other = pd.read_csv(symbol_list_other_path, sep="|")
    data_nasdaq_other_filtered = data_nasdaq_other[(data_nasdaq_other['Test Issue'] == 'N') & (data_nasdaq_other["NASDAQ Symbol"].str.contains(f"[.+=$^-]") != True) & (data_nasdaq_other['ETF'] == 'N')]
    # print(data_nasdaq_other_filtered['NASDAQ Symbol'].str.contains("nan"))

ftp_nasdaq.quit()

all_symbol_df = pd.concat([data_nasdaq_filtered['Symbol'], data_nasdaq_other_filtered['NASDAQ Symbol']])
symbol_list = list(set(all_symbol_df.tolist()))
symbol_list.sort(key=lambda x: str(x))

with open(stocks_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for symbol in symbol_list:
        writer.writerow([symbol])

print(f'Symbols written to {stocks_csv_path}')