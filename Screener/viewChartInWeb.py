import webbrowser
import pandas as pd
from config import screen_result_path

import_data = pd.read_excel(screen_result_path , usecols=['Symbol', 'Code 33'])
code33_filter = import_data[(import_data['Code 33'] == 'T')]['Symbol']
num_websites_per_batch = 10
start_from = 0

def run_w_normal_filter():
    for i in range(start_from, len(import_data), num_websites_per_batch):
        if i < num_websites_per_batch:
            input(f"Press enter to open {i} to {i+num_websites_per_batch} of {len(import_data)} websites...")
        else:
            input(f"Press enter to open {i} to {len(import_data)} of {len(import_data)} websites...")
        symbols_to_open = import_data[i:i+num_websites_per_batch]
        for symbol in symbols_to_open:
            webbrowser.open(f'https://tw.tradingview.com/chart/r88XNs7k/?symbol={symbol}')

def run_w_code33_filter():
    for i in range(start_from, len(code33_filter), num_websites_per_batch):
        if i < num_websites_per_batch:
            input(f"Press enter to open {i} to {i+num_websites_per_batch} of {len(code33_filter)} websites...")
        else:
            input(f"Press enter to open {i} to {len(code33_filter)} of {len(code33_filter)} websites...")
        symbols_to_open = code33_filter[i:i+num_websites_per_batch]
        for symbol in symbols_to_open:
            webbrowser.open(f'https://tw.tradingview.com/chart/r88XNs7k/?symbol={symbol}')

print(f"Filter with code 33? (Y/N)")
x = input()

if x.upper() == "Y":
    run_w_code33_filter()
elif x.upper() == "N":
    run_w_normal_filter()
else:
    print("Input Error")
