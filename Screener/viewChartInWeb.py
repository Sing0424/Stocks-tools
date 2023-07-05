import webbrowser
import pandas as pd
from config import screen_result_path

import_data = pd.read_excel(screen_result_path)['Symbol']
num_websites_per_batch = 10
start_from = 150

for i in range(start_from, len(import_data), num_websites_per_batch):
    if i < num_websites_per_batch:
        input(f"Press enter to open {i} to {i+num_websites_per_batch} of {len(import_data)} websites...")
    else:
        input(f"Press enter to open {i} to {len(import_data)} of {len(import_data)} websites...")
    symbols_to_open = import_data[i:i+num_websites_per_batch]
    for symbol in symbols_to_open:
        webbrowser.open(f'https://tw.tradingview.com/chart/r88XNs7k/?symbol={symbol}')