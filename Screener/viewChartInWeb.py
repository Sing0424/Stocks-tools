import webbrowser
import pandas as pd
from config import screen_result_path

import_data = pd.read_excel(screen_result_path , usecols=['Symbol'])
normal_filter = import_data['Symbol']
num_websites_per_batch = 10
start_from = 0

def run_w_normal_filter():
    for i in range(start_from, len(normal_filter), num_websites_per_batch):
        if i < num_websites_per_batch:
            input(f"Press enter to open {i} to {i+num_websites_per_batch} of {len(normal_filter)} websites...")
        else:
            input(f"Press enter to open {i} to {len(normal_filter)} of {len(normal_filter)} websites...")
        symbols_to_open = normal_filter[i:i+num_websites_per_batch]
        for symbol in symbols_to_open:
            webbrowser.open(f'https://tw.tradingview.com/chart/r88XNs7k/?symbol={symbol}')

run_w_normal_filter()
