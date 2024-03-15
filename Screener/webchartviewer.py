import webbrowser
import pandas as pd
from config import screen_result_path

# import_data = pd.read_excel(screen_result_path , usecols=['Symbol', 'Code 33'])
import_data = pd.read_excel(screen_result_path , usecols=['Symbol'])
normal_filter = import_data['Symbol']
# code33_filter = import_data[(import_data['Code 33'] == 'T')]['Symbol']

def run_w_normal_filter():
    str_symbol = ','.join(normal_filter.astype(str))
    print(str_symbol)
    webbrowser.open(f'https://finviz.com/screener.ashx?v=211&p=w&t={str_symbol}')

# def run_w_code33_filter():
#     for i in range(start_from, len(code33_filter), num_websites_per_batch):
#         if i < num_websites_per_batch:
#             input(f"Press enter to open {i} to {i+num_websites_per_batch} of {len(code33_filter)} websites...")
#         else:
#             input(f"Press enter to open {i} to {len(code33_filter)} of {len(code33_filter)} websites...")
#         symbols_to_open = code33_filter[i:i+num_websites_per_batch]
#         for symbol in symbols_to_open:
#             webbrowser.open(f'https://finviz.com/quote.ashx?t={symbol}&ty=c&ta=1&p=w')

# print(f"Filter with code 33? (Y/N)")
# x = input()

# if x.upper() == "Y":
#     run_w_code33_filter()
# elif x.upper() == "N":
#     run_w_normal_filter()
# else:
#     print("Input Error")
run_w_normal_filter()