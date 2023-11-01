import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_Top_30_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time

Screen_result_list = []
import_data = pd.read_excel(daily_rs_rating_Top_30_path)

symbols = import_data["Symbol"]
#symbols = ['GD']

# stock_data = yf.download(tickers = symbol, period='max', progress=False)
def test_block(symbol):
    for symbol in symbols:
        print(symbol)
        ticker_data = yf.Ticker(symbol)
        #sector = info.get('sector')
        #industry = info.get('industry')

        inc_stat = ticker_data.quarterly_income_stmt
        try:
            eps_list = inc_stat.loc['Diluted EPS'].dropna()
            #print(eps_list)
            lenth_eps_list = len(eps_list)
            if lenth_eps_list >= 4:
                first_qtr_eps = eps_list.iloc[3]
                second_qtr_eps = eps_list.iloc[2]
                third_qtr_eps = eps_list.iloc[1]
                current_qtr_eps = eps_list.iloc[0]
            elif lenth_eps_list >= 3:
                first_qtr_eps = 0
                second_qtr_eps = eps_list.iloc[2]
                third_qtr_eps = eps_list.iloc[1]
                current_qtr_eps = eps_list.iloc[0]
            elif lenth_eps_list >= 2:
                first_qtr_eps = 0
                second_qtr_eps = 0
                third_qtr_eps = eps_list.iloc[1]
                current_qtr_eps = eps_list.iloc[0]
            elif lenth_eps_list >= 1:
                first_qtr_eps = 0
                second_qtr_eps = 0
                third_qtr_eps = 0
                current_qtr_eps = eps_list.iloc[0]
        except:
            first_qtr_eps = 0
            second_qtr_eps = 0
            third_qtr_eps = 0
            current_qtr_eps = 0

        try:
            inc_list = inc_stat.loc['Net Income'].dropna()
            #print(inc_list)
            lenth_inc_list = len(inc_list)
            if lenth_inc_list >= 4:
                first_qtr_inc = inc_list.iloc[3]
                second_qtr_inc = inc_list.iloc[2]
                third_qtr_inc = inc_list.iloc[1]
                current_qtr_inc = inc_list.iloc[0]
            elif lenth_inc_list >= 3:
                first_qtr_inc = 0
                second_qtr_inc = inc_list.iloc[2]
                third_qtr_inc = inc_list.iloc[1]
                current_qtr_inc = inc_list.iloc[0]
            elif lenth_inc_list >= 2:
                first_qtr_inc = 0
                second_qtr_inc = 0
                third_qtr_inc = inc_list.iloc[1]
                current_qtr_inc = inc_list.iloc[0]
            elif lenth_inc_list >= 1:
                first_qtr_inc = 0
                second_qtr_inc = 0
                third_qtr_inc = 0
                current_qtr_inc = inc_list.iloc[0]
        except:
            first_qtr_inc = 0
            second_qtr_inc = 0
            third_qtr_inc = 0
            current_qtr_inc = 0

        #Total Revenue
        try:
            rev_list = inc_stat.loc['Total Revenue'].dropna()
            #print(inc_list)
            lenth_rev_list = len(rev_list)
            if lenth_rev_list >= 4:
                first_qtr_rev = rev_list.iloc[3]
                second_qtr_rev = rev_list.iloc[2]
                third_qtr_rev = rev_list.iloc[1]
                current_qtr_rev = rev_list.iloc[0]
            elif lenth_rev_list >= 3:
                first_qtr_rev = 0
                second_qtr_rev = rev_list.iloc[2]
                third_qtr_rev = rev_list.iloc[1]
                current_qtr_rev = rev_list.iloc[0]
            elif lenth_rev_list >= 2:
                first_qtr_rev = 0
                second_qtr_rev = 0
                third_qtr_rev = rev_list.iloc[1]
                current_qtr_rev = rev_list.iloc[0]
            elif lenth_rev_list >= 1:
                first_qtr_rev = 0
                second_qtr_rev = 0
                third_qtr_rev = 0
                current_qtr_rev = rev_list.iloc[0]
        except:
            first_qtr_rev = 0
            second_qtr_rev = 0
            third_qtr_rev = 0
            current_qtr_rev = 0
        
        #profit_margin
        if first_qtr_inc > 0 and first_qtr_rev > 0:
            first_qtr_Pmar = first_qtr_inc / first_qtr_rev
        else:
            first_qtr_Pmar = 0 
        if second_qtr_inc > 0 and second_qtr_rev > 0:
            second_qtr_Pmar = second_qtr_inc / second_qtr_rev
        else:
            second_qtr_Pmar = 0 
        if third_qtr_inc > 0 and third_qtr_rev > 0:
            third_qtr_Pmar = third_qtr_inc / third_qtr_rev
        else:
            third_qtr_Pmar = 0 
        if current_qtr_inc > 0 and current_qtr_rev > 0:
            current_qtr_Pmar = current_qtr_inc / current_qtr_rev
        else:
            current_qtr_Pmar = 0 

        if (current_qtr_eps > third_qtr_eps > second_qtr_eps > first_qtr_eps) and (current_qtr_inc > third_qtr_inc > second_qtr_inc > first_qtr_inc) and (current_qtr_Pmar > third_qtr_Pmar > second_qtr_Pmar > first_qtr_Pmar):
            code33 = 'T'
        else:
            code33 = 'F'

        print ({
            'Symbol': symbol,
            '1st qtr EPS': first_qtr_eps,
            '2nd qtr EPS': second_qtr_eps,
            '3rd qtr EPS': third_qtr_eps,
            'Current qtr EPS': current_qtr_eps,
            '1st qtr Inc': first_qtr_inc,
            '2nd qtr Inc': second_qtr_inc,
            '3rd qtr Inc': third_qtr_inc,
            'Current qtr Inc': current_qtr_inc,
            '1st qtr Rev': first_qtr_rev,
            '2nd qtr Rev': second_qtr_rev,
            '3rd qtr Rev': third_qtr_rev,
            'Current qtr Rev': current_qtr_rev,
            '1st qtr Profit Margin': first_qtr_Pmar,
            '2nd qtr Profit Margin': second_qtr_Pmar,
            '3rd qtr Profit Margin': third_qtr_Pmar,
            'Current qtr Profit Margin': current_qtr_Pmar,
            'Code 33': code33
        })

test_block(symbols)

