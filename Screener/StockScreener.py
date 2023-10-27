import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_Top_30_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time

def get_stock_data(symbol, rsr):
    stock_data = yf.download(tickers = symbol, period='max', progress=False)
    ipo_date = stock_data.index[0].strftime("%Y-%m-%d")
    sma_50 = stock_data['Adj Close'].rolling(window=50).mean()
    sma_150 = stock_data['Adj Close'].rolling(window=150).mean()
    sma_200 = stock_data['Adj Close'].rolling(window=200).mean()
    try:
        month_ago_sma_200 = sma_200[-21]
    except:
        month_ago_sma_200 = 0
    avg_vol_30 = stock_data['Volume'].rolling(window=30).mean()

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
        elif lenth_eps_list == 3:
            first_qtr_eps = 0
            second_qtr_eps = eps_list.iloc[2]
            third_qtr_eps = eps_list.iloc[1]
            current_qtr_eps = eps_list.iloc[0]
        elif lenth_eps_list == 2:
            first_qtr_eps = 0
            second_qtr_eps = 0
            third_qtr_eps = eps_list.iloc[1]
            current_qtr_eps = eps_list.iloc[0]
        elif lenth_eps_list == 1:
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
        elif lenth_inc_list == 3:
            first_qtr_inc = 0
            second_qtr_inc = inc_list.iloc[2]
            third_qtr_inc = inc_list.iloc[1]
            current_qtr_inc = inc_list.iloc[0]
        elif lenth_inc_list == 2:
            first_qtr_inc = 0
            second_qtr_inc = 0
            third_qtr_inc = inc_list.iloc[1]
            current_qtr_inc = inc_list.iloc[0]
        elif lenth_inc_list == 1:
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
        rev_list = rev_stat.loc['Total Revenue'].dropna()
        #print(inc_list)
        lenth_rev_list = len(rev_list)
        if lenth_rev_list >= 4:
            first_qtr_rev = rev_list.iloc[3]
            second_qtr_rev = rev_list.iloc[2]
            third_qtr_rev = rev_list.iloc[1]
            current_qtr_rev = rev_list.iloc[0]
        elif lenth_rev_list == 3:
            first_qtr_rev = 0
            second_qtr_rev = rev_list.iloc[2]
            third_qtr_rev = rev_list.iloc[1]
            current_qtr_rev = rev_list.iloc[0]
        elif lenth_rev_list == 2:
            first_qtr_rev = 0
            second_qtr_rev = 0
            third_qtr_rev = rev_list.iloc[1]
            current_qtr_rev = rev_list.iloc[0]
        elif lenth_rev_list == 1:
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

    return {
        'Symbol': symbol,
        #'Sector': sector,
        #'Industry': industry,
        'IPO Date': ipo_date,
        'Current price': stock_data['Adj Close'][-1],
        '30D Avg Vol': avg_vol_30[-1],
        'SMA 50': sma_50[-1],
        'SMA 150': sma_150[-1],
        'SMA 200': sma_200[-1],
        'Month ago SMA 200': month_ago_sma_200,
        'Week 52 low': stock_data['Adj Close'].min(),
        'Week 52 high': stock_data['Adj Close'].max(),
        'RS Rating': rsr,
        '1st qtr EPS': first_qtr_eps,
        '2nd qtr EPS': second_qtr_eps,
        '3rd qtr EPS': third_qtr_eps,
        'Current qtr EPS': current_qtr_eps,
        '1st qtr Inc': first_qtr_inc,
        '2nd qtr Inc': second_qtr_inc,
        '3rd qtr Inc': third_qtr_inc,
        'Current qtr Inc': current_qtr_inc,
        '1st qtr Profit Margin': first_qtr_Pmar,
        '2nd qtr Profit Margin': second_qtr_Pmar,
        '3rd qtr Profit Margin': third_qtr_Pmar,
        'Current qtr Profit Margin': current_qtr_Pmar,
        'Code 33': code33
    }

def Screener(symbol_rating_tuple):
    symbol = symbol_rating_tuple[0]
    rs_rating = symbol_rating_tuple[1]

    stock_data = get_stock_data(symbol, rs_rating)
    
    if ((stock_data['Current price'] > stock_data['SMA 150']) and (stock_data['Current price'] > stock_data['SMA 200'])) and (stock_data['SMA 150'] > stock_data['SMA 200']) and (stock_data['SMA 200'] > stock_data['Month ago SMA 200']) and (stock_data['SMA 50'] > stock_data['SMA 150'] and stock_data['SMA 50'] > stock_data['SMA 200']) and (stock_data['Current price'] > stock_data['SMA 50']) and (stock_data['Current price'] > (1.3 * stock_data['Week 52 low'])) and (stock_data['Current price'] > (0.75 * stock_data['Week 52 high'])) and stock_data['30D Avg Vol'] >= 250000:
        return stock_data
    else:
        return None

def run_Screener():
    if __name__ == '__main__':
        Screen_result_list = []
        import_data = pd.read_excel(daily_rs_rating_Top_30_path)

        args = zip(import_data["Symbol"], import_data["RS Rating"])

        cpu_count = os.cpu_count() / 2
        pool = Pool(processes=int(cpu_count), maxtasksperchild=1)

        process_bar = tqdm(desc='Screening', unit=' stocks', total=len(import_data), ncols=80, smoothing=1, miniters=cpu_count)

        for result in pool.imap_unordered(Screener, args):
            if result is not None:
                Screen_result_list.append(result)
            process_bar.update()
        process_bar.close()

        Screen_result_list.sort(key=lambda x: x['RS Rating'], reverse = True)

        # Create a DataFrame from Screen_result_list
        df = pd.DataFrame(Screen_result_list)

        # Write the DataFrame to an Excel file
        output_file_path = os.path.join(screen_result_path)
        df.to_excel(output_file_path, index=False)

run_Screener()