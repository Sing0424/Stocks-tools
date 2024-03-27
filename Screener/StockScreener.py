import os
import pandas as pd
import yfinance as yf
from yahooquery import Ticker
from config import rs_rating_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time
import logging

def get_stock_data(symbol, rsr):
    ticker_data = yf.Ticker(symbol)
    stock_data = ticker_data.history(period="max")
    try:
        ipo_date = stock_data.index[0].strftime("%Y-%m-%d")
    except:
        ipo_date = 0

    sma_50 = stock_data['Close'].rolling(window=50).mean()
    sma_150 = stock_data['Close'].rolling(window=150).mean()
    sma_200 = stock_data['Close'].rolling(window=200).mean()

    try:
        month_ago_sma_200 = sma_200[-21]
    except:
        month_ago_sma_200 = 0

    week52_high = stock_data.tail(250)['Close'].max()
    week52_low = stock_data.tail(250)['Close'].min()

    try:
        if stock_data['Close'][-1] > stock_data['Close'][-64]:
            growth_in_qtr = ((stock_data['Close'][-1] / stock_data['Close'][-64]) - 1) * 100
        else:
            growth_in_qtr = 0
    except:
        growth_in_qtr = 0

    try:
        if stock_data['Close'][-1] > stock_data['Close'][-250]:
            growth_in_yr = ((stock_data['Close'][-1] / stock_data['Close'][-250]) - 1) * 100
        else:
            growth_in_yr = 0
    except:
        growth_in_yr = 0

    avg_vol_30 = stock_data['Volume'].rolling(window=30).mean()[-1]

    info = ticker_data.info
    sector = info.get('sector')
    industry = info.get('industry')

    inc_stat_q = ticker_data.quarterly_income_stmt
    inc_stat_a = ticker_data.income_stmt

    #ROE
    yq_data = Ticker(symbol)
    try:
        ROE = (yq_data.financial_data[symbol]['returnOnEquity']) * 100
    except:
        ROE = 0

    #EPS Q/Q
    try:
        if pd.notna(inc_stat_q.loc['Diluted EPS'][0]) and pd.notna(inc_stat_q.loc['Diluted EPS'][4]):
            EPS_Q = round(((inc_stat_q.loc['Diluted EPS'][0] - inc_stat_q.loc['Diluted EPS'][4]) / inc_stat_q.loc['Diluted EPS'][4]) * 100)
        elif pd.isna(inc_stat_q.loc['Diluted EPS'][0]) and pd.notna(inc_stat_q.loc['Diluted EPS'][4]):
            EPS_Q = round(((yq_data.income_statement(frequency="q", trailing=True)["DilutedEPS"].reset_index(drop=True).iloc[4] - inc_stat_q.loc['Diluted EPS'][4]) / inc_stat_q.loc['Diluted EPS'][4]) * 100)
        else:
            EPS_Q = 0
    except:
        EPS_Q = 0

    #Total Revenue
    try:
        REV_Q_LIST = inc_stat_q.loc['Operating Revenue'].dropna()
        if pd.notna(REV_Q_LIST[0]) and pd.notna(REV_Q_LIST[4]):
            REV_Q = round(((REV_Q_LIST.iloc[0] - REV_Q_LIST.iloc[4]) / abs(REV_Q_LIST.iloc[4])) * 100)
        elif pd.isna(REV_Q_LIST[0]) and pd.notna(REV_Q_LIST[4]):
            REV_Q = round(((yq_data.income_statement(frequency="q", trailing=False)["OperatingRevenue"][4] - REV_Q_LIST.iloc[4]) / abs(REV_Q_LIST.iloc[4])) * 100)
    except:
            REV_Q = 0

    #EPS Annual
    try:
        EPS_A_LIST = inc_stat_a.loc['Diluted EPS']
        if pd.notna(EPS_A_LIST[0]) and pd.notna(EPS_A_LIST[1]):
            EPS_A = round(((EPS_A_LIST.iloc[0] - EPS_A_LIST.iloc[1]) / abs(EPS_A_LIST.iloc[1])) * 100)
        elif pd.isna(EPS_A_LIST[0]) and pd.notna(EPS_A_LIST[1]):
            EPS_A = round(((yq_data.income_statement(frequency="a", trailing=False)["DilutedEPS"][3] - inc_stat_q.loc['Diluted EPS'][1]) / abs(inc_stat_q.loc['Diluted EPS'][1])) * 100)
        else:
            EPS_A = 0
    except:
        EPS_A = 0

    return {
        'Symbol': symbol,
        'Sector': sector,
        'Industry': industry,
        'IPO Date': ipo_date,
        'Current price': stock_data['Close'][-1],
        '30D Avg Vol': avg_vol_30,
        'SMA 50': sma_50[-1],
        'SMA 150': sma_150[-1],
        'SMA 200': sma_200[-1],
        'Month ago SMA 200': month_ago_sma_200,
        '52 Week low': week52_low,
        '52 Week high': week52_high,
        'growth_in_qtr': growth_in_qtr,
        'growth_in_yr': growth_in_yr,
        'RS Rating': rsr,
        'EPS_Q': EPS_Q,
        'EPS_A': EPS_A,
        'REV_Q': REV_Q,
        'ROE': ROE
    }

def Screener(symbol_rating_tuple):

    symbol = symbol_rating_tuple[0]
    rs_rating = symbol_rating_tuple[1]

    stock_data = get_stock_data(symbol, rs_rating)

    #if ((stock_data['Current price'] > stock_data['SMA 150']) and (stock_data['Current price'] > stock_data['SMA 200'])) and (stock_data['SMA 150'] > stock_data['SMA 200']) and (stock_data['SMA 200'] > stock_data['Month ago SMA 200']) and (stock_data['SMA 50'] > stock_data['SMA 150'] and stock_data['SMA 50'] > stock_data['SMA 200']) and (stock_data['Current price'] > stock_data['SMA 50']) and (stock_data['Current price'] > (1.3 * stock_data['52 Week low'])) and (stock_data['Current price'] > (0.75 * stock_data['52 Week high'])) and stock_data['30D Avg Vol'] >= 200000 and stock_data['EPS_Q'] >= 25 and stock_data['EPS_A'] > 25 and stock_data['REV_C'] > 25 and stock_data['ROE'] > 0 and (stock_data['growth_in_qtr'] > 20 or stock_data['growth_in_yr'] > 50):

    if ((stock_data['Current price'] > stock_data['SMA 150']) and (stock_data['Current price'] > stock_data['SMA 200'])) and (stock_data['SMA 150'] > stock_data['SMA 200']) and (stock_data['SMA 200'] > stock_data['Month ago SMA 200']) and (stock_data['SMA 50'] > stock_data['SMA 150'] and stock_data['SMA 50'] > stock_data['SMA 200']) and (stock_data['Current price'] > stock_data['SMA 50']) and (stock_data['Current price'] > (1.3 * stock_data['52 Week low'])) and (stock_data['Current price'] > (0.75 * stock_data['52 Week high'])) and stock_data['30D Avg Vol'] >= 200000 and stock_data['ROE'] > 0 and stock_data['EPS_Q'] >= 25 and stock_data['EPS_A'] > 25 and stock_data['REV_Q'] > 10 and (stock_data['growth_in_qtr'] > 30 or stock_data['growth_in_yr'] > 50) :
        return stock_data
    else:
        return None

def run_Screener():
    if __name__ == '__main__':
        Screen_result_list = []
        import_data = pd.read_csv(rs_rating_path)

        args = zip(import_data["Symbol"], import_data["RS Rating"])

        cpu_count = os.cpu_count() / 2
        pool = Pool(processes=int(cpu_count))

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
        df.to_excel(output_file_path, index=False, freeze_panes=[1,1])

run_Screener()