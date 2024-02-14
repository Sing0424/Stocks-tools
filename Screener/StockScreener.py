import os
import pandas as pd
import yfinance as yf
from config import daily_rs_rating_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time
import logging

def get_stock_data(symbol, rsr):
    stock_data = yf.download(tickers = symbol, period='max', progress=False)
    try:
        ipo_date = stock_data.index[0].strftime("%Y-%m-%d")
    except:
        ipo_date = 0
    sma_50 = stock_data['Adj Close'].rolling(window=50).mean()
    sma_150 = stock_data['Adj Close'].rolling(window=150).mean()
    sma_200 = stock_data['Adj Close'].rolling(window=200).mean()
    try:
        month_ago_sma_200 = sma_200[-21]
    except:
        month_ago_sma_200 = 0
    avg_vol_30 = stock_data['Volume'].rolling(window=30).mean()

    ticker_data = yf.Ticker(symbol)

    info = ticker_data.info
    sector = info.get('sector')
    industry = info.get('industry')

    inc_stat_q = ticker_data.quarterly_income_stmt
    inc_stat_a = ticker_data.income_stmt

    try:
        try:
            logging.basicConfig(level=logging.CRITICAL)
            eps_list = ticker_data.get_earnings_dates(limit=20).reset_index(drop=True).dropna()['Reported EPS']
            logging.basicConfig(level=logging.WARNING)
        except:
            eps_list = []
        lenth_eps_list = len(eps_list)
        if lenth_eps_list >= 7:
            EPS_QoQ_C = round(((eps_list.iloc[0] - eps_list.iloc[4]) / eps_list.iloc[4]) * 100)
            EPS_QoQ_LQ = round(((eps_list.iloc[1] - eps_list.iloc[5]) / eps_list.iloc[5]) * 100)
            EPS_QoQ_L2Q = round(((eps_list.iloc[2] - eps_list.iloc[6]) / eps_list.iloc[6]) * 100)
        elif lenth_eps_list >= 6:
            EPS_QoQ_C = round(((eps_list.iloc[0] - eps_list.iloc[4]) / eps_list.iloc[4]) * 100)
            EPS_QoQ_LQ = round(((eps_list.iloc[1] - eps_list.iloc[5]) / eps_list.iloc[5]) * 100)
            EPS_QoQ_L2Q = 0
        elif lenth_eps_list >= 5:
            EPS_QoQ_C = round(((eps_list.iloc[0] - eps_list.iloc[4]) / eps_list.iloc[4]) * 100)
            EPS_QoQ_LQ = 0
            EPS_QoQ_L2Q = 0
        else:
            EPS_QoQ_C = 0
            EPS_QoQ_LQ = 0
            EPS_QoQ_L2Q = 0
    except:
            EPS_QoQ_C = 0
            EPS_QoQ_LQ = 0
            EPS_QoQ_L2Q = 0

    #Total Revenue
    try:
        rev_list = inc_stat_q.loc['Total Revenue'].dropna()
        lenth_rev_list = len(rev_list)
        if lenth_rev_list >= 4:
            REV_C = round(((rev_list.iloc[0] - rev_list.iloc[1]) / rev_list.iloc[1]) * 100)
            REV_LQ = round(((rev_list.iloc[1] - rev_list.iloc[2]) / rev_list.iloc[2]) * 100)
            REV_L2Q = round(((rev_list.iloc[2] - rev_list.iloc[3]) / rev_list.iloc[3]) * 100)
        elif lenth_rev_list >= 3:
            REV_C = round(((rev_list.iloc[0] - rev_list.iloc[1]) / rev_list.iloc[1]) * 100)
            REV_LQ = round(((rev_list.iloc[1] - rev_list.iloc[2]) / rev_list.iloc[2]) * 100)
            REV_L2Q = 0
        else:
            REV_C = 0
            REV_LQ = 0
            REV_L2Q = 0
    except:
            REV_C = 0
            REV_LQ = 0
            REV_L2Q = 0

    try:
        EPS_list_A = inc_stat_a.loc['Diluted EPS']
        lenth_EPS_list_A = len(EPS_list_A)
        if lenth_EPS_list_A >=2:
            EPS_A = round(((EPS_list_A.iloc[0] - EPS_list_A.iloc[1]) / EPS_list_A.iloc[1]) * 100)
            EPS_A1 = round(((EPS_list_A.iloc[1] - EPS_list_A.iloc[2]) / EPS_list_A.iloc[2]) * 100)
        elif lenth_EPS_list_A >=1:
            EPS_A = round(((EPS_list_A.iloc[0] - EPS_list_A.iloc[1]) / EPS_list_A.iloc[1]) * 100)
            EPS_A1 = 0
    except:
        EPS_A = 0
        EPS_A1 = 0

    return {
        'Symbol': symbol,
        'Sector': sector,
        'Industry': industry,
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
        'EPS_QoQ_C': EPS_QoQ_C,
        'EPS_QoQ_LQ': EPS_QoQ_LQ,
        'EPS_QoQ_L2Q': EPS_QoQ_L2Q,
        'EPS_A': EPS_A,
        'EPS_A1': EPS_A1,
        'REV_C': REV_C,
        'REV_LQ': REV_LQ,
        'REV_L2Q': REV_L2Q
    }

def Screener(symbol_rating_tuple):

    symbol = symbol_rating_tuple[0]
    rs_rating = symbol_rating_tuple[1]

    stock_data = get_stock_data(symbol, rs_rating)
    
    if ((stock_data['Current price'] > stock_data['SMA 150']) and (stock_data['Current price'] > stock_data['SMA 200'])) and (stock_data['SMA 150'] > stock_data['SMA 200']) and (stock_data['SMA 200'] > stock_data['Month ago SMA 200']) and (stock_data['SMA 50'] > stock_data['SMA 150'] and stock_data['SMA 50'] > stock_data['SMA 200']) and (stock_data['Current price'] > stock_data['SMA 50']) and (stock_data['Current price'] > (1.3 * stock_data['Week 52 low'])) and (stock_data['Current price'] > (0.75 * stock_data['Week 52 high'])) and stock_data['30D Avg Vol'] >= 100000 and stock_data['EPS_QoQ_C'] >= 20 and stock_data['EPS_QoQ_LQ'] >= 20 and stock_data['EPS_QoQ_L2Q'] >= 20 and stock_data['EPS_A'] > 0 and stock_data['REV_C'] > 0:
        return stock_data
    else:
        return None

def run_Screener():
    if __name__ == '__main__':
        Screen_result_list = []
        import_data = pd.read_excel(daily_rs_rating_path)

        args = zip(import_data["Symbol"], import_data["RS Rating"])

        cpu_count = 1 #os.cpu_count() / 2
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