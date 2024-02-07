import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_path, screen_result_path
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import time

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
    yq_info = yq.Ticker(symbol)
    try:
        sector = yq_info.summary_profile[symbol]['sector']
    except:
        sector = 'N/A'
    try:
        industry = yq_info.summary_profile[symbol]['industry']
    except:
        industry = 'N/A'

    inc_stat_q = ticker_data.quarterly_income_stmt

    try:
        eps_list = ticker_data.earnings_dates.reset_index(drop=True).dropna()['Reported EPS']
        lenth_eps_list = len(eps_list)
        if lenth_eps_list >= 5:
            YoY_eps = eps_list.iloc[4]
            before_last_qtr_eps = eps_list.iloc[2]
            last_qtr_eps = eps_list.iloc[1]
            current_qtr_eps = eps_list.iloc[0]
            eps_growth_perc_last_qtr = ((current_qtr_eps - last_qtr_eps) / last_qtr_eps) * 100
            eps_growth_perc_yester_qtr = ((last_qtr_eps - before_last_qtr_eps) / before_last_qtr_eps) * 100
            eps_growth_perc_current_YoY = ((current_qtr_eps - YoY_eps) / YoY_eps) * 100
    except:
        yesteryear_qtr_eps = 0
        last_qtr_eps = 0
        current_qtr_eps = 0
        eps_growth_perc_last_qtr = 0
        eps_growth_perc_yester_qtr = 0
        eps_growth_perc_current_YoY = 0

    #Total Revenue
    try:
        rev_list = inc_stat_q.loc['Total Revenue'].dropna()
        lenth_rev_list = len(rev_list)
        if lenth_rev_list >= 4:
            fourth_qtr_before_rev = rev_list.iloc[3]
            before_last_qtr_rev = rev_list.iloc[2]
            last_qtr_rev = rev_list.iloc[1]
            current_qtr_rev = rev_list.iloc[0]
            rev_growth_perc_current_qtr = ((current_qtr_rev - third_qtr_rev) / third_qtr_rev) * 100
            rev_growth_perc_last_qtr = ((last_qtr_rev - before_last_qtr_rev) / before_last_qtr_rev) * 100
            rev_growth_perc_last_before_qtr = ((before_last_qtr_rev - fourth_qtr_before_rev) / fourth_qtr_before_rev) * 100
    except:
        fourth_qtr_before_rev = 0
        before_last_qtr_rev = 0
        last_qtr_rev = 0
        current_qtr_rev = 0
        rev_growth_perc_current_qtr = 0
        rev_growth_perc_last_qtr = 0
        rev_growth_perc_last_before_qtr = 0

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
        'eps_growth_perc_YoY': eps_growth_perc_current_YoY,
        'eps_growth_perc_last_qtr': eps_growth_perc_last_qtr,
        'eps_growth_perc_yester_qtr': eps_growth_perc_yester_qtr,
        'rev_growth_perc_current_qtr': rev_growth_perc_current_qtr,
        'rev_growth_perc_last_qtr': rev_growth_perc_last_qtr,
        'rev_growth_perc_last_before_qtr': rev_growth_perc_last_before_qtr
    }

def Screener(symbol_rating_tuple):
    symbol = symbol_rating_tuple[0]
    rs_rating = symbol_rating_tuple[1]

    stock_data = get_stock_data(symbol, rs_rating)
    
    if ((stock_data['Current price'] > stock_data['SMA 150']) and (stock_data['Current price'] > stock_data['SMA 200'])) and (stock_data['SMA 150'] > stock_data['SMA 200']) and (stock_data['SMA 200'] > stock_data['Month ago SMA 200']) and (stock_data['SMA 50'] > stock_data['SMA 150'] and stock_data['SMA 50'] > stock_data['SMA 200']) and (stock_data['Current price'] > stock_data['SMA 50']) and (stock_data['Current price'] > (1.3 * stock_data['Week 52 low'])) and (stock_data['Current price'] > (0.75 * stock_data['Week 52 high'])) and stock_data['30D Avg Vol'] >= 250000:
        return stock_data
    else:
        return None

    # and stock_data['eps_growth_perc_YoY'] >= 25 and stock_data['eps_growth_perc_last_qtr'] >= 25 and stock_data['eps_growth_perc_yester_qtr'] >= 25 and (stock_data['rev_growth_perc_current_qtr'] >= 25 | (stock_data['rev_growth_perc_current_qtr'] > 0 and stock_data['rev_growth_perc_last_qtr'] > 0 and stock_data['rev_growth_perc_last_before_qtr'] > 0))

def run_Screener():
    if __name__ == '__main__':
        Screen_result_list = []
        import_data = pd.read_excel(daily_rs_rating_path)

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