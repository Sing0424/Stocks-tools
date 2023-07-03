import os
import pandas as pd
import yfinance as yf
import yahooquery as yq
from config import daily_rs_rating_Top_30_path, screen_result_path
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from tqdm import tqdm
import datetime

@cache
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

    ticker = yf.Ticker(symbol)
    info = ticker.info
    sector = info.get('sector')
    industry = info.get('industry')

    yq_stock_data = yq.Ticker(symbol)
    try:
        inc_stat = yq_stock_data.income_statement('q')
        eps_list = pd.DataFrame(inc_stat['DilutedEPS']).dropna()
        lenth_eps_list = len(eps_list)
        first_qtr_eps = eps_list.iloc[lenth_eps_list-3,0]
        second_qtr_eps = eps_list.iloc[lenth_eps_list-2,0]
        current_qtr_eps = eps_list.iloc[lenth_eps_list-1,0]
    except:
        first_qtr_eps = 0
        second_qtr_eps = 0
        current_qtr_eps = 0

    try:
        inc_stat = yq_stock_data.income_statement('q')
        inc_list = pd.DataFrame(inc_stat['NetIncome']).dropna()
        lenth_inc_list = len(inc_list)
        first_qtr_inc = inc_list.iloc[lenth_inc_list-5,0]
        second_qtr_inc = inc_list.iloc[lenth_inc_list-4,0]
        current_qtr_inc = inc_list.iloc[lenth_inc_list-2,0]
    except:
        first_qtr_inc = 0
        second_qtr_inc = 0
        current_qtr_inc = 0
                                
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
        '1st qtr EPS': first_qtr_eps,
        '2nd qtr EPS': second_qtr_eps,
        'Current qtr EPS': current_qtr_eps,
        '1st qtr Inc': first_qtr_inc,
        '2nd qtr Inc': second_qtr_inc,
        'current_qtr_inc': current_qtr_inc
    }

def Screener(symbol, rs_rating):
    stock_data = get_stock_data(symbol, rs_rating)
    
    if ((stock_data['Current price'] > stock_data['SMA 150']) and (stock_data['Current price'] > stock_data['SMA 200'])) and (stock_data['SMA 150'] > stock_data['SMA 200']) and (stock_data['SMA 200'] > stock_data['Month ago SMA 200']) and (stock_data['SMA 50'] > stock_data['SMA 150'] and stock_data['SMA 50'] > stock_data['SMA 200']) and (stock_data['Current price'] > stock_data['SMA 50']) and (stock_data['Current price'] > (1.3 * stock_data['Week 52 low'])) and (stock_data['Current price'] > (0.75 * stock_data['Week 52 high'])) and stock_data['30D Avg Vol'] >= 250000 and stock_data['Current qtr EPS'] > stock_data['2nd qtr EPS'] and stock_data['2nd qtr EPS'] > stock_data['1st qtr EPS'] and (datetime.datetime.now().year - datetime.datetime.strptime(stock_data['IPO Date'], "%Y-%m-%d").year) <= 10:
        return stock_data
    else:
        return None

def run_Screener():
    if __name__ == '__main__':
        Screen_result_list = []
        import_data = pd.read_excel(daily_rs_rating_Top_30_path)

        with ProcessPoolExecutor(max_workers=None) as executor:
            results = tqdm(executor.map(Screener, import_data["Symbol"], import_data["RS Rating"]), desc='Screening', unit=' stocks', total=len(import_data), ncols=80, miniters=1)

            for result in results:
                if result is not None:
                    Screen_result_list.append(result)

        Screen_result_list.sort(key=lambda x: x['RS Rating'], reverse = True)

        # Create a DataFrame from Screen_result_list
        df = pd.DataFrame(Screen_result_list)

        # Write the DataFrame to an Excel file
        output_file_path = os.path.join(screen_result_path)
        df.to_excel(output_file_path, index=False)

run_Screener()