import os
import pandas as pd
import yfinance as yf
from config import daily_rs_rating_Top_30_path, screen_result_path
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from tqdm import tqdm
import time

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
    avg_vol_50 = stock_data['Volume'].rolling(window=50).mean()

    ticker = yf.Ticker(symbol)
    info = ticker.info
    sector = info.get('sector')
    industry = info.get('industry')
                                
    return {
        'Symbol': symbol,
        'Sector': sector,
        'Industry': industry,
        'IPO Date': ipo_date,
        'Current_price': stock_data['Adj Close'][-1],
        '50D_Avg_Vol': avg_vol_50[-1],
        'Sma_50': sma_50[-1],
        'Sma_150': sma_150[-1],
        'Sma_200': sma_200[-1],
        'Month_ago_sma_200': month_ago_sma_200,
        'Week_52_low': stock_data['Adj Close'].min(),
        'Week_52_high': stock_data['Adj Close'].max(),
        'RS Rating': rsr
    }

def Screener(symbol, rs_rating):
    stock_data = get_stock_data(symbol, rs_rating)
    
    #Condition 1: Current Price > 150 SMA and > 200 SMA
    if ((stock_data['Current_price'] > stock_data['Sma_150']) and (stock_data['Current_price'] > stock_data['Sma_200'])) and (stock_data['Sma_150'] > stock_data['Sma_200']) and (stock_data['Sma_200'] > stock_data['Month_ago_sma_200']) and (stock_data['Sma_50'] > stock_data['Sma_150'] and stock_data['Sma_50'] > stock_data['Sma_200']) and (stock_data['Current_price'] > stock_data['Sma_50']) and (stock_data['Current_price'] > (1.3 * stock_data['Week_52_low'])) and (stock_data['Current_price'] > (0.75 * stock_data['Week_52_high'])) and stock_data['50D_Avg_Vol'] >= 50000:
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