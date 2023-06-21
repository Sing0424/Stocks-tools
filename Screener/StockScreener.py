import pandas as pd
import yfinance as yf

def get_stock_data(symbol):
    stock_data = yf.download(tickers = symbol, period='1y')
    sma_50 = stock_data['Adj Close'].rolling(window=50).mean()
    sma_150 = stock_data['Adj Close'].rolling(window=150).mean()
    sma_200 = stock_data['Adj Close'].rolling(window=200).mean()
    try:
        month_ago_sma_200 = sma_200[-21]
    except:
        month_ago_sma_200 = 0
                                
    return {
        'Symbol': symbol,
        'Current_price': stock_data['Adj Close'][-1],
        'Sma_50': sma_50[-1],
        'Sma_150': sma_150[-1],
        'Sma_200': sma_200[-1],
        'Month_ago_sma_200': month_ago_sma_200,
        'Seek_52_low': stock_data['Adj Close'].min(),
        'Week_52_high': stock_data['Adj Close'].max(),
    }

def Screener():
    # Read input Excel file
    input_df = pd.read_excel(r'C:\Ivan\Repo\Stocks-tools\Screener\data\csv\rs_rating_top_30.xlsx')

    for index, row in input_df.iterrows():
        stock_symbol = row['Symbol']
        rs_rating = row['RS Rating']
        stock_data = get_stock_data(str(stock_symbol))
        print(stock_data)
        
		
        
Screener()
