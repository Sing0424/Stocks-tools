import yfinance as yf

stock_data = yf.download(tickers = 'AMAM', period='1y')

print(stock_data)