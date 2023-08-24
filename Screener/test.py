import pandas as pd
import yfinance as yf

# Define symbols and date range
symbols = ['AAPL', 'MSFT', 'GOOG'] 
start = '2023-01-01'
end = '2023-03-31' 

# Download daily price data
data = yf.download(symbols, start=start, end=end)['Adj Close']

# Calculate daily returns for each stock and S&P 500
returns = data.pct_change()
sp500 = yf.download('^GSPC', start=start, end=end)['Adj Close'].pct_change()

# Calculate 14-day relative strength 
rs = returns.rolling(14).corr(sp500)

# Get maximum RS over the period  
max_rs = rs.max()

# Rank stocks by closeness to max RS
rs_rank = max_rs.rank(method='dense', ascending=False)

# Print results
print(rs_rank)