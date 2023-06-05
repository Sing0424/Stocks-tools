import yfinance as yf
import pandas as pd

# Define stock ticker and universe
stock_ticker = "NVDA"
universe = ["NVDA", "AMZN", "GOOG", "MSFT", "FB"]

# Download historical price data for stock and universe
stock_data = yf.download(stock_ticker, start="2022-06-05", end="2023-06-05")
universe_data = yf.download(universe, start="2022-06-05", end="2023-06-05")["Adj Close"]

# Calculate percentage change in stock price and universe price
stock_pct_change = stock_data["Adj Close"].pct_change()
universe_pct_change = universe_data.pct_change()

# Calculate cumulative percentage change for each stock
cum_pct_change = (universe_pct_change + 1).cumprod()

# Calculate relative strength rank (RSR) for each stock
rsr = pd.Series(index=universe)
for ticker in universe:
    rsr[ticker] = (
        cum_pct_change[ticker].iloc[-1] / cum_pct_change[stock_ticker].iloc[-1]
    )
rs_ranking = (rsr.rank(ascending=False) / len(universe)) * 98 + 1

# Print relative strength ranking for stock
print(f"Relative strength ranking for {stock_ticker}: {rs_ranking[stock_ticker]:.2f}")
