import yahoo_fin.stock_info as si
import pandas as pd

# get earnings history for AAPL
aapl_earnings_hist = si.get_earnings_history("aapl")

df = pd.DataFrame.from_dict(aapl_earnings_hist)

print(df)