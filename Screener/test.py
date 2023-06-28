import yfinance as yf

stock_data = yf.download('AFGS', period='1y', progress=False, threads = True)

print('aaa')
if stock_data.empty:
    print('bbb')