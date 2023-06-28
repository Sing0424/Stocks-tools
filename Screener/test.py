import yfinance as yf
import logging

for key in logging.Logger.manager.loggerDict:
    print(key)

logging.getLogger('urllib3').setLevel(logging.NOTSET)
logging.getLogger('requests').setLevel(logging.NOTSET)
logging.getLogger('charset_normalizer').setLevel(logging.NOTSET)

stock_data = yf.download('AFGS', period='1y', progress=False, threads = True)
print('aaa')
