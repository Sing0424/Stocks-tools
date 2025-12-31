# config.py
import os

class Config:

    # API Key
    ALPHA_VANTAGE_API_KEY = '7D80AAZF1EFC0TZJ'
    TG_BOT_TOKEN = '8157791654:AAHjR3iuS9s2OhhWjUAmb_bjgPd2kEsQXkY'
    TG_CHAT_ID = 730875759
    
    # Folder paths
    data_folder  = os.path.join('.','data')

    # File paths
    LISTING_STATUS_FILE = os.path.join('.', 'data', 'listing_status.csv')
    FILTERED_SYMBOLS_FILE = os.path.join('.', 'data', 'filtered_symbols.csv')
    FINAL_RESULTS_FILE = os.path.join('.', 'data', 'screenResults.csv')
    CONSOLIDATED_PRICE_DATA_FILE = os.path.join('.', 'data', 'consolidated_price_data.csv')

    # Google drive credential path
    CREDENTIAL = os.path.join('.', 'GoogleAPI', 'credentials.json')
    TOKEN = os.path.join('.', 'GoogleAPI', 'token.json')

    # Web app data paths
    # CONSOLIDATED_PRICE_DATA_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','consolidated_price_data.csv'))
    # FINAL_RESULTS_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','screenResults.csv'))
    
    # CPU Threads config
    if os.cpu_count() >= 4:
        WORKERS = 4 #4
        BATCH_SIZE = 72
    elif os.cpu_count() >= 2:
        WORKERS = 2
        BATCH_SIZE = 24
    else:
        WORKERS = 1
        BATCH_SIZE = 12
    
    # Data download
    # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    PRICE_DATA_PERIOD = "256d"
    
    # Screening criteria
    MIN_RS_RANK = 89
    
    # Flags
    FORCE_REFRESH_SYMBOLS = True
    FORCE_REFRESH_FILTERS = True
    FORCE_REFRESH_PRICE_DATA = True
    FORCE_REFRESH_ANALYZE_DATA = True
