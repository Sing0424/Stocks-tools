# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    # API Key
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')
    TG_CHAT_ID = int(os.getenv('TG_CHAT_ID'))
    
    # Folder paths
    data_folder  = os.path.join('.','data')
    GoogleAPI_folder = os.path.join('.','GoogleAPI')
    WEBAPP_DATA_FOLDER = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data'))

    # File paths
    LISTING_STATUS_FILE = os.path.join('.', 'data', 'listing_status.csv')
    FILTERED_SYMBOLS_FILE = os.path.join('.', 'data', 'filtered_symbols.csv')
    FINAL_RESULTS_FILE = os.path.join('.', 'data', 'screenResults.csv')
    CONSOLIDATED_PRICE_DATA_FILE = os.path.join('.', 'data', 'consolidated_price_data.csv')

    # Google drive credential path
    CREDENTIAL = os.path.join('.', 'GoogleAPI', 'credentials.json')
    TOKEN = os.path.join('.', 'GoogleAPI', 'token.json')

    # Web app data paths
    CONSOLIDATED_PRICE_DATA_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','consolidated_price_data.csv'))
    FINAL_RESULTS_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','screenResults.csv'))
    
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
    MIN_PRICE = 12
    MIN_CLOSE_VOLUME_30D = 8000000
    MIN_52W_LOW_INCREASE_FACTOR = 1.25
    MIN_52W_HIGH_DECREASE_FACTOR = 0.75
    # The stock's 200-day moving average should be trending up for at least 1 month
    SMA200_TREND_DAYS = 21

    # RS score weights for 3, 6, 9, and 12 months
    RS_WEIGHT_3M = 0.4
    RS_WEIGHT_6M = 0.2
    RS_WEIGHT_9M = 0.2
    RS_WEIGHT_12M = 0.2
    
    # Flags
    FORCE_REFRESH_SYMBOLS = True
    FORCE_REFRESH_FILTERS = True
    FORCE_REFRESH_PRICE_DATA = True
    DOWNLOAD_FOR_WEBAPP = False
