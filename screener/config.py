# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    BASE_DIR = BASE_DIR

    # API Keys and External Services
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')
    _raw_chat_id = os.getenv('TG_CHAT_ID')
    try:
        TG_CHAT_ID = int(_raw_chat_id) if _raw_chat_id and _raw_chat_id.strip().lstrip('-').isdigit() else None
    except (ValueError, TypeError):
        TG_CHAT_ID = None
    ENABLE_TELEGRAM = bool(TG_BOT_TOKEN and TG_CHAT_ID)
    
    # Google Drive configuration
    GDRIVE_FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID', '1XJmBN164biI7oE3c_ZuQp7UHn7pa3tiG')
    GDRIVE_FILE_ID = os.getenv('GDRIVE_FILE_ID', '1xHoV8EW40ziRAud57N28kOlw_G_RimpYUpN5LH8sVNs')

    # Folder paths
    data_folder = os.path.join(BASE_DIR, 'data')
    GoogleAPI_folder = os.path.join(BASE_DIR, 'GoogleAPI')
    WEBAPP_DATA_FOLDER = os.path.join(BASE_DIR, 'stock-chart-viewer', 'public', 'data')

    # File paths
    LISTING_STATUS_FILE = os.path.join(data_folder, 'listing_status.csv')
    FILTERED_SYMBOLS_FILE = os.path.join(data_folder, 'filtered_symbols.csv')
    CONSOLIDATED_PRICE_DATA_FILE = os.path.join(data_folder, 'consolidated_price_data.csv')
    CONSOLIDATED_PRICE_DATA_PARQUET = os.path.join(data_folder, 'consolidated_price_data.parquet')
    EXCEL_REPORT_FILE = os.path.join(data_folder, 'final_report.xlsx')

    # Google drive credential paths
    CREDENTIAL = os.path.join(GoogleAPI_folder, 'credentials.json')
    TOKEN = os.path.join(GoogleAPI_folder, 'token.json')

    # Web app data paths
    CONSOLIDATED_PRICE_DATA_FILE_WEBAPP = os.path.join(WEBAPP_DATA_FOLDER, 'consolidated_price_data.csv')
    
    # CPU Threads config
    if os.cpu_count() >= 4:
        WORKERS = 4
        BATCH_SIZE = 24
    elif os.cpu_count() >= 2:
        WORKERS = 2
        BATCH_SIZE = 12
    else:
        WORKERS = 1
        BATCH_SIZE = 3
    
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
    USE_PARQUET = True
