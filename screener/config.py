# config.py
import os
import math

class Config:

    # API Key
    ALPHA_VANTAGE_API_KEY = '7D80AAZF1EFC0TZJ'  # Replace with your key
    
    # Folder paths
    data_folder  = os.path.join('.','data')
    GoogleAPI_folder = os.path.join('.','GoogleAPI')

    # File paths
    LISTING_STATUS_FILE = os.path.join('.', 'data', 'listing_status.csv')
    FILTERED_SYMBOLS_FILE = os.path.join('.', 'data', 'filtered_symbols.csv')
    TECHNICAL_RESULTS_FILE = os.path.join('.', 'data', 'technical_results.csv')
    FINAL_RESULTS_FILE = os.path.join('.', 'data', 'screenResults.csv')
    CONSOLIDATED_PRICE_DATA_FILE = os.path.join('.', 'data', 'consolidated_price_data.csv')

    # Google drive credential path
    CREDENTIAL = os.path.join('.', 'GoogleAPI', 'credentials.json')
    TOKEN = os.path.join('.', 'GoogleAPI', 'token.json')

    # Web app data paths
    CONSOLIDATED_PRICE_DATA_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','consolidated_price_data.csv'))
    FINAL_RESULTS_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','screenResults.csv'))
    
    # CPU Threads config
    MAX_WORKERS = os.cpu_count()
    if os.cpu_count() >= 4:
        DOWNLOAD_WORKERS = 4 #4
        BATCH_SIZE = 72
    elif os.cpu_count() >= 2:
        DOWNLOAD_WORKERS = 2
        BATCH_SIZE = 24
    else:
        DOWNLOAD_WORKERS = 1
        BATCH_SIZE = 12
    
    # Data download
    PRICE_DATA_PERIOD = "13mo"
    
    # Screening criteria
    MIN_RS_RANK = 89
    
    # Flags
    FORCE_REFRESH_SYMBOLS = True
    FORCE_REFRESH_FILTERS = True
    FORCE_REFRESH_PRICE_DATA = True
