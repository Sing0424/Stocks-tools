# config.py
import os
import math

class Config:
    # API Key
    ALPHA_VANTAGE_API_KEY = '7D80AAZF1EFC0TZJ'  # Replace with your key
    
    # File paths
    LISTING_STATUS_FILE = os.path.abspath(os.path.join('.', 'data', 'listing_status.csv'))
    FILTERED_SYMBOLS_FILE = os.path.abspath(os.path.join('.', 'data', 'filtered_symbols.csv'))
    TECHNICAL_RESULTS_FILE = os.path.abspath(os.path.join('.', 'data', 'technical_results.csv'))
    FINAL_RESULTS_FILE = os.path.abspath(os.path.join('.', 'data', 'screenResults.csv'))
    CONSOLIDATED_PRICE_DATA_FILE = os.path.abspath(os.path.join('.', 'data', 'consolidated_price_data.csv'))

    # Web app data paths
    CONSOLIDATED_PRICE_DATA_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','consolidated_price_data.csv'))
    FINAL_RESULTS_FILE_WEBAPP = os.path.abspath(os.path.join('..', 'stock-chart-viewer', 'public', 'data','screenResults.csv'))
    
    # Data download
    floored_num = math.floor(os.cpu_count() / 2)
    if floored_num % 2 == 0:
        MAX_WORKERS = int(floored_num)
    else:
        MAX_WORKERS = int(floored_num - 1)
    
    # Data download
    PRICE_DATA_PERIOD = "13mo"
    
    # Screening criteria
    MIN_RS_RANK = 70
    
    # Flags
    FORCE_REFRESH_SYMBOLS = True
    FORCE_REFRESH_FILTERS = True
    FORCE_REFRESH_PRICE_DATA = True
