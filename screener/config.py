# config.py

class Config:
    # API Key
    ALPHA_VANTAGE_API_KEY = '7D80AAZF1EFC0TZJ'  # Replace with your key
    
    # File paths
    LISTING_STATUS_FILE = '././data/listing_status.csv'
    FILTERED_SYMBOLS_FILE = '././data/filtered_symbols.csv'
    TECHNICAL_RESULTS_FILE = '././data/technical_results.csv'
    FINAL_RESULTS_FILE = '././ScreenResult/screenResults.csv'
    CONSOLIDATED_PRICE_DATA_FILE = '././data/consolidated_price_data.csv'
    
    # Data download
    MAX_WORKERS = 8
    PRICE_DATA_PERIOD = "13mo"
    
    # Screening criteria
    MIN_RS_RANK = 70
    
    # Flags
    FORCE_REFRESH_SYMBOLS = True
    FORCE_REFRESH_FILTERS = True
    FORCE_REFRESH_PRICE_DATA = True
