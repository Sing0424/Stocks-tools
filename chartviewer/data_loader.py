# data_loader.py

import pandas as pd
import pytz

def load_consolidated_data(path: str = './data/consolidated_price_data.csv') -> pd.DataFrame:
    """
    讀取本地合併資料並解析時區為 UTC
    """
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    return df

def prepare_symbol_data(
    df: pd.DataFrame,
    symbol: str,
    timezone: str = 'Asia/Hong_Kong',
    resample_weekly: bool = False
) -> pd.DataFrame:
    """
    篩選單一 symbol，轉換時區、可選擇週線重採樣
    """
    hkt = pytz.timezone(timezone)
    data = df[df['Symbol'] == symbol].copy()
    data['Date'] = data['Date'].dt.tz_convert(hkt)

    if resample_weekly:
        data = (
            data.set_index('Date')
                .resample('W', closed='right', label='right')
                .agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                .dropna()
                .reset_index()
        )

    data = data.sort_values('Date')
    data['Date_str'] = data['Date'].dt.strftime('%Y-%m-%d')
    return data
