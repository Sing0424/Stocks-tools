import pandas as pd

def add_ema(df: pd.DataFrame, spans: list[int]) -> pd.DataFrame:
    """批次計算 EMA，回傳含 EMA 欄位的 DataFrame"""
    for span in spans:
        df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    return df

def add_sma(df: pd.DataFrame, spans: dict[int,str]) -> pd.DataFrame:
    """批次計算 SMA，spans 為 {period: color}"""
    for span in spans:
        df[f'SMA{span}'] = df['Close'].rolling(window=span).mean()
    return df
