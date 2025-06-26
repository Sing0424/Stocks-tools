import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

results = pd.read_csv('./ScreenResult/screenResults.csv')
symbols = results['symbol'].unique()

st.title("Stock Screener Candlestick Viewer")

symbol = st.selectbox("Select stock", symbols)
period = st.radio("Candle Period", ['Daily', 'Weekly'])
interval = '1d' if period == 'Daily' else '1wk'

data = yf.download(symbol, period='6mo', interval=interval, auto_adjust=True)

# Flatten multi-level columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    data = data.rename(columns={
        f'Open_{symbol}': 'Open',
        f'High_{symbol}': 'High',
        f'Low_{symbol}': 'Low',
        f'Close_{symbol}': 'Close',
        f'Volume_{symbol}': 'Volume'
    })

# Reset index for plotting
data = data.reset_index()

st.write(data.head())  # For debugging

if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
    st.warning("No price data found or columns missing.")
else:
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(title=f"{symbol} - {period} Candlestick", xaxis_title="Date", yaxis_title="Price")
    # Remove gaps for weekends
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ]
    )

    st.plotly_chart(fig, use_container_width=True)
