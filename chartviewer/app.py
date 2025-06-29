import streamlit as st
import pandas as pd
from data_loader import load_consolidated_data, prepare_symbol_data
from indicators import add_ema, add_sma
from plotting import create_figure

# 頁面設定：寬版
st.set_page_config(page_title="美股篩選結果", layout="wide")

# 讀取篩選結果
results = pd.read_csv('./ScreenResult/screenResults.csv')
symbols = list(results['symbol'])

# 初始化索引
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# 上／下一個 callback
def go_prev():
    if st.session_state.idx > 0:
        st.session_state.idx -= 1

def go_next():
    if st.session_state.idx < len(symbols) - 1:
        st.session_state.idx += 1

# 週期與 Y 軸
col1, col2 = st.columns(2)
with col1:
    period = st.radio("K 線週期", ['日線', '週線'])
with col2:
    yaxis_type = st.radio("Y 軸刻度", ["線性 (linear)", "對數 (log)"], index=0)
is_weekly = (period == '週線')

# 顯示股票清單表格（唯讀）
st.subheader("所有可選股票")
display_df = results[['symbol', 'price', 'rs_rank']].copy()
display_df.columns = ['股票代碼', '現價', 'RS Rank']
st.dataframe(display_df, use_container_width=True, height=200)

# ← 上一個 / 下拉選擇 / 下一個 →
col_prev, col_title, col_next = st.columns([1,2,1])
with col_prev:
    st.button("← 上一個", on_click=go_prev)
with col_title:
    # 讓 selectbox 預設為 idx
    symbol = st.selectbox("當前選股", symbols, index=st.session_state.idx, key="selbox")
    st.session_state.idx = symbols.index(symbol)
with col_next:
    st.button("下一個 →", on_click=go_next)

# 最終選定
symbol = symbols[st.session_state.idx]

# 撈取資料
df_all = load_consolidated_data()
data = prepare_symbol_data(df_all, symbol, resample_weekly=is_weekly)
if data.empty:
    st.warning(f"找不到 {symbol} 的價格數據")
    st.stop()

# 計算指標
EMA_PARAMS = {6:'deepskyblue', 12:'limegreen', 24:'orangered'}
SMA_PARAMS = {50:'lightsteelblue', 150:'rosybrown', 200:'darkseagreen'}
data = add_ema(data, list(EMA_PARAMS.keys()))
data = add_sma(data, list(SMA_PARAMS.keys()))

# 繪圖
fig = create_figure(data, symbol, period, yaxis_type, EMA_PARAMS, SMA_PARAMS)
st.plotly_chart(fig, use_container_width=True)

# 顯示基本資訊
info = results[results['symbol'] == symbol].iloc[0]
st.write(f"**現價：** {info['price']:.2f}    **RS Rank：** {info['rs_rank']:.2f}")
