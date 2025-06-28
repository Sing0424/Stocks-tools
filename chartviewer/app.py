import streamlit as st
import pandas as pd
from data_loader import load_consolidated_data, prepare_symbol_data
from indicators import add_ema, add_sma
from plotting import create_figure

st.set_page_config(
    page_title="美股篩選結果",
    layout="wide"
)

# 參數設定
EMA_PARAMS = {6: 'deepskyblue', 12: 'limegreen', 24: 'orangered'}
SMA_PARAMS = {50: 'lightsteelblue', 150: 'rosybrown', 200: 'darkseagreen'}

def main():
    # 讀取篩選結果
    results = pd.read_csv('./ScreenResult/screenResults.csv')
    
    st.title("美股篩選結果 K 線圖")
    
    # 週期和 Y 軸選擇放在上方
    col1, col2 = st.columns(2)
    with col1:
        period = st.radio("K 線週期", ['日線', '週線'])
    with col2:
        yaxis_type = st.radio("Y 軸刻度", ["線性 (linear)", "對數 (log)"], index=0)
    
    is_weekly = (period == '週線')
    
    # 股票選擇表格
    st.subheader("選擇股票")
    
    # 準備顯示用的表格資料
    display_df = results[['symbol', 'price', 'rs_rank']].copy()
    display_df.columns = ['股票代碼', '現價', 'RS Rank']
    
    # 使用 st.dataframe 的選擇功能
    event = st.dataframe(
    display_df,
    use_container_width=True,
    height=200,             # ← 限制表格高度
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "股票代碼": st.column_config.TextColumn("股票代碼", width="medium"),
        "現價":    st.column_config.NumberColumn("現價", format="%.2f"),
        "RS Rank": st.column_config.NumberColumn("RS Rank", format="%.2f")
    }
)
    
    # 檢查是否有選擇
    if event.selection.rows:
        selected_idx = event.selection.rows[0]  # 取第一個選中的行
        symbol = results.iloc[selected_idx]['symbol']
        
        st.success(f"已選擇: {symbol}")
        
        # 讀取與準備資料
        df_all = load_consolidated_data()
        data = prepare_symbol_data(df_all, symbol, resample_weekly=is_weekly)
        
        if data.empty:
            st.warning(f"找不到 {symbol} 的價格數據")
            return
        
        # 加入技術指標
        data = add_ema(data, list(EMA_PARAMS.keys()))
        data = add_sma(data, list(SMA_PARAMS.keys()))
        
        # 繪圖
        fig = create_figure(data, symbol, period, yaxis_type, EMA_PARAMS, SMA_PARAMS)
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示詳細資訊
        info = results[results['symbol'] == symbol].iloc[0]
        st.info(f"**現價：** {info['price']:.2f}    **RS Rank：** {info['rs_rank']:.2f}")
    
    else:
        st.info("👆 請點擊表格中的任一行選擇股票")

if __name__ == '__main__':
    main()
