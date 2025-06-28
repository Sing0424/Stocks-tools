import math
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def create_figure(
    data, symbol: str, period: str, yaxis_type: str,
    ema_params: dict[int, str], sma_params: dict[int, str]
) -> go.Figure:
    """
    根據資料與參數回傳 Plotly Figure
    """
    # 【關鍵修正】這裡的判斷條件必須與 app.py 中的 radio button 選項文字完全一致
    is_log = (yaxis_type == "對數 (log)")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.03,
        subplot_titles=(f"{symbol} - {period} K 線圖", "成交量")
    )

    # K 線圖
    fig.add_trace(go.Candlestick(
        x=data['Date_str'],
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name='K線', line_width=1
    ), row=1, col=1)

    # EMA
    for span, color in ema_params.items():
        fig.add_trace(go.Scatter(
            x=data['Date_str'], y=data[f'EMA{span}'],
            mode='lines', name=f'EMA{span}',
            line=dict(width=1, color=color)
        ), row=1, col=1)

    # SMA
    for span, color in sma_params.items():
        fig.add_trace(go.Scatter(
            x=data['Date_str'], y=data[f'SMA{span}'],
            mode='lines', name=f'SMA{span}',
            line=dict(width=2, color=color)
        ), row=1, col=1)

    # 成交量
    vol_colors = [
        'rgba(70,128,92,0.4)' if c >= o else 'rgba(135,94,97,0.4)'
        for o, c in zip(data['Open'], data['Close'])
    ]
    fig.add_trace(go.Bar(
        x=data['Date_str'], y=data['Volume'],
        marker_color=vol_colors, name='成交量'
    ), row=2, col=1)

    # X 軸刻度設定 (category → 自動跳過無資料日)
    tick_step = max(len(data) // 10, 1)
    tickvals = data['Date_str'][::tick_step]
    ticktext = tickvals.tolist()
    fig.update_xaxes(
        type='category', tickangle=45,
        tickvals=tickvals, ticktext=ticktext,
        tickfont=dict(size=12), row=2, col=1
    )
    fig.for_each_xaxis(lambda ax: ax.update(type='category'))

    # 價格 Y 軸 (線性或對數)
    if is_log:
        low, high = max(data['Low'].min(), 1e-4), data['High'].max()
        base = 10 ** math.floor(math.log10(high))
        next_tick = next((base * m for m in [1,2,5,10] if base * m >= high), 10 * base)
        log_ticks, v = [], 10 ** math.floor(math.log10(low))
        while v <= next_tick:
            for m in [1,2,5]:
                t = v * m
                if low <= t <= next_tick:
                    log_ticks.append(t)
            v *= 10
        if log_ticks and log_ticks[-1] < next_tick:
            log_ticks.append(next_tick)
        fig.update_yaxes(
            type='log', tickvals=log_ticks,
            ticktext=[str(x) for x in log_ticks],
            showgrid=True, gridcolor='rgba(128,128,128,0.1)',
            row=1, col=1
        )
        fig.add_hline(y=next_tick, line_color='rgba(128,128,128,0.1)', line_width=1, row=1, col=1)
    else:
        fig.update_yaxes(
            type='linear', showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            row=1, col=1
        )

    # 成交量 Y 軸 (固定從0開始且禁止縮放)
    fig.update_yaxes(
        title="成交量", rangemode='tozero', fixedrange=True,showgrid=True, gridcolor='rgba(128,128,128,0.1)',
        row=2, col=1
    )
    fig.update_yaxes(title="價格", row=1, col=1)

    # 版面配置
    fig.update_layout(
        height=700, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False
    )

    return fig
