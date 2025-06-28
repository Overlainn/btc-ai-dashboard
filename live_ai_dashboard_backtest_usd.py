
# This is a corrected version with:
# ✅ Stop-loss at 2%
# ✅ Take-profit at 4%
# ✅ Fixed equity curve updates
# ✅ Accurate entry/exit tracking

import ccxt
import pandas as pd
import ta
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout='wide')
st.title("📊 BTC AI Dashboard with SL/TP")

# ===== Parameters =====
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.04  # 4%
BASE_CAPITAL = 100

# ===== Model Training =====
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=5)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    df.dropna(inplace=True)
    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)

    X = df[['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']]
    y = df['Target']

    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=50)
    model.fit(scaler.fit_transform(X), y)

    return model, scaler

# ===== Backtest Runner =====
def run_backtest(df, model, scaler):
    df['Prediction'] = model.predict(scaler.transform(df[['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']]))
    df['Signal'] = df['Prediction']
    trades = []
    capital = BASE_CAPITAL

    for i in range(len(df) - 5):
        if df['Signal'].iloc[i] == 1:
            entry_price = df['Close'].iloc[i]
            for j in range(1, 6):
                future_idx = i + j
                if future_idx >= len(df):
                    break
                low = df['Low'].iloc[future_idx]
                high = df['High'].iloc[future_idx]
                exit_reason = None

                if low <= entry_price * (1 - STOP_LOSS_PCT):
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                    exit_time = df.index[future_idx]
                    exit_reason = 'SL'
                    break
                elif high >= entry_price * (1 + TAKE_PROFIT_PCT):
                    exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    exit_time = df.index[future_idx]
                    exit_reason = 'TP'
                    break
            else:
                exit_price = df['Close'].iloc[i + 5]
                exit_time = df.index[i + 5]
                exit_reason = 'Timed'

            pct_return = (exit_price - entry_price) / entry_price
            capital *= 1 + pct_return

            trades.append({
                'Entry Time': df.index[i],
                'Entry Price': entry_price,
                'Exit Time': exit_time,
                'Exit Price': exit_price,
                'PnL (%)': pct_return * 100,
                'Reason': exit_reason
            })

    equity_curve = pd.DataFrame(trades)
    equity_curve['Equity'] = BASE_CAPITAL * (1 + equity_curve['PnL (%)'] / 100).cumprod()
    return equity_curve

# ===== Main =====
exchange = ccxt.coinbase()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.set_index('Timestamp', inplace=True)

# Add indicators
df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
df['RSI'] = ta.momentum.rsi(df['Close'])
df['MACD'] = ta.trend.macd(df['Close'])
df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
df['ROC'] = ta.momentum.roc(df['Close'], window=5)
df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
df.dropna(inplace=True)

model, scaler = train_model()
equity = run_backtest(df.copy(), model, scaler)

# ===== Display =====
st.subheader("Backtest Equity Curve (SL 2%, TP 4%)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity['Exit Time'], y=equity['Equity'], mode='lines+markers', name='Equity', line=dict(color='green')))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trades")
equity['PnL (%)'] = equity['PnL (%)'].apply(lambda x: f"📈 <span style='color:green'>{x:.2f}%</span>" if float(x) > 0 else f"📉 <span style='color:red'>{x:.2f}%</span>")
st.markdown(equity.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    placeholder = st.empty()
    df = get_data()
    current_price = df['Close'].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

    long_signals = df[df['Prediction'] == 1]
    short_signals = df[df['Prediction'] == 0]

    fig.add_trace(go.Scatter(
        x=long_signals.index, y=long_signals['Close'],
        mode='markers', name='📈 Long Signal',
        marker=dict(size=10, color='green', symbol='triangle-up')
    ))

    fig.add_trace(go.Scatter(
        x=short_signals.index, y=short_signals['Close'],
        mode='markers', name='📉 Short Signal',
        marker=dict(size=10, color='red', symbol='triangle-down')
    ))

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=1.1, showarrow=False,
        text=f"<b>Current BTC Price: ${current_price:.2f}</b>",
        font=dict(size=16, color='white'),
        bgcolor="black",
        borderpad=4
    )

    fig.update_layout(
        title='BTC/USDT Price with AI Predictions',
        xaxis_title='Time',
        yaxis_title='Price',
        height=700,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    placeholder.plotly_chart(fig, use_container_width=True)

