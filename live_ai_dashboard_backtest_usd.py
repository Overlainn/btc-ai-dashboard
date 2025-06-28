import ccxt
import pandas as pd
import ta
import streamlit as st
import plotly.graph_objs as go
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== Model Training ==========
@st.cache_resource
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
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
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()
exchange = ccxt.coinbase()
st.set_page_config(layout="wide")
st.title("📊 BTC/USDT AI Dashboard")

# Styling
st.markdown("""
<style>
    .main, .block-container { background-color: #1e1e1e !important; color: white; }
</style>
""", unsafe_allow_html=True)

# ========== Load Data ==========
def get_data():
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=5)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df.dropna(inplace=True)

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    df['Prediction'] = model.predict(scaler.transform(df[features]))

    return df

# ========== Backtest ==========
def run_backtest(df):
    entry_times = []
    equity = [100]
    capital = 100
    last_trade_time = df.index[0] - pd.Timedelta(hours=2)
    trades = []

    for i in range(len(df) - 1):
        now = df.index[i]
        if df['Prediction'].iloc[i] == 1 and now - last_trade_time >= pd.Timedelta(hours=2):
            entry_price = df['Close'].iloc[i]
            tp_price = entry_price * 1.04
            sl_price = entry_price * 0.98

            exit_price = None
            exit_time = None

            for j in range(i + 1, min(i + 10, len(df))):  # check next 5 candles
                high = df['High'].iloc[j]
                low = df['Low'].iloc[j]
                exit_index = df.index[j]

                if high >= tp_price:
                    exit_price = tp_price
                    exit_time = exit_index
                    break
                elif low <= sl_price:
                    exit_price = sl_price
                    exit_time = exit_index
                    break

            if exit_price is None:
                exit_price = df['Close'].iloc[min(i + 3, len(df) - 1)]
                exit_time = df.index[min(i + 3, len(df) - 1)]

            pnl = (exit_price - entry_price) / entry_price
            capital *= (1 + pnl)
            equity.append(capital)
            entry_times.append(now)
            trades.append({
                'Entry Time': now,
                'Exit Time': exit_time,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'PnL (%)': pnl * 100,
                'Type': 'Long'
            })
            last_trade_time = now

    equity_df = pd.Series(equity, index=[df.index[0]] + entry_times)
    trades_df = pd.DataFrame(trades)
    return equity_df, trades_df

# ========== Display Tabs ==========
mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)

df = get_data()

if mode == "Live":
    current_price = df['Close'].iloc[-1]
    st.markdown(f"### 💰 Current BTC/USDT: **${current_price:.2f}**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

    signals = df[df['Prediction'] == 1]
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Close'], mode='markers', name='Long', marker=dict(color='green', symbol='triangle-up', size=10)))

    fig.update_layout(height=600, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Backtest":
    equity_curve, trades_df = run_backtest(df)

    st.subheader("📈 Equity Curve with Entries & Exits")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name='Equity Curve', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=trades_df['Entry Time'], y=trades_df['Entry Price'], mode='markers', name='Long Entry', marker=dict(color='green', symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=trades_df['Exit Time'], y=trades_df['Exit Price'], mode='markers', name='Exit', marker=dict(color='white', symbol='x')))

    fig.update_layout(height=600, plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)

    trades_df['PnL (%)'] = trades_df['PnL (%)'].apply(lambda x: f"<span style='color: {'green' if x >= 0 else 'red'}'>{x:.2f}%</span>")
    st.markdown("### 📊 Backtest Trades")
    st.markdown(trades_df[['Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL (%)', 'Type']].to_html(escape=False, index=False), unsafe_allow_html=True)

    st.metric("Total Trades", len(trades_df))
    st.metric("Final Equity", f"${equity_curve.iloc[-1]:.2f}")
