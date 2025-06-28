import ccxt
import pandas as pd
import ta
import time
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Train AI model
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=500)
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaler.fit_transform(X), y)
    return model, scaler

model, scaler = train_model()
exchange = ccxt.coinbase()

st.set_page_config(layout="wide")
st.title("💹 BTC/USDT AI Dashboard")

bg_color = "#1e1e1e"
text_color = "#ffffff"
st.markdown(f"""
    <style>
        .main, .block-container {{
            background-color: {bg_color} !important;
            color: {text_color};
        }}
        .dataframe th, .dataframe td {{
            text-align: center;
        }}
    </style>
""", unsafe_allow_html=True)

mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)

# === Fetch & Predict ===
def fetch_data():
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

# === Backtest Logic ===
def backtest(df):
    trades = []
    last_entry = None
    equity = 100000
    equity_curve = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        if row['Prediction'] == 1:
            now = row.name
            if last_entry is None or (now - last_entry).total_seconds() >= 7200:
                entry_price = row['Close']
                for j in range(i+1, len(df)):
                    future = df.iloc[j]
                    exit_price = future['Close']
                    change = (exit_price - entry_price) / entry_price

                    if change >= 0.04 or change <= -0.02 or j == len(df) - 1:
                        pnl = (exit_price - entry_price) / entry_price * 100
                        trades.append({
                            "Entry Time": row.name,
                            "Exit Time": future.name,
                            "Entry Price": entry_price,
                            "Exit Price": exit_price,
                            "PnL (%)": pnl,
                            "Position": "Long"
                        })
                        last_entry = now
                        break

        equity_curve.append(equity)
        if trades:
            equity = 100000
            for t in trades:
                equity *= (1 + t['PnL (%)'] / 100)

    trade_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, index=df.iloc[:len(equity_curve)].index)

    win_rate = (trade_df["PnL (%)"] > 0).mean() if not trade_df.empty else 0
    total_return = equity_series.iloc[-1] - 100000

    st.metric("📈 Win Rate", f"{win_rate:.2%}")
    st.metric("💰 Total Return", f"{total_return:.2f} USD")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series, name="Equity Curve", line=dict(color="green")))

    if not trade_df.empty:
        fig.add_trace(go.Scatter(
            x=trade_df["Entry Time"], y=trade_df["Entry Price"],
            mode='markers', name="Long Entry", marker=dict(color='green', symbol='triangle-up', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=trade_df["Exit Time"], y=trade_df["Exit Price"],
            mode='markers', name="Exit", marker=dict(color='white', symbol='x', size=9)
        ))

    fig.update_layout(
        title="📉 Equity Curve with Entries & Exits",
        plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    if not trade_df.empty:
        trade_df["PnL (%)"] = trade_df["PnL (%)"].map(lambda x: f"<span style='color:{'green' if x > 0 else 'red'}'>{x:.2f}%</span>")
        st.markdown(trade_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# === Live Mode ===
def live_chart(df):
    current_price = df['Close'].iloc[-1]
    st.markdown(f"### 🪙 Current BTC/USDT: **${current_price:,.2f}**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

    longs = df[df['Prediction'] == 1]
    shorts = df[df['Prediction'] == 0]

    fig.add_trace(go.Scatter(x=longs.index, y=longs['Close'],
        mode='markers', name='Long', marker=dict(color='lime', symbol='triangle-up', size=8)))
    fig.add_trace(go.Scatter(x=shorts.index, y=shorts['Close'],
        mode='markers', name='Short', marker=dict(color='red', symbol='triangle-down', size=8)))

    fig.update_layout(
        title='Live BTC/USDT with Indicators',
        plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

# Run app
df = fetch_data()
if mode == "Live":
    live_chart(df)
else:
    backtest(df)
