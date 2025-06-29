# === ALL IMPORTS AND CONFIG ===
import ccxt
import pandas as pd
import ta
import time
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pytz
import requests
import os
from datetime import datetime

st.set_page_config(layout='wide')
st.title("📈 Enhanced AI Dashboard: BTC, SOL, ETH")

bg_color = "#2e2e2e"
text_color = "#ffffff"
est = pytz.timezone('US/Eastern')
refresh_interval = 60

# === STYLE AND REFRESH ===
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = time.time()
    st.rerun()
st.markdown(f"""
    <style>
        .main, .block-container {{
            background-color: {bg_color} !important;
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# === NOTIFICATIONS ===
push_user_key = "u4bs3eqg8gqsv8npdxrcqp8iezf4ad"
push_app_token = "apccb8tmg8j9pupcirg1acfkg3pzaj"
def send_push_notification(message):
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": push_app_token, "user": push_user_key, "message": message
        })
    except Exception as e:
        print("Push notification failed:", e)

# === TRAIN MODEL FUNCTION ===
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # Indicators
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=5)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    # Crossover Features
    df['EMA_Crossover'] = (df['EMA9'] > df['EMA21']).astype(int) - (df['EMA9'] < df['EMA21']).astype(int)
    df['Price_vs_VWAP'] = (df['Close'] > df['VWAP']).astype(int) - (df['Close'] < df['VWAP']).astype(int)

    df.dropna(inplace=True)

    df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = df['Return_3'].apply(lambda x: 2 if x > 0.0025 else (0 if x < -0.0025 else 1))

    feature_cols = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV', 'EMA_Crossover', 'Price_vs_VWAP']
    X = df[feature_cols]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model()
exchange = ccxt.coinbase()
alert_log_file = "btc_alert_log.csv"
if not os.path.exists(alert_log_file):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(alert_log_file, index=False)

# === DATA FETCH FUNCTION ===
def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)

    # Indicators
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=5)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    df['EMA_Crossover'] = (df['EMA9'] > df['EMA21']).astype(int) - (df['EMA9'] < df['EMA21']).astype(int)
    df['Price_vs_VWAP'] = (df['Close'] > df['VWAP']).astype(int) - (df['Close'] < df['VWAP']).astype(int)

    df.dropna(inplace=True)
    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV', 'EMA_Crossover', 'Price_vs_VWAP']
    df['Prediction'] = model.predict(scaler.transform(df[features]))
    probs = model.predict_proba(scaler.transform(df[features]))
    df['Score_0'], df['Score_1'], df['Score_2'] = probs[:, 0], probs[:, 1], probs[:, 2]
    return df

# === LIVE DASHBOARD ===
dash_mode = st.radio("Mode", ("Live", "Backtest"), horizontal=True)
if dash_mode == "Live":
    def display_chart(symbol, label):
        df = get_data(symbol)
        current_price = df['Close'].iloc[-1]

        if symbol == 'BTC/USDT':
            last_signal = st.session_state.get("last_btc_signal")
            new_signal = df['Prediction'].iloc[-1]
            if new_signal != last_signal and new_signal in [0, 2]:
                st.session_state.last_btc_signal = new_signal
                signal_name = "📈 LONG" if new_signal == 2 else "📉 SHORT"
                scores = f"{df['Score_0'].iloc[-1]:.2f}, {df['Score_1'].iloc[-1]:.2f}, {df['Score_2'].iloc[-1]:.2f}"
                msg = f"BTC Signal: {signal_name}\nTime: {df.index[-1]}\nPrice: ${current_price:.2f}\nScores: {scores}"
                send_push_notification(msg)
                pd.DataFrame([{"Timestamp": df.index[-1], "Price": current_price, "Signal": signal_name, "Scores": scores}])\
                  .to_csv(alert_log_file, mode='a', header=False, index=False)

        st.subheader(f"📊 {label} Live Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple')))

        fig.add_trace(go.Scatter(x=df[df['Prediction'] == 2].index, y=df[df['Prediction'] == 2]['Close'],
                                 mode='markers', name='📈 Long', marker=dict(size=10, color='green', symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=df[df['Prediction'] == 0].index, y=df[df['Prediction'] == 0]['Close'],
                                 mode='markers', name='📉 Short', marker=dict(size=10, color='red', symbol='triangle-down')))

        fig.update_layout(title=f'{label} Signals', height=600,
                          plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                          font=dict(color=text_color))
        st.plotly_chart(fig, use_container_width=True)

    display_chart('BTC/USDT', 'BTC')
    display_chart('ETH/USD', 'ETH')
    display_chart('SOL/USDT', 'SOL')
    st.subheader("🔔 BTC Signal Log")
    st.dataframe(pd.read_csv(alert_log_file).tail(10), use_container_width=True)
