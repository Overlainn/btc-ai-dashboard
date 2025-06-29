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

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

refresh_interval = 60
if time.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Notification ==========
push_user_key = "u4bs3eqg8gqsv8npdxrcqp8iezf4ad"
push_app_token = "apccb8tmg8j9pupcirg1acfkg3pzaj"

def send_push_notification(message):
    payload = {
        "token": push_app_token,
        "user": push_user_key,
        "message": message
    }
    try:
        requests.post("https://api.pushover.net/1/messages.json", data=payload)
    except Exception as e:
        print("Push notification failed:", e)

# ========== Train Model ==========
def train_live_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=300)
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
    
    # RELAXED LABELING: Thresholds ±0.15% instead of ±0.25%
    df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = df['Return_3'].apply(lambda x: 2 if x > 0.0015 else (0 if x < -0.0015 else 1))

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_live_model()
exchange = ccxt.coinbase()
est = pytz.timezone('US/Eastern')

# ========== Streamlit Setup ==========
st.set_page_config(layout='wide')
st.title("📈 AI Dashboard (BTC, SOL, ETH) – 15m Live Retraining")

bg_color = "#1e1e1e"
text_color = "#f0f0f0"

st.markdown(f"""
    <style>
        .main, .block-container {{
            background-color: {bg_color} !important;
            color: {text_color};
        }}
        .dataframe th, .dataframe td {{
            text-align: center !important;
        }}
    </style>
""", unsafe_allow_html=True)

dash_mode = st.radio("Mode", ("Live", "Backtest"), horizontal=True)

# ========== Log Setup ==========
alert_log_file = os.path.join(os.getcwd(), "btc_alert_log.csv")
if not os.path.exists(alert_log_file):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(alert_log_file, index=False)

last_btc_signal = st.session_state.get("last_btc_signal")

# ========== Data and Chart ==========
def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
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
    X = scaler.transform(df[features])
    df['Prediction'] = model.predict(X)
    proba = model.predict_proba(X)
    df['Score_0'], df['Score_1'], df['Score_2'] = proba[:, 0], proba[:, 1], proba[:, 2]

    # ONLY SHOW SIGNAL IF CONFIDENCE > 60%
    df['Prediction'] = df.apply(lambda row: row['Prediction'] if row[f'Score_{int(row["Prediction"])}'] > 0.6 else 1, axis=1)

    # PRINT DIAGNOSTICS
    print("Prediction counts:", df['Prediction'].value_counts().to_dict())
    print("Max Scores (Last Row):", df[['Score_0', 'Score_1', 'Score_2']].iloc[-1].to_dict())

    return df

def display_chart(symbol, label):
    df = get_data(symbol)
    current_price = df['Close'].iloc[-1]

    if symbol == 'BTC/USDT':
        current_signal = df['Prediction'].iloc[-1]
        if current_signal != last_btc_signal:
            st.session_state.last_btc_signal = current_signal

            # Notify ONLY when signal is LONG or SHORT (not Neutral)
            if current_signal in [0, 2]:
                signal_name = "📈 LONG" if current_signal == 2 else "📉 SHORT"
                s0, s1, s2 = df.iloc[-1][['Score_0', 'Score_1', 'Score_2']]
                timestamp = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                msg = f"BTC Signal Changed: {signal_name}\nTime: {timestamp}\nPrice: ${current_price:.2f}\nScores - Short: {s0:.2f}, Neutral: {s1:.2f}, Long: {s2:.2f}"
                send_push_notification(msg)

                pd.DataFrame([{
                    "Timestamp": timestamp,
                    "Price": current_price,
                    "Signal": signal_name,
                    "Scores": f"{s0:.2f}, {s1:.2f}, {s2:.2f}"
                }]).to_csv(alert_log_file, mode='a', header=False, index=False)

    st.subheader(f"{label} Live Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple')))
    fig.add_trace(go.Scatter(
        x=df[df['Prediction'] == 2].index,
        y=df[df['Prediction'] == 2]['Close'],
        mode='markers', name='📈 Long',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    fig.add_trace(go.Scatter(
        x=df[df['Prediction'] == 0].index,
        y=df[df['Prediction'] == 0]['Close'],
        mode='markers', name='📉 Short',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

    fig.update_layout(height=600, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                      font=dict(color=text_color), xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

# ========== Run Dashboard ==========
if dash_mode == "Live":
    display_chart('BTC/USDT', 'BTC/USDT')
    display_chart('SOL/USDT', 'SOL/USDT')
    display_chart('ETH/USD', 'ETH/USD')
    st.subheader("🔔 BTC Signal Alert Log")
    st.dataframe(pd.read_csv(alert_log_file).tail(10), use_container_width=True)
