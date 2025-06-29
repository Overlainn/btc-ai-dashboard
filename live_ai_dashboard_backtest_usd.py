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

# ========== Push Alert Function ==========
PUSHOVER_USER_KEY = "u4bs3eqg8gqsv8npdxrcqp8iezf4ad"
PUSHOVER_APP_TOKEN = "apccb8tmg8j9pupcirg1acfkg3pzaj"

def send_pushover_alert(signal, price, timestamp):
    signal_str = "📉 SHORT" if signal == 0 else "📈 LONG"
    msg = f"{signal_str} signal for BTC\n💰 Price: ${price:.2f}\n🕒 Time: {timestamp}"
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": msg,
            "title": "BTC Signal Alert"
        })
    except Exception as e:
        print(f"❌ Failed to send alert: {e}")

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'btc_last_signal' not in st.session_state:
    st.session_state.btc_last_signal = None

refresh_interval = 60
if time.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Load and Train Model ==========
def train_dummy_model():
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

    df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = df['Return_3'].apply(lambda x: 2 if x > 0.0025 else (0 if x < -0.0025 else 1))

    X = df[['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_dummy_model()
exchange = ccxt.coinbase()
est = pytz.timezone('US/Eastern')

# ========== Streamlit UI ==========
st.set_page_config(layout='wide')
st.title("📈 Enhanced AI Dashboard: BTC, SOL, ETH")

bg_color = "#2e2e2e"
text_color = "#ffffff"

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

# ========== Fetch Data ==========
def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=200)
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
    df['Prediction'] = model.predict(scaler.transform(df[features]))

    # ========== Send Alert if BTC Signal Changed ==========
    if symbol == 'BTC/USDT':
        last_signal = st.session_state.btc_last_signal
        current_signal = df['Prediction'].iloc[-1]
        if current_signal in [0, 2] and current_signal != last_signal:
            current_price = df['Close'].iloc[-1]
            timestamp = df.index[-1].strftime("%Y-%m-%d %H:%M %Z")
            send_pushover_alert(current_signal, current_price, timestamp)
            st.session_state.btc_last_signal = current_signal

    return df

# The rest of your script can follow unchanged — tabs, charts, etc.
