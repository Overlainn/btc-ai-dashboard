# app.py
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
import pickle
from datetime import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload

# ========== Google Drive Model Persistence ==========
SCOPES = ['https://www.googleapis.com/auth/drive.file']
MODEL_FILENAME = "btc_model.pkl"
SCALER_FILENAME = "btc_scaler.pkl"

def get_drive_service():
    flow = InstalledAppFlow.from_client_config({
        "installed": {
            "client_id": st.secrets["google"]["client_id"],
            "client_secret": st.secrets["google"]["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"]
        }
    }, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(filename):
    service = get_drive_service()
    file_metadata = {'name': filename}
    media = MediaFileUpload(filename, resumable=True)
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def download_file_if_exists(filename):
    if os.path.exists(filename):
        return
    service = get_drive_service()
    results = service.files().list(q=f"name='{filename}'", pageSize=1).execute()
    items = results.get('files', [])
    if items:
        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        with open(filename, 'wb') as f:
            downloader = MediaFileUpload(filename, resumable=True)
            downloader = downloader._fd = f
            request.execute()

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Notification ==========
push_user_key = "u4bs3eqg8gqsv8npdxrcqp8iezf4ad"
push_app_token = "apccb8tmg8j9pupcirg1acfkg3pzaj"
def send_push_notification(message):
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": push_app_token,
            "user": push_user_key,
            "message": message
        })
    except Exception as e:
        print("Notification Error:", e)

# ========== Train or Load ==========
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=5)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Price_Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(inplace=True)

    df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = df['Return_3'].apply(lambda x: 2 if x > 0.002 else (0 if x < -0.002 else 1))

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Price_Above_VWAP']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    with open(MODEL_FILENAME, "wb") as f: pickle.dump(model, f)
    with open(SCALER_FILENAME, "wb") as f: pickle.dump(scaler, f)
    upload_file_to_drive(MODEL_FILENAME)
    upload_file_to_drive(SCALER_FILENAME)
    return model, scaler

# ========== Load from Drive or Train ==========
if os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME):
    with open(MODEL_FILENAME, "rb") as f: model = pickle.load(f)
    with open(SCALER_FILENAME, "rb") as f: scaler = pickle.load(f)
else:
    download_file_if_exists(MODEL_FILENAME)
    download_file_if_exists(SCALER_FILENAME)
    if os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME):
        with open(MODEL_FILENAME, "rb") as f: model = pickle.load(f)
        with open(SCALER_FILENAME, "rb") as f: scaler = pickle.load(f)
    else:
        model, scaler = train_model()

exchange = ccxt.coinbase()
est = pytz.timezone('US/Eastern')
alert_log_file = "btc_alert_log.csv"
if not os.path.exists(alert_log_file):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(alert_log_file, index=False)

# ========== Streamlit UI ==========
st.set_page_config(layout='wide')
st.title("📈 Enhanced AI Dashboard: BTC, SOL, ETH")

# Styling
st.markdown("""
    <style>
        .main, .block-container {
            background-color: #2e2e2e !important;
            color: #ffffff;
        }
        .dataframe th, .dataframe td {
            text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)

dash_mode = st.radio("Mode", ("Live", "Backtest"), horizontal=True)
last_btc_signal = st.session_state.get("last_btc_signal")

# ========== Live Mode ==========
def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'], window=5)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Price_Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

    df.dropna(inplace=True)
    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Price_Above_VWAP']
    X = scaler.transform(df[features])
    df['Prediction'] = model.predict(X)
    probs = model.predict_proba(X)
    df['Score_0'], df['Score_1'], df['Score_2'] = probs[:, 0], probs[:, 1], probs[:, 2]
    return df

if dash_mode == "Live":
    def display_chart(symbol, label):
        df = get_data(symbol)
        current_price = df['Close'].iloc[-1]
        if symbol == 'BTC/USDT':
            current_signal = df['Prediction'].iloc[-1]
            confidence = max(df['Score_0'].iloc[-1], df['Score_2'].iloc[-1])
            if current_signal in [0, 2] and confidence >= 0.60 and current_signal != last_btc_signal:
                st.session_state.last_btc_signal = current_signal
                signal_name = "📈 LONG" if current_signal == 2 else "📉 SHORT"
                message = f"BTC Signal Changed: {signal_name}\nTime: {df.index[-1]}\nPrice: ${current_price:.2f}\nScores - Short: {df['Score_0'].iloc[-1]:.2f}, Long: {df['Score_2'].iloc[-1]:.2f}"
                send_push_notification(message)
                pd.DataFrame([{
                    "Timestamp": df.index[-1],
                    "Price": current_price,
                    "Signal": signal_name,
                    "Scores": f"{df['Score_0'].iloc[-1]:.2f}, {df['Score_2'].iloc[-1]:.2f}"
                }]).to_csv(alert_log_file, mode='a', header=False, index=False)

        st.subheader(f"📊 {label} Live Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))
        df_long = df[(df['Prediction'] == 2) & (df['Score_2'] > 0.60)]
        df_short = df[(df['Prediction'] == 0) & (df['Score_0'] > 0.60)]
        fig.add_trace(go.Scatter(x=df_long.index, y=df_long['Close'], mode='markers', name='📈 Long',
                                 marker=dict(size=10, color='green', symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=df_short.index, y=df_short['Close'], mode='markers', name='📉 Short',
                                 marker=dict(size=10, color='red', symbol='triangle-down')))
        fig.update_layout(title=f'{label} AI Signals', xaxis_title='Time', yaxis_title='Price',
                          height=600, plot_bgcolor='#2e2e2e', paper_bgcolor='#2e2e2e',
                          font=dict(color='#ffffff'))
        st.plotly_chart(fig, use_container_width=True)

    display_chart('BTC/USDT', 'BTC')
    display_chart('ETH/USD', 'ETH')
    display_chart('SOL/USDT', 'SOL')
    st.subheader("🔔 BTC Signal Alert Log")
    st.dataframe(pd.read_csv(alert_log_file).tail(10), use_container_width=True)
