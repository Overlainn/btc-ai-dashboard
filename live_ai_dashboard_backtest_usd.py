# app.py

import ccxt, pandas as pd, ta, time, streamlit as st, plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pytz, requests, os, pickle, io
from datetime import datetime, date
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ========= Constants =========
MODEL_FILE = "btc_model.pkl"
LAST_TRAIN_FILE = "last_train.txt"
FOLDER_NAME = "StreamlitAI"
CANDLE_LIMIT = 300  # reduced for Streamlit Cloud
FEATURES = [
    'EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
    'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP'
]

# ========= Google Drive Setup =========
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]
creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

def get_folder_id():
    q = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    r = drive_service.files().list(q=q, spaces='drive', fields='files(id)').execute()
    files = r.get('files', [])
    if files: return files[0]['id']
    meta = {'name': FOLDER_NAME, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive_service.files().create(body=meta, fields='id').execute()
    return folder['id']

def upload_to_drive(filename):
    folder_id = get_folder_id()
    media = MediaFileUpload(filename, resumable=True)
    query = f"name='{filename}' and '{folder_id}' in parents"
    existing = drive_service.files().list(q=query, fields='files(id)').execute().get('files', [])
    if existing:
        drive_service.files().delete(fileId=existing[0]['id']).execute()
    drive_service.files().create(body={'name': filename, 'parents': [folder_id]}, media_body=media).execute()

def download_from_drive(filename):
    folder_id = get_folder_id()
    q = f"name='{filename}' and '{folder_id}' in parents"
    r = drive_service.files().list(q=q, fields='files(id)').execute()
    files = r.get('files', [])
    if not files: return False
    request = drive_service.files().get_media(fileId=files[0]['id'])
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done: _, done = downloader.next_chunk()
    with open(filename, 'wb') as f: f.write(fh.getvalue())
    return True

# ========= Push Notification =========
PUSH_USER_KEY = st.secrets["pushover"]["user"]
PUSH_APP_TOKEN = st.secrets["pushover"]["token"]
def send_push(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": PUSH_APP_TOKEN, "user": PUSH_USER_KEY, "message": msg
    })

# ========= Auto-refresh =========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========= Train Model =========
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=CANDLE_LIMIT)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df['Target'] = ((df['Close'].shift(-3) - df['Close']) / df['Close']).apply(
        lambda x: 2 if x > 0.002 else (0 if x < -0.002 else 1))
    df.dropna(inplace=True)

    X = df[FEATURES]
    y = df['Target']
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X, y)
    scaler = StandardScaler().fit(X_bal)
    model = RandomForestClassifier(n_estimators=50).fit(scaler.transform(X_bal), y_bal)

    with open(MODEL_FILE, 'wb') as f: pickle.dump((model, scaler), f)
    with open(LAST_TRAIN_FILE, 'w') as f: f.write(str(date.today()))
    upload_to_drive(MODEL_FILE)
    upload_to_drive(LAST_TRAIN_FILE)
    return model, scaler

# ========= Load or Train =========
if os.path.exists(LAST_TRAIN_FILE):
    with open(LAST_TRAIN_FILE) as f:
        if f.read().strip() != str(date.today()):
            model, scaler = train_model()
        else:
            with open(MODEL_FILE, 'rb') as f: model, scaler = pickle.load(f)
else:
    if download_from_drive(MODEL_FILE) and download_from_drive(LAST_TRAIN_FILE):
        with open(LAST_TRAIN_FILE) as f:
            if f.read().strip() != str(date.today()):
                model, scaler = train_model()
            else:
                with open(MODEL_FILE, 'rb') as f: model, scaler = pickle.load(f)
    else:
        model, scaler = train_model()

# ========= Streamlit UI =========
st.set_page_config(layout='wide')
st.title("📈 BTC AI Dashboard — Streamlit Cloud Optimized")
mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
est = pytz.timezone('US/Eastern')
exchange = ccxt.coinbase()

logfile = "btc_alert_log.csv"
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(logfile, index=False)

def get_data():
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '30m', limit=200),
                      columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(inplace=True)

    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    df['Prediction'] = model.predict(X_scaled)
    df[['S0', 'S1', 'S2']] = model.predict_proba(X_scaled)
    return df

if mode == "Live":
    df = get_data()
    cp = df['Close'].iloc[-1]
    pred = df['Prediction'].iloc[-1]
    conf = max(df['S0'].iloc[-1], df['S2'].iloc[-1])
    last_sig = st.session_state.get("last_btc_signal")

    if pred in [0, 2] and conf >= 0.6 and pred != last_sig:
        st.session_state['last_btc_signal'] = pred
        name = "📈 LONG" if pred == 2 else "📉 SHORT"
        msg = f"{name} | ${cp:.2f} | S0={df['S0'].iloc[-1]:.2f}, S2={df['S2'].iloc[-1]:.2f}"
        send_push(msg)
        pd.DataFrame([{
            "Timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M"),
            "Price": cp,
            "Signal": name,
            "Scores": f"{df['S0'].iloc[-1]:.2f}, {df['S2'].iloc[-1]:.2f}"
        }]).to_csv(logfile, mode='a', header=False, index=False)

    st.subheader(f"📊 BTC Live — Price: ${cp:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name="EMA9"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name="EMA21"))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name="VWAP"))
    df_long = df[(df['Prediction'] == 2) & (df['S2'] > 0.6)]
    df_short = df[(df['Prediction'] == 0) & (df['S0'] > 0.6)]
    fig.add_trace(go.Scatter(x=df_long.index, y=df_long['Close'], mode='markers', name='📈 Long', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=df_short.index, y=df_short['Close'], mode='markers', name='📉 Short', marker=dict(color='red')))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.read_csv(logfile).tail(10))
