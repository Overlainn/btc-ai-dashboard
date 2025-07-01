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

# ========== Google Drive ==========
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_INFO = st.secrets["google_service_account"]
creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

MODEL_FILE = "btc_model.pkl"
LAST_TRAIN_FILE = "last_train.txt"
FOLDER_NAME = "StreamlitAI"

def get_folder_id():
    query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    response = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    files = response.get('files', [])
    if files: return files[0]['id']
    meta = {'name': FOLDER_NAME, 'mimeType': 'application/vnd.google-apps.folder'}
    return drive_service.files().create(body=meta, fields='id').execute()['id']

def upload_to_drive(filename):
    folder_id = get_folder_id()
    media = MediaFileUpload(filename, resumable=True)
    meta = {'name': filename, 'parents': [folder_id]}
    q = f"name='{filename}' and '{folder_id}' in parents"
    old = drive_service.files().list(q=q, fields='files(id)').execute().get('files', [])
    if old: drive_service.files().delete(fileId=old[0]['id']).execute()
    drive_service.files().create(body=meta, media_body=media).execute()

def download_from_drive(filename):
    folder_id = get_folder_id()
    q = f"name='{filename}' and '{folder_id}' in parents"
    r = drive_service.files().list(q=q, fields='files(id)').execute()
    files = r.get('files', [])
    if not files: return False
    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    while not downloader.next_chunk()[1]: pass
    with open(filename, 'wb') as f: f.write(fh.getvalue())
    return True

# ========== Push Notifications ==========
push_user_key = st.secrets["pushover"]["user"]
push_app_token = st.secrets["pushover"]["token"]
def send_push(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": push_app_token, "user": push_user_key, "message": msg
    })

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Train Model ==========
def train_model():
    exchange = ccxt.coinbase()
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '30m', limit=1000),
                      columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # Indicators
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], 9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], 21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], 12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], 26)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
    df['WR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
    df['STOCH_RSI'] = ta.momentum.stochrsi(df['Close'])
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_Width'] = bb.bollinger_wband()

    # Binary features
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > ta.volume.volume_weighted_average_price(
        df['High'], df['Low'], df['Close'], df['Volume'])).astype(int)

    df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = df['Return_3'].apply(lambda x: 2 if x > 0.002 else (0 if x < -0.002 else 1))
    df.dropna(inplace=True)

    features = ['EMA9', 'EMA21', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
                'ROC', 'OBV', 'CCI', 'WR', 'STOCH_RSI', 'BB_Width',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Balance classes
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_bal, y_bal)

    with open(MODEL_FILE, 'wb') as f: pickle.dump((model, scaler), f)
    with open(LAST_TRAIN_FILE, 'w') as f: f.write(str(date.today()))
    upload_to_drive(MODEL_FILE)
    upload_to_drive(LAST_TRAIN_FILE)
    return model, scaler

# ========== Load or Train ==========
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

# ========== Streamlit UI ==========
st.set_page_config(layout='wide')
st.title("📈 BTC AI Dashboard + Smart Retraining")
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

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], 9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], 21)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], 12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], 26)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
    df['WR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
    df['STOCH_RSI'] = ta.momentum.stochrsi(df['Close'])
    df['BB_Width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > ta.volume.volume_weighted_average_price(
        df['High'], df['Low'], df['Close'], df['Volume'])).astype(int)
    df.dropna(inplace=True)

    features = ['EMA9', 'EMA21', 'EMA12', 'EMA26', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
                'ROC', 'OBV', 'CCI', 'WR', 'STOCH_RSI', 'BB_Width',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']
    df['Prediction'] = model.predict(scaler.transform(df[features]))
    df[['S0', 'S1', 'S2']] = model.predict_proba(scaler.transform(df[features]))
    return df

if mode == "Live":
    df = get_data()
    price = df['Close'].iloc[-1]
    pred = df['Prediction'].iloc[-1]
    conf = max(df['S0'].iloc[-1], df['S2'].iloc[-1])
    last_sig = st.session_state.get('last_btc_signal')

    if pred in [0, 2] and conf >= 0.6 and pred != last_sig:
        st.session_state['last_btc_signal'] = pred
        name = "📈 LONG" if pred == 2 else "📉 SHORT"
        t = df.index[-1].strftime("%Y-%m-%d %H:%M")
        msg = f"BTC {name} | {t} | ${price:.2f} | S0:{df['S0'].iloc[-1]:.2f}, S2:{df['S2'].iloc[-1]:.2f}"
        send_push(msg)
        pd.DataFrame([{"Timestamp": t, "Price": price, "Signal": name,
                       "Scores": f"{df['S0'].iloc[-1]:.2f},{df['S2'].iloc[-1]:.2f}"}]).to_csv(logfile, mode='a', header=False, index=False)

    st.subheader(f"📊 BTC Live — ${price:.2f}")
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
