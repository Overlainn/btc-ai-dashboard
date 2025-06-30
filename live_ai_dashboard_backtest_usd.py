import ccxt, pandas as pd, ta, time, streamlit as st, plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pytz, requests, os, pickle, io
from datetime import datetime, date
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

# ========== Google Drive Setup ==========
SCOPES = ['https://www.googleapis.com/auth/drive.file']
MODEL_FILE = "btc_model.pkl"
LAST_TRAIN_FILE = "last_train.txt"
DRIVE_FOLDER = "StreamlitAI"

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

def ensure_folder(service):
    results = service.files().list(q=f"name='{DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder'",
                                   fields='files(id)').execute()
    files = results.get('files', [])
    if files:
        return files[0]['id']
    folder_metadata = {'name': DRIVE_FOLDER, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder['id']

def upload_to_drive(local_path, remote_name):
    service = get_drive_service()
    folder_id = ensure_folder(service)
    media = MediaFileUpload(local_path, resumable=True)
    service.files().create(body={'name': remote_name, 'parents':[folder_id]}, media_body=media).execute()

def download_from_drive(remote_name, local_path):
    service = get_drive_service()
    folder_id = ensure_folder(service)
    results = service.files().list(q=f"'{folder_id}' in parents and name='{remote_name}'", fields="files(id)").execute()
    files = results.get('files', [])
    if not files:
        return False
    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(local_path, "wb") as f:
        f.write(fh.getbuffer())
    return True

# ========== Notification ==========
push_user_key = st.secrets["pushover"]["user"]
push_app_token = st.secrets["pushover"]["token"]
def send_push_notification(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": push_app_token, "user": push_user_key, "message": msg
    })

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Training ==========
def train_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # Indicators
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

    # Binary crossovers
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Price_Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

    df.dropna(inplace=True)

    df['Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = df['Return_3'].apply(lambda x: 2 if x > 0.002 else (0 if x < -0.002 else 1))

    features = [
        'EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal',
        'ATR', 'ROC', 'OBV', 'EMA12_Cross_26', 'EMA9_Cross_21', 'Price_Above_VWAP'
    ]

    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    with open(MODEL_FILE, "wb") as f: pickle.dump((model, scaler), f)
    with open(LAST_TRAIN_FILE, "w") as f: f.write(str(date.today()))

    upload_to_drive(MODEL_FILE, MODEL_FILE)
    upload_to_drive(LAST_TRAIN_FILE, LAST_TRAIN_FILE)
    return model, scaler

# ========== Load or Train ==========
model, scaler = None, None
if download_from_drive(MODEL_FILE, MODEL_FILE) and download_from_drive(LAST_TRAIN_FILE, LAST_TRAIN_FILE):
    last_train = open(LAST_TRAIN_FILE).read().strip()
    if last_train != str(date.today()):
        model, scaler = train_model()
    else:
        model, scaler = pickle.load(open(MODEL_FILE, "rb"))
else:
    model, scaler = train_model()

# ========== Streamlit UI ==========
exchange = ccxt.coinbase()
est = pytz.timezone('US/Eastern')
st.set_page_config(layout="wide")
st.title("📈 Enhanced AI Dashboard: BTC, ETH, SOL")
bg_color = "#1e1e1e"; text_color = "#ffffff"
st.markdown(f"""
    <style>
        .main, .block-container {{ background-color: {bg_color}; color: {text_color}; }}
        .dataframe th, .dataframe td {{ text-align: center; }}
    </style>
""", unsafe_allow_html=True)

logfile = "btc_alert_log.csv"
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(logfile, index=False)

dash_mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
last_btc_signal = st.session_state.get("last_btc_signal")

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

    features = [
        'EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal',
        'ATR', 'ROC', 'OBV', 'EMA12_Cross_26', 'EMA9_Cross_21', 'Price_Above_VWAP'
    ]
    X = scaler.transform(df[features])
    df['Prediction'] = model.predict(X)
    df[['Score_0', 'Score_1', 'Score_2']] = model.predict_proba(X)
    return df

def display_chart(symbol, label):
    df = get_data(symbol)
    cp = df['Close'].iloc[-1]

    if symbol == 'BTC/USDT':
        sig = df['Prediction'].iloc[-1]
        conf = max(df['Score_0'].iloc[-1], df['Score_2'].iloc[-1])
        if sig in [0, 2] and conf >= 0.6 and sig != last_btc_signal:
            st.session_state['last_btc_signal'] = sig
            name = "📈 LONG" if sig == 2 else "📉 SHORT"
            msg = f"{label} Signal: {name}\nPrice: ${cp:.2f}\nScores: S0={df['Score_0'].iloc[-1]:.2f}, S2={df['Score_2'].iloc[-1]:.2f}"
            send_push_notification(msg)
            pd.DataFrame([{
                "Timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M"),
                "Price": cp,
                "Signal": name,
                "Scores": f"{df['Score_0'].iloc[-1]:.2f}, {df['Score_2'].iloc[-1]:.2f}"
            }]).to_csv(logfile, mode='a', header=False, index=False)

    st.subheader(f"📊 {label} Live")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

    df_long = df[(df['Prediction'] == 2) & (df['Score_2'] > 0.6)]
    df_short = df[(df['Prediction'] == 0) & (df['Score_0'] > 0.6)]

    fig.add_trace(go.Scatter(x=df_long.index, y=df_long['Close'], mode='markers', name='📈 Long',
                             marker=dict(size=10, color='green', symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=df_short.index, y=df_short['Close'], mode='markers', name='📉 Short',
                             marker=dict(size=10, color='red', symbol='triangle-down')))
    fig.update_layout(title=f'{label} AI Signals', height=600, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                      font=dict(color=text_color))
    st.plotly_chart(fig, use_container_width=True)

if dash_mode == "Live":
    display_chart("BTC/USDT", "BTC")
    display_chart("ETH/USD", "ETH")
    display_chart("SOL/USDT", "SOL")
    st.subheader("🔔 BTC Alert Log")
    st.dataframe(pd.read_csv(logfile).tail(10), use_container_width=True)
