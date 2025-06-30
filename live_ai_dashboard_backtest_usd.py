# app.py
import ccxt, os, io, time, pickle, pytz, requests
import pandas as pd
import ta
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow

# ========== Config ==========
SCOPES = ['https://www.googleapis.com/auth/drive.file']
DRIVE_FOLDER_NAME = 'StreamlitAI'
MODEL_FILE = 'btc_model.pkl'
LAST_TRAIN_FILE = 'last_train.txt'
est = pytz.timezone('US/Eastern')
features = [
    'EMA9','EMA21','VWAP','RSI','MACD','MACD_Signal',
    'ATR','ROC','OBV','EMA12_Cross_26','EMA9_Cross_21','Above_VWAP'
]

# ========== Auth + Drive ==========
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
    q = f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    resp = service.files().list(q=q, fields='files(id)').execute()
    if resp['files']: return resp['files'][0]['id']
    meta = {'name': DRIVE_FOLDER_NAME, 'mimeType':'application/vnd.google-apps.folder'}
    return service.files().create(body=meta, fields='id').execute()['id']

def upload_to_drive(local_path, remote_name):
    service = get_drive_service()
    folder_id = ensure_folder(service)
    media = MediaFileUpload(local_path, resumable=True)
    service.files().create(body={'name': remote_name, 'parents':[folder_id]}, media_body=media).execute()

def download_from_drive(remote_name, local_path):
    service = get_drive_service()
    folder_id = ensure_folder(service)
    q = f"'{folder_id}' in parents and name='{remote_name}'"
    resp = service.files().list(q=q, fields='files(id)').execute()
    if not resp['files']: return False
    file_id = resp['files'][0]['id']
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, service.files().get_media(fileId=file_id))
    done = False
    while not done: _, done = downloader.next_chunk()
    with open(local_path, 'wb') as f: f.write(fh.getbuffer())
    return True

# ========== Pushover Alerts ==========
def send_push(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": st.secrets["pushover"]["token"],
        "user": st.secrets["pushover"]["user"],
        "message": msg
    })

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Train & Save ==========
def train_and_save():
    exchange = ccxt.coinbase()
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '30m', limit=300),
                      columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    for w in [9,21,12,26]:
        df[f'EMA{w}'] = ta.trend.ema_indicator(df['Close'], window=w)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df.High, df.Low, df.Close, df.Volume)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df.High, df.Low, df.Close)
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(inplace=True)

    df['Target'] = ((df['Close'].shift(-3) - df['Close']) / df['Close']).apply(
        lambda x: 2 if x > 0.002 else 0 if x < -0.002 else 1)

    X = df[features]
    y = df['Target']
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50).fit(scaler.transform(X), y)

    with open(MODEL_FILE, 'wb') as f: pickle.dump((model, scaler), f)
    upload_to_drive(MODEL_FILE, MODEL_FILE)

    with open(LAST_TRAIN_FILE, 'w') as f: f.write(datetime.utcnow().isoformat())
    upload_to_drive(LAST_TRAIN_FILE, LAST_TRAIN_FILE)

    return model, scaler

# ========== Load or Train ==========
model, scaler = None, None
if download_from_drive(MODEL_FILE, MODEL_FILE) and download_from_drive(LAST_TRAIN_FILE, LAST_TRAIN_FILE):
    last = datetime.fromisoformat(open(LAST_TRAIN_FILE).read())
    if datetime.utcnow() - last > timedelta(days=1):
        model, scaler = train_and_save()
    else:
        model, scaler = pickle.load(open(MODEL_FILE, 'rb'))
else:
    model, scaler = train_and_save()

# ========== Streamlit UI ==========
exchange = ccxt.coinbase()
st.set_page_config(layout='wide')
st.title("📈 BTC AI Dashboard (Daily Retrain + Push Notifications)")
bg_color = "#2e2e2e"
text_color = "#ffffff"
st.markdown(f"""<style>.main,.block-container{{background:{bg_color};color:{text_color};}}</style>""", unsafe_allow_html=True)
mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)

# ========== Signal Logic ==========
logfile = 'btc_alert_log.csv'
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp","Price","Signal","Scores"]).to_csv(logfile, index=False)
last_sig = st.session_state.get("last_btc_signal")

def get_df():
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '30m', limit=200),
                      columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)
    for w in [9,21,12,26]:
        df[f'EMA{w}'] = ta.trend.ema_indicator(df['Close'], window=w)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df.High, df.Low, df.Close, df.Volume)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df.High, df.Low, df.Close)
    df['ROC'] = ta.momentum.roc(df['Close'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(inplace=True)
    X = scaler.transform(df[features])
    df['Pred'] = model.predict(X)
    df[['S0','S1','S2']] = model.predict_proba(X)
    return df

if mode == "Live":
    df = get_df()
    cp = df['Close'].iloc[-1]
    sig = df['Pred'].iloc[-1]
    conf = max(df['S0'].iloc[-1], df['S2'].iloc[-1])
    if sig in [0,2] and conf >= 0.6 and sig != last_sig:
        st.session_state['last_btc_signal'] = sig
        name = "📈 LONG" if sig==2 else "📉 SHORT"
        t = df.index[-1].strftime("%Y-%m-%d %H:%M")
        msg = f"BTC {name} | {t} | ${cp:.2f} | S0:{df['S0'].iloc[-1]:.2f}, S2:{df['S2'].iloc[-1]:.2f}"
        send_push(msg)
        pd.DataFrame([{
            "Timestamp":t, "Price":cp, "Signal":name,
            "Scores":f"{df['S0'].iloc[-1]:.2f},{df['S2'].iloc[-1]:.2f}"
        }]).to_csv(logfile, mode='a', header=False, index=False)

    st.subheader(f"📊 BTC Live — ${cp:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))
    fig.add_trace(go.Scatter(
        x=df[df['Pred']==2].index, y=df[df['Pred']==2]['Close'],
        mode='markers', name='📈 Long', marker=dict(size=10, color='green', symbol='triangle-up')))
    fig.add_trace(go.Scatter(
        x=df[df['Pred']==0].index, y=df[df['Pred']==0]['Close'],
        mode='markers', name='📉 Short', marker=dict(size=10, color='red', symbol='triangle-down')))
    fig.update_layout(
        xaxis_title='Time', yaxis_title='Price',
        height=600, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color=text_color))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔔 BTC Signal Log")
    st.dataframe(pd.read_csv(logfile).tail(10), use_container_width=True)
