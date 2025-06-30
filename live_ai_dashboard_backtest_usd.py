# app.py
import ccxt
import pandas as pd
import ta
import time
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pytz, requests, os, pickle, io
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

# ========== Google Drive Setup ==========
SCOPES = ['https://www.googleapis.com/auth/drive.file']
DRIVE_FOLDER_NAME = 'StreamlitAI'
MODEL_FILE = 'btc_model.pkl'
LAST_TRAIN_FILE = 'last_train.txt'

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
    resp = service.files().list(q=f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'",
                                fields='files(id)').execute()
    if resp['files']:
        return resp['files'][0]['id']
    meta = {'name': DRIVE_FOLDER_NAME, 'mimeType':'application/vnd.google-apps.folder'}
    folder = service.files().create(body=meta, fields='id').execute()
    return folder['id']

def upload_to_drive(local_path, remote_name):
    service = get_drive_service()
    folder_id = ensure_folder(service)
    media = MediaFileUpload(local_path, resumable=True)
    service.files().create(body={'name': remote_name, 'parents':[folder_id]}, media_body=media).execute()

def download_from_drive(remote_name, local_path):
    service = get_drive_service()
    folder_id = ensure_folder(service)
    resp = service.files().list(q=f"'{folder_id}' in parents and name='{remote_name}'",
                                fields='files(id)').execute()
    items = resp.get('files', [])
    if not items: return False
    file_id = items[0]['id']
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, service.files().get_media(fileId=file_id))
    done = False
    while not done: _, done = downloader.next_chunk()
    with open(local_path, 'wb') as f: f.write(fh.getbuffer())
    return True

# ========== Notification ==========
push_key = st.secrets["pushover"]["user"]
push_token = st.secrets["pushover"]["token"]
def send_push(message):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": push_token,
        "user": push_key,
        "message": message
    })

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = 0
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# ========== Training ==========
def train_and_save():
    exchange = ccxt.coinbase()
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT','30m',limit=300),
                      columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    for w in [9,21,12,26]: df[f'EMA{w}'] = ta.trend.ema_indicator(df['Close'], window=w)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df.High,df.Low,df.Close,df.Volume)
    for col in ['RSI','MACD','MACD_Signal','ATR','ROC','OBV']:
        df[col] = getattr(ta, 'momentum' if col in ['RSI','ROC'] else 'trend' if col in ['MACD','MACD_Signal'] else 'volatility' if col=='ATR' else 'volume').__dict__[col.lower()](df['Close'] if col!='OBV' else df['Close'], df['Volume'] if col=='OBV' else None)
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(inplace=True)
    df['Target'] = ((df['Close'].shift(-3)-df['Close'])/df['Close']).apply(lambda x: 2 if x>0.002 else (0 if x<-0.002 else 1))

    features = ['EMA9','EMA21','VWAP','RSI','MACD','MACD_Signal','ATR','ROC','OBV','EMA12_Cross_26','EMA9_Cross_21','Above_VWAP']
    X = df[features]; y = df.Target
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50).fit(scaler.transform(X), y)

    with open(MODEL_FILE,'wb') as f: pickle.dump((model, scaler), f)
    upload_to_drive(MODEL_FILE, MODEL_FILE)

    with open(LAST_TRAIN_FILE,'w') as f: f.write(datetime.utcnow().isoformat())
    upload_to_drive(LAST_TRAIN_FILE, LAST_TRAIN_FILE)

    return model, scaler

# ========== Load or train ==========
model, scaler = None, None
if download_from_drive(MODEL_FILE, MODEL_FILE) and download_from_drive(LAST_TRAIN_FILE, LAST_TRAIN_FILE):
    last = datetime.fromisoformat(open(LAST_TRAIN_FILE).read())
    if datetime.utcnow() - last > timedelta(days=1):
        model, scaler = train_and_save()
    else:
        model, scaler = pickle.load(open(MODEL_FILE,'rb'))
else:
    model, scaler = train_and_save()

# ========== UI & Signal ==========
exchange = ccxt.coinbase()
est = pytz.timezone('US/Eastern')
st.set_page_config(layout='wide')
st.title("📈 AI BTC Dashboard + Daily Retrain")
st.markdown(f"""<style>.main, .block-container {{background:{bg_color}:!important;color:{text_color};}}</style>""", unsafe_allow_html=True)
mode = st.radio("Mode", ["Live","Backtest"], horizontal=True)

logfile = 'btc_alert_log.csv'
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp","Price","Signal","Scores"]).to_csv(logfile,index=False)
last_sig = st.session_state.get("last_btc_signal")

def get_df():
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT','30m',limit=200),
                      columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
    df.set_index('Timestamp', inplace=True)
    for w in [9,21,12,26]: df[f'EMA{w}'] = ta.trend.ema_indicator(df['Close'],window=w)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df.High,df.Low,df.Close,df.Volume)
    for col in ['RSI','MACD','MACD_Signal','ATR','ROC','OBV']:
        df[col] = getattr(ta, 'momentum' if col in ['RSI','ROC'] else 'trend' if col in ['MACD','MACD_Signal'] else 'volatility' if col=='ATR' else 'volume').__dict__[col.lower()](df['Close'] if col!='OBV' else df['Close'], df['Volume'] if col=='OBV' else None)
    df['EMA12_Cross_26'] = (df['EMA12'] > df['EMA26']).astype(int)
    df['EMA9_Cross_21'] = (df['EMA9'] > df['EMA21']).astype(int)
    df['Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
    df.dropna(inplace=True)
    X = scaler.transform(df[[*features]])
    df['Pred'] = model.predict(X)
    df[['S0','S1','S2']] = model.predict_proba(X)
    return df

if mode=="Live":
    df = get_df()
    cp = df.Close.iloc[-1]
    sig = df.Pred.iloc[-1]
    conf = max(df.S0.iloc[-1], df.S2.iloc[-1])
    if sig in [0,2] and conf>=0.6 and sig!=last_sig:
        st.session_state['last_btc_signal'] = sig
        name = "📈 LONG" if sig==2 else "📉 SHORT"
        t = df.index[-1].strftime("%Y-%m-%d %H:%M")
        msg = f"BTC {name} | {t} | ${cp:.2f} | S0:{df.S0.iloc[-1]:.2f}, S2:{df.S2.iloc[-1]:.2f}"
        send_push(msg)
        pd.DataFrame([{"Timestamp":t,"Price":cp,"Signal":name,"Scores":f"{df.S0.iloc[-1]:.2f},{df.S2.iloc[-1]:.2f}"}]).to_csv(logfile,mode='a',header=False,index=False)

    st.subheader(f"📊 BTC Live — Current: ${cp:.2f}")
    fig=go.Figure()
    fig.add_trace(...)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.read_csv(logfile).tail(10))
