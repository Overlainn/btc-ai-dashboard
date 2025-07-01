# app.py
import ccxt, pandas as pd, ta, time, streamlit as st, plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pytz, requests, os, pickle, io
from datetime import datetime, date
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ========== Google Drive Setup ==========
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
    if files:
        return files[0]['id']
    file_metadata = {'name': FOLDER_NAME, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    return folder['id']

def upload_to_drive(filename):
    folder_id = get_folder_id()
    media = MediaFileUpload(filename, resumable=True)
    file_metadata = {'name': filename, 'parents': [folder_id]}
    existing = drive_service.files().list(q=f"name='{filename}' and '{folder_id}' in parents",
                                          fields='files(id)').execute().get('files', [])
    if existing:
        drive_service.files().delete(fileId=existing[0]['id']).execute()
    drive_service.files().create(body=file_metadata, media_body=media).execute()

def download_from_drive(filename):
    folder_id = get_folder_id()
    results = drive_service.files().list(q=f"name='{filename}' and '{folder_id}' in parents",
                                         fields="files(id)").execute()
    files = results.get('files', [])
    if not files: return False
    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done: _, done = downloader.next_chunk()
    with open(filename, 'wb') as f: f.write(fh.getvalue())
    return True

# ========== Push Notifications ==========
push_user_key = st.secrets["pushover"]["user"]
push_app_token = st.secrets["pushover"]["token"]
def send_push_notification(msg):
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": push_app_token, "user": push_user_key, "message": msg
    })

# ========== Auto-refresh ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ========== Train ==========
def train_model():
    exchange = ccxt.coinbase()
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '30m', limit=1000),
                      columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
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

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50).fit(scaler.transform(X), y)

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
st.title("📈 BTC AI Dashboard + Daily Retrain")
mode = st.radio("Mode", ["Live", "Backtest"], horizontal=True)
est = pytz.timezone('US/Eastern')
exchange = ccxt.coinbase()
logfile = "btc_alert_log.csv"
if not os.path.exists(logfile):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(logfile, index=False)

def get_data():
    df = pd.DataFrame(exchange.fetch_ohlcv('BTC/USDT', '30m', limit=1000),
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

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV',
                'EMA12_Cross_26', 'EMA9_Cross_21', 'Above_VWAP']
    df['Prediction'] = model.predict(scaler.transform(df[features]))
    df[['S0', 'S1', 'S2']] = model.predict_proba(scaler.transform(df[features]))
    return df

if mode == "Live":
    df = get_data()
    price = df['Close'].iloc[-1]
    pred = df['Prediction'].iloc[-1]
    conf = max(df['S0'].iloc[-1], df['S2'].iloc[-1])
    t = df.index[-1].strftime("%Y-%m-%d %H:%M")

    # Initialize session state
    if 'open_trade' not in st.session_state:
        st.session_state['open_trade'] = None

    trade = st.session_state['open_trade']

    # Entry logic
    if pred in [0, 2] and conf >= 0.6:
        if not trade:
            signal_name = "LONG" if pred == 2 else "SHORT"
            st.session_state['open_trade'] = {
                "signal": pred,
                "entry_price": price,
                "entry_time": t,
                "entry_conf": conf
            }
            msg = f"BTC 📥 ENTRY — {signal_name} | {t} | ${price:.2f} | Conf: {conf:.2f}"
            send_push_notification(msg)
            pd.DataFrame([{"Timestamp": t, "Price": price, "Signal": f"ENTRY {signal_name}", "Scores": f"{conf:.2f}"}]).to_csv(logfile, mode='a', header=False, index=False)

    # Exit logic
    elif trade:
        reason = None
        if pred != trade["signal"]:
            reason = "Signal flipped"
        elif conf < 0.6:
            reason = "Confidence dropped"

        if reason:
            signal_name = "LONG" if trade["signal"] == 2 else "SHORT"
            msg = f"BTC ❌ EXIT — {signal_name} | {t} | ${price:.2f} | Reason: {reason}"
            send_push_notification(msg)
            pd.DataFrame([{"Timestamp": t, "Price": price, "Signal": f"EXIT {signal_name}", "Scores": reason}]).to_csv(logfile, mode='a', header=False, index=False)
            st.session_state['open_trade'] = None

    # Plot + Display
    st.subheader(f"📊 BTC Live — Current: ${price:.2f}")
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

elif mode == "Backtest":
    df = get_data()
    trades = []
    in_position = None
    entry_time = None
    entry_price = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_position is None:
            if row['Prediction'] == 2 and row['S2'] > 0.6:
                in_position = "LONG"
                entry_time = row.name
                entry_price = row['Close']
            elif row['Prediction'] == 0 and row['S0'] > 0.6:
                in_position = "SHORT"
                entry_time = row.name
                entry_price = row['Close']
        else:
            exit_reason = None
            if in_position == "LONG":
                if row['Prediction'] != 2:
                    exit_reason = "Signal Flip"
                elif row['S2'] < 0.6:
                    exit_reason = "Confidence Drop"
            elif in_position == "SHORT":
                if row['Prediction'] != 0:
                    exit_reason = "Signal Flip"
                elif row['S0'] < 0.6:
                    exit_reason = "Confidence Drop"

            if exit_reason:
                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": row.name,
                    "Direction": in_position,
                    "Entry Price": entry_price,
                    "Exit Price": row['Close'],
                    "PNL": row['Close'] - entry_price if in_position == "LONG" else entry_price - row['Close'],
                    "Reason": exit_reason
                })
                in_position = None

    st.subheader("🧪 Backtest — Signal-Based Trade Log")
    
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades['PNL Color'] = df_trades['PNL'].apply(lambda x: 'color: green' if x > 0 else 'color: red')
        st.dataframe(df_trades.style.apply(lambda x: df_trades['PNL Color'], axis=1, subset=['PNL']))

        # Plot chart with trade markers
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))

        for trade in trades:
            entry_color = 'green' if trade['Direction'] == 'LONG' else 'red'
            exit_color = 'gray'

            fig.add_trace(go.Scatter(
                x=[trade['Entry Time']], y=[trade['Entry Price']],
                mode='markers+text', name='Entry',
                marker=dict(color=entry_color, size=10),
                text=[f"{trade['Direction']} Entry"],
                textposition='top center'))

            fig.add_trace(go.Scatter(
                x=[trade['Exit Time']], y=[trade['Exit Price']],
                mode='markers+text', name='Exit',
                marker=dict(color=exit_color, size=10),
                text=["Exit"],
                textposition='bottom center'))

        fig.update_layout(title="📈 BTC Price with Trade Markers", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Optional performance summary
        st.markdown(f"**Total Trades:** {len(df_trades)}")
        st.markdown(f"**Win Rate:** {100 * (df_trades['PNL'] > 0).mean():.1f}%")
        st.markdown(f"**Total PnL:** {df_trades['PNL'].sum():.2f}")
    else:
        st.info("No trades triggered in backtest.")
