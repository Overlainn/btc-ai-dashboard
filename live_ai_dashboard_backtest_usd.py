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

# ========== Model Training ==========
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
    return model, scaler

model, scaler = train_model()
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

# ========== State and Logging ==========
last_btc_signal = st.session_state.get("last_btc_signal")
alert_log_file = "btc_alert_log.csv"
if not os.path.exists(alert_log_file):
    pd.DataFrame(columns=["Timestamp", "Price", "Signal", "Scores"]).to_csv(alert_log_file, index=False)

# ========== Data Fetch & Prediction ==========
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
    probs = model.predict_proba(X)
    df['Score_0'], df['Score_1'], df['Score_2'] = probs[:, 0], probs[:, 1], probs[:, 2]

    return df

# ========== Main Dashboard ==========
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
                score0 = df['Score_0'].iloc[-1]
                score2 = df['Score_2'].iloc[-1]
                timestamp = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")

                message = f"BTC Signal Changed: {signal_name}\nTime: {timestamp}\nPrice: ${current_price:.2f}\nScores - Short: {score0:.2f}, Long: {score2:.2f}"
                send_push_notification(message)

                pd.DataFrame([{
                    "Timestamp": timestamp,
                    "Price": current_price,
                    "Signal": signal_name,
                    "Scores": f"{score0:.2f}, {score2:.2f}"
                }]).to_csv(alert_log_file, mode='a', header=False, index=False)

        st.subheader(f"📊 {label} Live Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

        df_long = df[(df['Prediction'] == 2) & (df['Score_2'] > 0.60)]
        df_short = df[(df['Prediction'] == 0) & (df['Score_0'] > 0.60)]

        fig.add_trace(go.Scatter(
            x=df_long.index, y=df_long['Close'],
            mode='markers', name='📈 Long',
            marker=dict(size=10, color='green', symbol='triangle-up')
        ))
        fig.add_trace(go.Scatter(
            x=df_short.index, y=df_short['Close'],
            mode='markers', name='📉 Short',
            marker=dict(size=10, color='red', symbol='triangle-down')
        ))

        fig.update_layout(
            title=f'{label} AI Signals',
            xaxis_title='Time', yaxis_title='Price',
            height=600,
            plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            font=dict(color=text_color)
        )
        st.plotly_chart(fig, use_container_width=True)

    display_chart('BTC/USDT', 'BTC')
    display_chart('ETH/USD', 'ETH')
    display_chart('SOL/USDT', 'SOL')

    st.subheader("🔔 BTC Signal Alert Log")
    log_df = pd.read_csv(alert_log_file).tail(10)
    st.dataframe(log_df, use_container_width=True)
