
import ccxt
import pandas as pd
import ta
import time
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== Load and Train a Simple Model ==========
def train_dummy_model():
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])

    df.dropna(inplace=True)

    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    X = df[['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal']]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_dummy_model()
exchange = ccxt.coinbase()

# ========== Streamlit App ==========
st.set_page_config(layout='wide')
st.title("🔮 Live BTC/USDT AI Dashboard")

# Theme toggle
mode = st.radio("Select Theme:", ("Light", "Dark"), horizontal=True)

# CSS injection
if mode == "Light":
    bg_color = "#ffffff"
    text_color = "#000000"
else:
    bg_color = "#0e1117"
    text_color = "#ffffff"

st.markdown(
    f"""
    <style>
        .main, .block-container {{
            background-color: {bg_color} !important;
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Backtest toggle
dashboard_mode = st.radio("Mode", ("Live", "Backtest"), horizontal=True)

def get_data():
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])

    df.dropna(inplace=True)

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal']
    df['Prediction'] = model.predict(scaler.transform(df[features]))

    return df

def run_backtest(df):
    df['Future_Close'] = df['Close'].shift(-3)
    df['Return'] = (df['Future_Close'] - df['Close']) / df['Close']
    df['Strategy_Return'] = df['Return'] * df['Prediction']
    df['Equity'] = 100 * (1 + df['Strategy_Return']).cumprod()

    win_rate = (df['Strategy_Return'] > 0).sum() / df['Prediction'].sum() if df['Prediction'].sum() > 0 else 0
    total_return = df['Equity'].iloc[-1] - 100

    st.metric("📈 Win Rate", f"{win_rate:.2%}")
    st.metric("💰 Total Return", f"{total_return:.2%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name="Equity Curve", line=dict(color="green")))
    fig.update_layout(title="Backtest Equity Curve", height=500,
                      plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color))
    st.plotly_chart(fig, use_container_width=True)

if dashboard_mode == "Backtest":
    df = get_data()
    run_backtest(df)
else:
    placeholder = st.empty()
    while True:
        df = get_data()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

        long_signals = df[df['Prediction'] == 1]
        short_signals = df[df['Prediction'] == 0]

        fig.add_trace(go.Scatter(
            x=long_signals.index, y=long_signals['Close'],
            mode='markers', name='📈 Long Signal',
            marker=dict(size=10, color='green', symbol='triangle-up')
        ))

        fig.add_trace(go.Scatter(
            x=short_signals.index, y=short_signals['Close'],
            mode='markers', name='📉 Short Signal',
            marker=dict(size=10, color='red', symbol='triangle-down')
        ))

        fig.update_layout(
            title='BTC/USDT Price with AI Predictions',
            xaxis_title='Time',
            yaxis_title='Price',
            height=600,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color)
        )

        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(60)
