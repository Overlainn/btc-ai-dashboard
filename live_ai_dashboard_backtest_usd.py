import ccxt
import pandas as pd
import ta
import time
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== Auto-refresh every 60 seconds ==========
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

refresh_interval = 60  # seconds

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

    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    X = df[['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_dummy_model()
exchange = ccxt.coinbase()

# ========== Streamlit Setup ==========
st.set_page_config(layout='wide')
st.title("📈 Enhanced BTC & SOL AI Dashboard")

# Fixed theme
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

# ========== Mode Toggle ==========
dashboard_mode = st.radio("Mode", ("Live", "Backtest"), horizontal=True)

# ========== Fetch Data ==========

def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=200)
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

    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    df['Prediction'] = model.predict(scaler.transform(df[features]))

    return df

# ========== Backtest ==========
def run_backtest(df):
    df['Future_Close'] = df['Close'].shift(-3)
    df['Return'] = (df['Future_Close'] - df['Close']) / df['Close']
    df['Prediction'] = df['Prediction'].astype(float)
    df['Strategy_Return'] = df['Return'] * df['Prediction']
    df['Strategy_Return'].fillna(0, inplace=True)
    df['Equity'] = 100 * (1 + df['Strategy_Return']).cumprod()

    valid_trades = df['Prediction'].sum()
    win_trades = (df['Strategy_Return'] > 0).sum()
    win_rate = win_trades / valid_trades if valid_trades > 0 else 0
    total_return = df['Equity'].iloc[-1] - 100

    st.metric("📈 Win Rate", f"{win_rate:.2%}")
    st.metric("💰 Total Return", f"{total_return:.2f} USD")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name="Equity Curve", line=dict(color="green")))
    fig.update_layout(title="Backtest Equity Curve", height=500,
                      plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color))
    st.plotly_chart(fig, use_container_width=True)

# ========== App ==========
if dashboard_mode == "Backtest":
    df = get_data('BTC/USDT')
    run_backtest(df)
else:
    # BTC/USDT Live
    df_btc = get_data('BTC/USDT')
    st.subheader("📊 BTC/USDT Live Chart")
    current_price_btc = df_btc['Close'].iloc[-1]

    fig_btc = go.Figure()
    fig_btc.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name='Close', line=dict(color='black')))
    fig_btc.add_trace(go.Scatter(x=df_btc.index, y=df_btc['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig_btc.add_trace(go.Scatter(x=df_btc.index, y=df_btc['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig_btc.add_trace(go.Scatter(x=df_btc.index, y=df_btc['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

    fig_btc.add_trace(go.Scatter(
        x=df_btc[df_btc['Prediction'] == 1].index,
        y=df_btc[df_btc['Prediction'] == 1]['Close'],
        mode='markers', name='📈 Long',
        marker=dict(size=10, color='green', symbol='triangle-up')
    ))
    fig_btc.add_trace(go.Scatter(
        x=df_btc[df_btc['Prediction'] == 0].index,
        y=df_btc[df_btc['Prediction'] == 0]['Close'],
        mode='markers', name='📉 Short',
        marker=dict(size=10, color='red', symbol='triangle-down')
    ))

    fig_btc.add_annotation(
        xref="paper", yref="paper", x=0, y=1.1, showarrow=False,
        text=f"<b>Current BTC Price: ${current_price_btc:.2f}</b>",
        font=dict(size=16, color='white'),
        bgcolor="black"
    )

    fig_btc.update_layout(
        title='BTC/USDT AI Signals',
        xaxis_title='Time',
        yaxis_title='Price',
        height=600,
        plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )

    st.plotly_chart(fig_btc, use_container_width=True)

    # SOL/USDT Live
    df_sol = get_data('SOL/USDT')
    st.subheader("📊 SOL/USDT Live Chart")
    current_price_sol = df_sol['Close'].iloc[-1]

    fig_sol = go.Figure()
    fig_sol.add_trace(go.Scatter(x=df_sol.index, y=df_sol['Close'], name='Close', line=dict(color='black')))
    fig_sol.add_trace(go.Scatter(x=df_sol.index, y=df_sol['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
    fig_sol.add_trace(go.Scatter(x=df_sol.index, y=df_sol['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
    fig_sol.add_trace(go.Scatter(x=df_sol.index, y=df_sol['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

    fig_sol.add_trace(go.Scatter(
        x=df_sol[df_sol['Prediction'] == 1].index,
        y=df_sol[df_sol['Prediction'] == 1]['Close'],
        mode='markers', name='📈 Long',
        marker=dict(size=10, color='green', symbol='triangle-up')
    ))
    fig_sol.add_trace(go.Scatter(
        x=df_sol[df_sol['Prediction'] == 0].index,
        y=df_sol[df_sol['Prediction'] == 0]['Close'],
        mode='markers', name='📉 Short',
        marker=dict(size=10, color='red', symbol='triangle-down')
    ))

    fig_sol.add_annotation(
        xref="paper", yref="paper", x=0, y=1.1, showarrow=False,
        text=f"<b>Current SOL Price: ${current_price_sol:.2f}</b>",
        font=dict(size=16, color='white'),
        bgcolor="black"
    )

    fig_sol.update_layout(
        title='SOL/USDT AI Signals',
        xaxis_title='Time',
        yaxis_title='Price',
        height=600,
        plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )

    st.plotly_chart(fig_sol, use_container_width=True)
