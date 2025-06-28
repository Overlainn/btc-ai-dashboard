import ccxt
import pandas as pd
import ta
import time
import streamlit as st
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pytz

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
est = pytz.timezone('US/Eastern')

# ========== Streamlit Setup ==========
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

dashboard_mode = st.radio("Mode", ("Live", "Backtest"), horizontal=True)

# ========== Fetch Data ==========
def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(est)
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

# ========== Backtest Function ==========
def run_backtest(df, title):
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
    fig.update_layout(title=f"{title} Backtest Equity Curve", height=500,
                      plot_bgcolor=bg_color, paper_bgcolor=bg_color, font=dict(color=text_color))
    st.plotly_chart(fig, use_container_width=True)

    # ===== Trade Log =====
    trades = df[df['Prediction'] == 1][['Close']].copy()
    trades['Entry Time'] = trades.index
    trades['Exit Time'] = trades.index + pd.Timedelta(minutes=90)
    trades['Exit Price'] = df['Close'].shift(-3).loc[trades.index]
    trades['PnL (%)'] = (trades['Exit Price'] - trades['Close']) / trades['Close'] * 100
    trades['Position'] = trades['PnL (%)'].apply(lambda x: 'Long' if x >= 0 else 'Short')
    trades = trades.rename(columns={'Close': 'Entry Price'})
    trades.sort_values(by='Entry Time', ascending=False, inplace=True)
    trades['PnL (%)'] = trades['PnL (%)'].map(lambda x: f"<span style='color: {'green' if x >= 0 else 'red'}'>{x:.2f}%</span>" if pd.notna(x) else 'N/A')

    st.subheader(f"📅 {title} Backtest Trade Log (EST)")
    st.markdown(trades[['Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL (%)', 'Position']].to_html(escape=False, index=False), unsafe_allow_html=True)

# ========== Live or Backtest ==========
if dashboard_mode == "Backtest":
    tabs = st.tabs(["BTC/USDT", "ETH/USD", "SOL/USDT"])
    with tabs[0]:
        df = get_data('BTC/USDT')
        run_backtest(df, "BTC/USDT")
    with tabs[1]:
        df = get_data('ETH/USD')
        run_backtest(df, "ETH/USD")
    with tabs[2]:
        df = get_data('SOL/USDT')
        run_backtest(df, "SOL/USDT")
else:
    def display_chart(symbol, label):
        df = get_data(symbol)
        current_price = df['Close'].iloc[-1]

        st.subheader(f"📊 {label} Live Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], name='EMA9', line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], name='EMA21', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='purple', dash='dot')))

        fig.add_trace(go.Scatter(
            x=df[df['Prediction'] == 1].index,
            y=df[df['Prediction'] == 1]['Close'],
            mode='markers', name='📈 Long',
            marker=dict(size=10, color='green', symbol='triangle-up')
        ))
        fig.add_trace(go.Scatter(
            x=df[df['Prediction'] == 0].index,
            y=df[df['Prediction'] == 0]['Close'],
            mode='markers', name='📉 Short',
            marker=dict(size=10, color='red', symbol='triangle-down')
        ))

        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=1.1, showarrow=False,
            text=f"<b>Current {label} Price: ${current_price:.2f}</b>",
            font=dict(size=16, color='white'),
            bgcolor="black"
        )

        fig.update_layout(
            title=f'{label} AI Signals',
            xaxis_title='Time',
            yaxis_title='Price',
            height=600,
            plot_bgcolor=bg_color, paper_bgcolor=bg_color,
            font=dict(color=text_color)
        )

        st.plotly_chart(fig, use_container_width=True)

    display_chart('BTC/USDT', 'BTC/USDT')
    display_chart('SOL/USDT', 'SOL/USDT')
    display_chart('ETH/USD', 'ETH/USD')
