import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import ta  # pip install ta

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "bc48722c038b41bf974a094044f6c99a")  # Your key here
DEFAULT_SYMBOL = "BKSY"
INTERVALS = ["1day", "1week", "1month"]

@st.cache_data(ttl=3600)
def get_stock_data(symbol, interval, limit):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": limit,
        "format": "JSON"
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if "values" not in data:
        st.error("No data returned. Check symbol or API key.")
        st.stop()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(high, low, close, window=14):
    tr = pd.DataFrame({
        'h-l': high - low,
        'h-pc': abs(high - close.shift(1)),
        'l-pc': abs(low - close.shift(1))
    }).max(axis=1)
    return tr.rolling(window).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def generate_signals(df, rsi_thresh, atr_mult, ema_fast, ema_slow, rsi_period, atr_period):
    df['RSI'] = compute_rsi(df['close'], rsi_period)
    df['EMA_fast'] = compute_ema(df['close'], ema_fast)
    df['EMA_slow'] = compute_ema(df['close'], ema_slow)
    df['ATR'] = compute_atr(df['high'], df['low'], df['close'], atr_period)
    df['MACD'], df['MACD_signal'] = compute_macd(df['close'])
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        volume_ok = (df['volume'] > 1.5 * df['volume_ma']).fillna(False)
    else:
        volume_ok = True

    uptrend = (df['close'] > df['EMA_fast']) & (df['EMA_fast'] > df['EMA_slow']) & (df['ADX'] > 25)
    rsi_bullish = (df['RSI'] < rsi_thresh) & (df['RSI'].shift(1) < df['RSI'])
    macd_bullish = df['MACD'] > df['MACD_signal']

    buy_cond = uptrend & rsi_bullish & macd_bullish & volume_ok

    recent_resistance = df['close'].rolling(20).max().shift(1)
    breakout = df['close'] > recent_resistance

    df['buy_signal'] = buy_cond | breakout

    df['trailing_stop'] = df['high'].rolling(5).max() - atr_mult * df['ATR']
    resistance = df['high'].rolling(20).max()
    near_resistance = df['close'] > 0.97 * resistance

    sell_cond = (df['close'] < df['trailing_stop']) | near_resistance | (df['MACD'] < df['MACD_signal'])
    df['sell_signal'] = sell_cond

    return df

def backtest_strategy(df):
    trades = []
    in_position = False
    entry_price = 0
    stop_loss = 0

    for i in range(len(df)):
        if not in_position and df['buy_signal'].iloc[i]:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price * 0.97
            entry_idx = i
            in_position = True

        elif in_position:
            current_price = df['close'].iloc[i]
            if current_price < stop_loss:
                trades.append({
                    'Entry Date': df.index[entry_idx],
                    'Exit Date': df.index[i],
                    'Entry': entry_price,
                    'Exit': stop_loss,
                    'P/L%': (stop_loss - entry_price) / entry_price * 100,
                    'Reason': 'Stop Loss'
                })
                in_position = False
            elif df['sell_signal'].iloc[i]:
                exit_price = df['close'].iloc[i]
                trades.append({
                    'Entry Date': df.index[entry_idx],
                    'Exit Date': df.index[i],
                    'Entry': entry_price,
                    'Exit': exit_price,
                    'P/L%': (exit_price - entry_price) / entry_price * 100,
                    'Reason': 'Exit Signal'
                })
                in_position = False

    return pd.DataFrame(trades)

def evaluate_performance(trades):
    if trades.empty:
        return 0, 0, 0
    win_rate = len(trades[trades['P/L%'] > 0]) / len(trades) * 100
    avg_win = trades[trades['P/L%'] > 0]['P/L%'].mean()
    avg_loss = trades[trades['P/L%'] < 0]['P/L%'].mean()
    return win_rate, avg_win, avg_loss

def optimize_parameters(df):
    best_params = None
    best_score = -np.inf

    rsi_range = range(25, 40, 5)       # 25,30,35
    atr_mult_range = np.arange(1.5, 3.1, 0.5)  # 1.5, 2.0, 2.5, 3.0
    ema_fast_range = range(10, 60, 20) # 10, 30, 50
    ema_slow_range = range(100, 210, 50) # 100, 150, 200

    for rsi_thresh in rsi_range:
        for atr_mult in atr_mult_range:
            for ema_fast in ema_fast_range:
                for ema_slow in ema_slow_range:
                    if ema_fast >= ema_slow:
                        continue
                    df_test = generate_signals(df.copy(), rsi_thresh, atr_mult, ema_fast, ema_slow, rsi_period=14, atr_period=14)
                    trades = backtest_strategy(df_test)
                    win_rate, avg_win, avg_loss = evaluate_performance(trades)
                    # Score by weighted formula (win rate + avg win - avg loss)
                    score = win_rate + (avg_win or 0) - abs(avg_loss or 0)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'rsi_thresh': rsi_thresh,
                            'atr_mult': atr_mult,
                            'ema_fast': ema_fast,
                            'ema_slow': ema_slow
                        }
    return best_params, best_score

def main():
    st.title("🔥 Smart Stock Trader with Optimization 🔥")
    symbol = st.text_input("Stock Symbol", DEFAULT_SYMBOL).upper()
    interval = st.selectbox("Interval", INTERVALS, index=0)
    limit = st.slider("Data Points", 500, 1500, 1000)

    df = get_stock_data(symbol, interval, limit)

    st.info("Running parameter optimization... This might take some time.")
    best_params, best_score = optimize_parameters(df)
    st.success(f"Best Params: RSI Buy: {best_params['rsi_thresh']}, ATR Mult: {best_params['atr_mult']}, "
               f"EMA Fast: {best_params['ema_fast']}, EMA Slow: {best_params['ema_slow']}")
    st.write(f"Optimization Score: {best_score:.2f}")

    df_signals = generate_signals(df, best_params['rsi_thresh'], best_params['atr_mult'],
                                  best_params['ema_fast'], best_params['ema_slow'], rsi_period=14, atr_period=14)
    trades = backtest_strategy(df_signals)

    # Next action
    if df_signals['buy_signal'].iloc[-1]:
        next_action = "BUY"
    elif df_signals['sell_signal'].iloc[-1]:
        next_action = "SELL"
    else:
        next_action = "HOLD"
    st.markdown(f"### Next Action: **{next_action}**")

    # Plotting
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_signals.index, df_signals['close'], label='Close Price')
    ax.plot(df_signals.index, df_signals['EMA_fast'], label=f"EMA {best_params['ema_fast']}")
    ax.plot(df_signals.index, df_signals['EMA_slow'], label=f"EMA {best_params['ema_slow']}")
    buys = df_signals[df_signals['buy_signal']]
    sells = df_signals[df_signals['sell_signal']]
    ax.scatter(buys.index, buys['close'], marker='^', color='green', s=100, label='Buy Signal')
    ax.scatter(sells.index, sells['close'], marker='v', color='red', s=100, label='Sell Signal')
    ax.legend()
    ax.set_title(f"{symbol} Price & Signals")
    st.pyplot(fig)

    # Show trades & metrics
    st.subheader("Trade Log & Metrics")
    if not trades.empty:
        win_rate, avg_win, avg_loss = evaluate_performance(trades)
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.metric("Avg Win", f"{avg_win:.2f}%")
        st.metric("Avg Loss", f"{avg_loss:.2f}%")
        st.dataframe(trades)
    else:
        st.warning("No trades found with this strategy.")

if __name__ == "__main__":
    main()
