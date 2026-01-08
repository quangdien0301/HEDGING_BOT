import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
#import joblib
from datetime import datetime
from xgboost import XGBClassifier

# =====================
# CONFIG
# =====================
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M1
LOT = 0.01

CONFIDENCE = 0.65
MAX_SPREAD = 30
MAGIC = 888999

MODEL_PATH = "xau_ai_model.json"

FEATURES = [
    "return", "ema_fast", "ema_slow",
    "atr", "rsi", "body", "range"
]

# =====================
# MT5 CONNECT
# =====================
def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError("❌ MT5 initialize failed")
    print("✅ MT5 connected")

# =====================
# DATA
# =====================
def get_rates(n=300):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# =====================
# FEATURE ENGINEERING
# =====================
def build_features(df):
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["ema_fast"] = df["close"].ewm(span=10).mean()
    df["ema_slow"] = df["close"].ewm(span=30).mean()

    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    df.dropna(inplace=True)
    return df

# =====================
# AI
# =====================
model = XGBClassifier()
model.load_model("MODEL_PATH")
def ai_predict(df):
    X = df[FEATURES].iloc[-1].values.reshape(1, -1)
    prob = model.predict_proba(X)[0]
    return prob  # [DOWN, UP]

# =====================
# ORDERS
# =====================
def get_positions():
    return mt5.positions_get(symbol=SYMBOL)

def send_order(order_type):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "deviation": 30,
        "magic": MAGIC,
        "comment": "AI_HEDGE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)

def open_hedge():
    print("📥 OPEN HEDGE")
    send_order(mt5.ORDER_TYPE_BUY)
    time.sleep(0.2)
    send_order(mt5.ORDER_TYPE_SELL)

def close_position(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": SYMBOL,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 30,
        "magic": MAGIC,
        "comment": "AI_CLOSE",
    }
    mt5.order_send(request)

# =====================
# AI HEDGE MANAGEMENT
# =====================
def manage_ai(df):
    positions = get_positions()
    if not positions:
        return

    prob_down, prob_up = ai_predict(df)

    print(
        f"{datetime.now()} | AI prob UP={prob_up:.2f} DOWN={prob_down:.2f}"
    )

    for p in positions:
        # BUY nhưng AI nghiêng DOWN → cắt BUY
        if p.type == mt5.ORDER_TYPE_BUY and prob_down > CONFIDENCE:
            print("❌ CLOSE BUY (AI)")
            close_position(p)

        # SELL nhưng AI nghiêng UP → cắt SELL
        elif p.type == mt5.ORDER_TYPE_SELL and prob_up > CONFIDENCE:
            print("❌ CLOSE SELL (AI)")
            close_position(p)

# =====================
# MAIN LOOP
# =====================
def run():
    connect_mt5()
    print("🚀 AI HEDGING BOT STARTED")

    while True:
        spread = mt5.symbol_info(SYMBOL).spread
        if spread > MAX_SPREAD:
            print("⚠️ Spread too high")
            time.sleep(1)
            continue

        df = build_features(get_rates())
        positions = get_positions()

        if not positions:
            open_hedge()
        else:
            manage_ai(df)

        time.sleep(1)

# =====================
if __name__ == "__main__":
    run()

