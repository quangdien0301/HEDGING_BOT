import MetaTrader5 as mt5
import pandas as pd
import numpy as np
#import joblib
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================
# CONFIG
# =====================
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKAHEAD = 5          # dự đoán sau 5 nến
DATA_MONTHS = 6        # số tháng dữ liệu
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
        raise RuntimeError("MT5 init failed")
    print("✅ MT5 connected")

# =====================
# LOAD HISTORICAL DATA
# =====================
def load_data():
    to_date = datetime.now()
    from_date = to_date - timedelta(days=30 * DATA_MONTHS)

    rates = mt5.copy_rates_range(
        SYMBOL,
        TIMEFRAME,
        from_date,
        to_date
    )

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
# CREATE LABEL
# =====================
def create_label(df):
    df["future_close"] = df["close"].shift(-LOOKAHEAD)
    df["y"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

# =====================
# TRAIN MODEL
# =====================
def train_model(df):
    X = df[FEATURES]
    y = df["y"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("📊 MODEL PERFORMANCE")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    return model

# =====================
# MAIN
# =====================
def main():
    connect_mt5()
    print("📥 Loading historical data...")
    df = load_data()

    print("⚙️ Building features...")
    df = build_features(df)

    print("🏷️ Creating labels...")
    df = create_label(df)

    print(f"📈 Training samples: {len(df)}")

    model = train_model(df)

    #joblib.dump(model, MODEL_PATH)
    model.save_model(MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    mt5.shutdown()

if __name__ == "__main__":
    main()

