import sys
import json
import joblib
import pandas as pd
import numpy as np
from math import log
from config.constants import (
    SYMBOL_NAME, DEFAULT_LOTS,
    REQUIRED_INPUT_FIELDS, OPTIONAL_INPUT_FIELDS, FEATURE_COLUMNS, TRADE_SUMMARY_FIELDS
)
from config.paths import DATA_PATH, MODEL_PATH

from datetime import datetime


# === Load Trained Pipeline ===
def load_pipeline():
    return joblib.load(MODEL_PATH)


# === Validate Required Fields ===
def validate_input(data: dict):
    missing = [f for f in REQUIRED_INPUT_FIELDS if f not in data or str(data[f]).strip() == ""]
    if missing:
        raise ValueError(f"❌ Missing required fields: {missing}")


# === Apply Fallbacks for Optional Fields ===
def apply_defaults(data: dict, base_df: pd.DataFrame) -> tuple:
    fallback_defaults = {
        "Strategy Description": "Default strategy",
        "Trading Session": "London Session",
        "Live Sentiment": "Neutral",
        "Stoploss": 0.0,
        "Volume": 10000,
        "Volatility": 1.0,
        "Target Price": data.get("Entry Price", 0) * 1.01 if "Entry Price" in data else 0.0
    }

    target_provided = True
    for field in OPTIONAL_INPUT_FIELDS:
        if field not in data or str(data[field]).strip() == "":
            data[field] = fallback_defaults.get(field, None)
            if field == "Target Price":
                target_provided = False  # fallback was applied for Target Price

    return data, target_provided



# === Feature Engineering (Match training) ===
import pandas_ta as ta

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["Entry Timestamp"] = pd.to_datetime(df["Entry Timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["Target Timestamp"] = pd.to_datetime(df["Target Timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["EntryPrice"] = pd.to_numeric(df["Entry Price"], errors="coerce")
    df["TargetPrice"] = pd.to_numeric(df.get("Target Price", df["EntryPrice"] * 1.01), errors="coerce")
    df["Stoploss"] = pd.to_numeric(df["Stoploss"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["Volatility"] = pd.to_numeric(df["Volatility"], errors="coerce")

    df["LogEntry"] = df["EntryPrice"].apply(lambda x: log(x) if x > 0 else 0)
    df["Hour"] = df["Entry Timestamp"].dt.hour
    df["DayOfWeek"] = df["Entry Timestamp"].dt.weekday
    df["TradeDurationMin"] = (df["Target Timestamp"] - df["Entry Timestamp"]).dt.total_seconds() / 60
    df["IsMorningTrade"] = df["Hour"].between(8, 12).astype(int)
    df["IsNYSession"] = df["Trading Session"].str.lower().str.contains("ny").astype(int)
    df["RiskRewardRatio"] = np.where(
        (df["Stoploss"] > 0) & (abs(df["EntryPrice"] - df["Stoploss"]) > 0),
        abs(df["TargetPrice"] - df["EntryPrice"]) / abs(df["EntryPrice"] - df["Stoploss"]),
        1.0
    )
    df["IsHighVolatility"] = (df["Volatility"] > df["Volatility"].median()).astype(int)

    # === 🧠 Apply Technical Indicators (as in training) ===
    # === Technical Indicators with Safe Handling ===
    df["EMA_9"] = ta.ema(df["EntryPrice"], length=9)
    df["EMA_21"] = ta.ema(df["EntryPrice"], length=21)
    df["RSI_14"] = ta.rsi(df["EntryPrice"], length=14)
    df["ATR_14"] = ta.atr(
        high=df["EntryPrice"],
        low=df["EntryPrice"],
        close=df["EntryPrice"],
        length=14
    )

    # Bollinger Bands
    bb = ta.bbands(df["EntryPrice"], length=20)
    if bb is not None and "BBU_20_2.0" in bb and "BBL_20_2.0" in bb:
        df["BB_Width"] = bb["BBU_20_2.0"] - bb["BBL_20_2.0"]
    else:
        df["BB_Width"] = np.nan

    # MACD
    macd = ta.macd(df["EntryPrice"], fast=12, slow=26, signal=9)
    if macd is not None and "MACD_12_26_9" in macd and "MACDh_12_26_9" in macd:
        df["MACD_Line"] = macd["MACD_12_26_9"]
        df["MACD_Hist"] = macd["MACDh_12_26_9"]
    else:
        df["MACD_Line"] = np.nan
        df["MACD_Hist"] = np.nan

    # === Drop rows with missing required indicators ===
    required_cols = [
        "LogEntry", "TradeDurationMin", "EMA_9", "EMA_21", "RSI_14",
        "ATR_14", "BB_Width", "MACD_Line", "MACD_Hist"
    ]

    # Instead of dropping rows, fill missing indicators with zero
    for col in required_cols:
        if col not in df.columns or df[col].isna().all():
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    return df[FEATURE_COLUMNS]



# === Predict Profit/Loss ===
def predict_from_pipeline(df_features: pd.DataFrame, pipeline) -> float:
    return pipeline.predict(df_features)[0]


# === Calculate Confidence ===
def compute_confidence(actual_pl, predicted_pl, penalty_inputs) -> tuple:
    delta = abs(actual_pl - predicted_pl)
    base_conf = max(0, 100 * (1 - delta / 20.0))

    # Penalize if unrealistic
    penalties = []
    if not 1000 <= penalty_inputs.get("Entry Price", 0) <= 5000:
        base_conf *= 0.7
        penalties.append("Entry Price")
    if not 100 <= penalty_inputs.get("Volume", 0) <= 1_000_000:
        base_conf *= 0.8
        penalties.append("Volume")
    if not 0 < penalty_inputs.get("Volatility", 0) <= 100:
        base_conf *= 0.9
        penalties.append("Volatility")

    reason = (
        "✅ High confidence — within acceptable range." if base_conf > 80 else
        "⚠️ Moderate confidence — slightly off." if base_conf > 50 else
        "❌ Low confidence — high deviation."
    )

    if penalties:
        reason += f" Adjusted for: {', '.join(penalties)}"

    return round(base_conf, 2), reason


# === Pretty Print Summary ===
def show_summary(data, predicted_pl, predicted_target, conf_user, reason_user, conf_model, reason_model, target_provided):
    entry_price = float(data["Entry Price"])
    actual_target = float(data.get("Target Price", 0))
    actual_pl = actual_target - entry_price if actual_target else None
    duration = (pd.to_datetime(data["Target Timestamp"], dayfirst=True) - pd.to_datetime(data["Entry Timestamp"], dayfirst=True)).total_seconds() / 60

    print("\n📋 Trade Summary:")
    print(f"🪙 {TRADE_SUMMARY_FIELDS['Symbol']}: {SYMBOL_NAME}")
    print(f"📦 {TRADE_SUMMARY_FIELDS['Lots']}: {DEFAULT_LOTS}")
    print(f"💸 {TRADE_SUMMARY_FIELDS['Entry Price']}: ₹{entry_price:.2f}")
    print(f"🎯 {TRADE_SUMMARY_FIELDS['Predicted Target Price']}: ₹{predicted_target:.2f}")
    print(f"💰 {TRADE_SUMMARY_FIELDS['Profit/Loss']}: ₹{predicted_pl:.2f}")

    if target_provided and actual_pl is not None:
        print(f"🎯 {TRADE_SUMMARY_FIELDS['Target Price (Given)']}: ₹{actual_target:.2f}")
        print(f"📐 {TRADE_SUMMARY_FIELDS['Actual Profit/Loss']}: ₹{actual_pl:.2f}")
        print(f"🔒 {TRADE_SUMMARY_FIELDS['Confidence (User Target)']}: {conf_user}%")
        print(f"ℹ️  {TRADE_SUMMARY_FIELDS['Confidence Reason (User Target)']}: {reason_user}")
    else:
        print("⚠️ Target Price was not provided by user — fallback value used for calculations.")

    print(f"🤖 {TRADE_SUMMARY_FIELDS['Confidence (Model Prediction)']}: {conf_model}%")
    print(f"ℹ️  {TRADE_SUMMARY_FIELDS['Confidence Reason (Model Prediction)']}: {reason_model}")
    print(f"\n🧠 {TRADE_SUMMARY_FIELDS['Strategy']}: {data.get('Strategy')}")
    print(f"📝 {TRADE_SUMMARY_FIELDS['Strategy Description']}: {data.get('Strategy Description')}")
    print(f"📊 {TRADE_SUMMARY_FIELDS['Trading Session']}: {data.get('Trading Session')}")
    print(f"📶 {TRADE_SUMMARY_FIELDS['Live Sentiment']}: {data.get('Live Sentiment')}")
    print(f"🛑 {TRADE_SUMMARY_FIELDS['Stoploss']}: ₹{data.get('Stoploss')}")
    print(f"📈 {TRADE_SUMMARY_FIELDS['Volatility']}: {data.get('Volatility')}")
    print(f"📉 {TRADE_SUMMARY_FIELDS['Volume']}: {data.get('Volume')}")
    print(f"🕒 {TRADE_SUMMARY_FIELDS['Entry Timestamp']}: {data.get('Entry Timestamp')}")
    print(f"⏱️ {TRADE_SUMMARY_FIELDS['Duration']}: {duration:.2f} minutes")


# === Main Prediction Flow ===
def predict_from_json(json_input: dict):
    base_df = pd.read_csv(DATA_PATH)
    validate_input(json_input)
    json_input, target_provided = apply_defaults(json_input, base_df)


    df_features = engineer_features(json_input)
    pipeline = load_pipeline()
    predicted_pl = predict_from_pipeline(df_features, pipeline)
    predicted_target = float(json_input["Entry Price"]) + predicted_pl

    target_price = float(json_input.get("Target Price", 0))
    entry_price = float(json_input["Entry Price"])
    actual_pl = target_price - entry_price if target_price else None

    # Confidence Calculations
    conf_user, reason_user = (None, None)
    if target_price:
        conf_user, reason_user = compute_confidence(actual_pl, predicted_pl, json_input)

    conf_model, reason_model = compute_confidence(predicted_pl, predicted_pl, json_input)

    show_summary(json_input, predicted_pl, predicted_target, conf_user, reason_user, conf_model, reason_model,
                 target_provided)


# === CLI Driver ===
def main():
    print("🔹 Type 'file' to read from input.json or 'manual' to paste JSON:")
    mode = input(">>> ").strip().lower()
    try:
        if mode == "file":
            with open("input.json", "r") as f:
                json_input = json.load(f)
        elif mode == "manual":
            print("📥 Paste your JSON input below (Enter twice to submit):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            json_input = json.loads("\n".join(lines))
        else:
            print("❌ Invalid mode.")
            return

        predict_from_json(json_input)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
