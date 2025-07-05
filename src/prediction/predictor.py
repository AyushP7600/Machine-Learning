import sys
import joblib
import pandas as pd
import json
from datetime import datetime
from math import log

from config.constants import (
    SYMBOL_NAME, DEFAULT_LOTS,
    FEATURE_COLUMNS, OPTIONAL_INPUT_FIELDS,
    REQUIRED_INPUT_FIELDS
)
from config.paths import DATA_PATH, PIPELINE_PATH, MODEL_PATH


def load_pipeline_and_model():
    preprocessor = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def validate_json(json_input: dict):
    missing = [field for field in REQUIRED_INPUT_FIELDS if field not in json_input or json_input[field] in [None, ""]]
    if missing:
        raise ValueError(f"❌ Missing required JSON fields: {missing}")


def fill_missing_values(input_data: dict, df_base: pd.DataFrame) -> dict:
    for field in OPTIONAL_INPUT_FIELDS:
        if field not in input_data or input_data[field] in [None, ""]:
            input_data[field] = df_base[field].mode().iloc[0] if field in df_base else "N/A"
    return input_data


def parse_and_augment_fields(input_data: dict) -> dict:
    entry_time = datetime.strptime(input_data['Entry Timestamp'], "%d-%m-%Y %H:%M")
    target_time = datetime.strptime(input_data['Target Timestamp'], "%d-%m-%Y %H:%M")

    input_data['Entry Timestamp'] = entry_time
    input_data['Target Timestamp'] = target_time
    input_data['Hour'] = entry_time.hour
    input_data['DayOfWeek'] = entry_time.weekday()
    input_data['Lots'] = DEFAULT_LOTS

    entry_price = float(input_data['Entry Price'])
    input_data['EntryPrice'] = entry_price
    input_data['LogEntry'] = log(entry_price)

    return input_data


def preprocess_input(json_data: dict, df_base: pd.DataFrame) -> pd.DataFrame:
    validate_json(json_data)
    json_data = fill_missing_values(json_data, df_base)
    json_data = parse_and_augment_fields(json_data)

    df = pd.DataFrame([json_data])
    return df


def predict_profit_loss(df_input: pd.DataFrame, preprocessor, model) -> float:
    X = df_input[FEATURE_COLUMNS]
    X_processed = preprocessor.transform(X)
    return model.predict(X_processed)[0]


def calculate_confidence(actual_pl: float, predicted_pl: float, reference: float = 20.0) -> tuple:
    delta = abs(actual_pl - predicted_pl)
    confidence = max(0, 100 * (1 - delta / reference))
    reason = (
        "✅ High confidence — well within expected range."
        if confidence > 80 else
        "⚠️ Moderate confidence — some deviation."
        if confidence > 50 else
        "❌ Low confidence — target deviates significantly."
    )
    return round(confidence, 2), reason


def penalize_unrealistic_input(input_data: dict, confidence: float, reason: str) -> tuple:
    penalties = []

    max_price, min_price = 5000, 1000
    max_vol, min_vol = 1_000_000, 100
    max_volatility = 100

    entry_price = float(input_data.get("Entry Price", 0))
    if entry_price > max_price or entry_price < min_price:
        confidence *= 0.7
        penalties.append("Entry Price")

    volume = float(input_data.get("Volume", 0))
    if volume > max_vol or volume < min_vol:
        confidence *= 0.8
        penalties.append("Volume")

    volatility = float(input_data.get("Volatility", 0))
    if volatility > max_volatility or volatility <= 0:
        confidence *= 0.9
        penalties.append("Volatility")

    if penalties:
        reason += f" ⚠️ Adjusted due to unrealistic fields: {', '.join(penalties)}."

    return round(confidence, 2), reason


def display_summary(input_data, target_price, predicted_pl, conf_user_target, reason_user, conf_model_target, reason_model):
    entry_price = float(input_data.get("Entry Price", 0.0))
    predicted_target = entry_price + predicted_pl

    entry_time = input_data.get("Entry Timestamp")
    target_time = input_data.get("Target Timestamp")
    duration_minutes = (target_time - entry_time).total_seconds() / 60

    print("\n🧾 Trade Summary:")
    print(f"🪙 Symbol: {SYMBOL_NAME}")
    print(f"📦 Lots: {DEFAULT_LOTS}")
    print(f"💸 Entry Price: ₹{entry_price:.2f}")
    print(f"🎯 Predicted Target Price: ₹{predicted_target:.2f}")
    print(f"💰 Estimated Profit/Loss: ₹{predicted_pl:.2f}")

    if target_price is not None:
        actual_pl = target_price - entry_price
        print(f"🎯 Actual Target Price Given: ₹{target_price:.2f}")
        print(f"📐 Actual Profit/Loss (Given): ₹{actual_pl:.2f}")

        print(f"\n🔒 Confidence based on your Target Price (input): {conf_user_target}%")
        print(f"ℹ️ Reason: {reason_user}")

    print(f"\n🤖 Confidence based on Predicted Target Price (model): {conf_model_target}%")
    print(f"ℹ️ Reason: {reason_model}")

    print(f"\n🧠 Strategy: {input_data.get('Strategy')}")
    print(f"📝 Description: {input_data.get('Strategy Description')}")
    print(f"📊 Session: {input_data.get('Trading Session')}")
    print(f"📶 Sentiment: {input_data.get('Live Sentiment')}")
    print(f"🛑 Stoploss: ₹{input_data.get('Stoploss')}")
    print(f"📈 Volatility: {input_data.get('Volatility')}")
    print(f"📉 Volume: {input_data.get('Volume')}")
    print(f"🕒 Entry → Target: {entry_time} → {target_time}")
    print(f"⏱️ Duration: {duration_minutes:.2f} minutes")


def predict_from_json(json_input: dict):
    df_base = pd.read_csv(DATA_PATH)
    df_input = preprocess_input(json_input, df_base)
    preprocessor, model = load_pipeline_and_model()
    predicted_pl = predict_profit_loss(df_input, preprocessor, model)

    entry_price = float(json_input["Entry Price"])
    target_price = float(json_input["Target Price"]) if "Target Price" in json_input else None

    # Confidence based on user-given Target Price
    if target_price is not None:
        actual_pl = target_price - entry_price
        conf_user, reason_user = calculate_confidence(actual_pl, predicted_pl)
        conf_user, reason_user = penalize_unrealistic_input(json_input, conf_user, reason_user)
    else:
        conf_user = reason_user = None

    # Confidence based on model’s own prediction
    conf_model, reason_model = calculate_confidence(predicted_pl, predicted_pl)
    conf_model, reason_model = penalize_unrealistic_input(json_input, conf_model, reason_model)

    display_summary(json_input, target_price, predicted_pl, conf_user, reason_user, conf_model, reason_model)


# === Entry Point ===
if __name__ == "__main__":
    print("🔷 Choose input mode — type 'file' to read from input.json OR 'manual' to paste JSON:")
    mode = input(">>> ").strip().lower()

    if mode == "file":
        try:
            with open("input.json", "r") as f:
                sample_json = json.load(f)
            predict_from_json(sample_json)
        except Exception as e:
            print(f"❌ Failed to read from file: {e}")
            sys.exit(1)

    elif mode == "manual":
        print("🔹 Paste your JSON input below (multi-line, press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        try:
            json_input_str = "\n".join(lines)
            sample_json = json.loads(json_input_str)
            predict_from_json(sample_json)
        except Exception as e:
            print(f"❌ Invalid JSON input: {e}")
            sys.exit(1)

    else:
        print("❌ Invalid mode. Please type 'file' or 'manual'.")
