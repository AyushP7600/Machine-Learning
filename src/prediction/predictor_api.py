import joblib
import pandas as pd
from math import log
from datetime import datetime
from typing import Dict, Any
from config.constants import (
    SYMBOL_NAME, DEFAULT_LOTS,
    FEATURE_COLUMNS, OPTIONAL_INPUT_FIELDS, REQUIRED_INPUT_FIELDS
)
from config.paths import DATA_PATH, PIPELINE_PATH, MODEL_PATH


def load_pipeline_and_model():
    preprocessor = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def validate_json(json_input: Dict[str, Any]):
    missing = [
        field for field in REQUIRED_INPUT_FIELDS
        if field not in json_input or json_input[field] in [None, ""]
    ]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def fill_missing_values(input_data: Dict[str, Any], df_base: pd.DataFrame) -> Dict[str, Any]:
    for field in OPTIONAL_INPUT_FIELDS:
        if field not in input_data or input_data[field] in [None, ""]:
            input_data[field] = df_base[field].mode().iloc[0] if field in df_base else "N/A"
    return input_data


def parse_and_augment_fields(input_data: Dict[str, Any]) -> Dict[str, Any]:
    entry_time = pd.to_datetime(input_data.get('Entry Timestamp'), errors='coerce')
    target_time = pd.to_datetime(input_data.get('Target Timestamp'), errors='coerce')

    input_data.update({
        "Entry Timestamp": entry_time,
        "Target Timestamp": target_time,
        "Hour": entry_time.hour if not pd.isna(entry_time) else 0,
        "DayOfWeek": entry_time.weekday() if not pd.isna(entry_time) else 0
    })

    entry_price = float(input_data.get('Entry Price', 0))
    target_price = float(input_data.get('Target Price', entry_price))
    stoploss = float(input_data.get('Stoploss', 0.0))

    trade_duration = (
        (target_time - entry_time).total_seconds() / 60
        if not pd.isna(target_time) and not pd.isna(entry_time) else 0
    )

    input_data.update({
        "EntryPrice": entry_price,
        "LogEntry": log(entry_price) if entry_price > 0 else 0,
        "TradeDurationMin": trade_duration,
        "RiskRewardRatio": (
            abs(target_price - entry_price) / abs(entry_price - stoploss)
            if stoploss > 0 and abs(entry_price - stoploss) > 0 else 1.0
        ),
        "IsMorningTrade": int(8 <= entry_time.hour <= 12) if not pd.isna(entry_time) else 0,
        "IsNYSession": int("ny" in str(input_data.get("Trading Session", "")).lower()),
        "IsHighVolatility": int(float(input_data.get("Volatility", 0)) > 1.5)
    })

    return input_data


def preprocess_input(json_data: Dict[str, Any], df_base: pd.DataFrame) -> pd.DataFrame:
    validate_json(json_data)
    json_data = fill_missing_values(json_data, df_base)
    json_data = parse_and_augment_fields(json_data)
    return pd.DataFrame([json_data])


def predict_profit_loss(df_input: pd.DataFrame, preprocessor, model) -> float:
    X = df_input[FEATURE_COLUMNS]
    X_processed = preprocessor.transform(X)
    return model.predict(X_processed)[0]


def calculate_confidence(actual_pl: float, predicted_pl: float, reference: float = 20.0) -> tuple:
    delta = abs(actual_pl - predicted_pl)
    confidence = max(0, 100 * (1 - delta / reference))
    reason = (
        "High confidence — well within expected range." if confidence > 80 else
        "Moderate confidence — some deviation." if confidence > 50 else
        "Low confidence — target deviates significantly."
    )
    return round(confidence, 2), reason


def penalize_unrealistic_input(input_data: Dict[str, Any], confidence: float, reason: str) -> tuple:
    penalties = []
    max_price, min_price = 5000, 1000
    max_vol, min_vol = 1_000_000, 100
    max_volatility = 100

    entry_price = float(input_data.get("Entry Price", 0))
    volume = float(input_data.get("Volume", 0))
    volatility = float(input_data.get("Volatility", 0))

    if not min_price <= entry_price <= max_price:
        confidence *= 0.7
        penalties.append("Entry Price")

    if not min_vol <= volume <= max_vol:
        confidence *= 0.8
        penalties.append("Volume")

    if not 0 < volatility <= max_volatility:
        confidence *= 0.9
        penalties.append("Volatility")

    if penalties:
        reason += f" Adjusted due to unrealistic fields: {', '.join(penalties)}."

    return round(confidence, 2), reason


def display_summary(
    input_data: Dict[str, Any],
    target_price: float,
    predicted_pl: float,
    conf_user_target: float,
    reason_user: str,
    conf_model_target: float,
    reason_model: str
) -> Dict[str, Any]:

    entry_price = float(input_data.get("Entry Price", 0))
    predicted_target = entry_price + predicted_pl
    entry_time = input_data.get("Entry Timestamp")
    target_time = input_data.get("Target Timestamp")

    try:
        duration_minutes = (target_time - entry_time).total_seconds() / 60
    except Exception:
        duration_minutes = 0

    summary = {
        "Trade Summary": {
            "Symbol": SYMBOL_NAME,
            "Lots": DEFAULT_LOTS,
            "Entry Price": round(entry_price, 2),
            "Predicted Target Price": round(predicted_target, 2),
            "Estimated Profit/Loss": round(predicted_pl, 2)
        },
        "User Target Info": {
            "Target Price (Given)": None,
            "Actual Profit/Loss": None,
            "Confidence (User Target)": None,
            "Reason (User Target)": None
        },
        "Model Prediction Info": {
            "Confidence (Model Prediction)": conf_model_target,
            "Reason (Model Prediction)": reason_model
        },
        "Trade Details": {
            "Strategy": input_data.get("Strategy"),
            "Description": input_data.get("Strategy Description"),
            "Session": input_data.get("Trading Session"),
            "Sentiment": input_data.get("Live Sentiment"),
            "Stoploss": input_data.get("Stoploss"),
            "Volatility": input_data.get("Volatility"),
            "Volume": input_data.get("Volume"),
            "Time Range": {
                "Entry Timestamp": str(entry_time),
                "Target Timestamp": str(target_time)
            },
            "Duration (minutes)": round(duration_minutes, 2)
        }
    }

    try:
        actual_pl = target_price - entry_price
        summary["User Target Info"].update({
            "Target Price (Given)": round(target_price, 2),
            "Actual Profit/Loss": round(actual_pl, 2),
            "Confidence (User Target)": conf_user_target,
            "Reason (User Target)": reason_user
        })
    except Exception:
        summary["User Target Info"]["Reason (User Target)"] = "No target price provided by user"

    return summary


def predict_from_json(json_input: Dict[str, Any]) -> Dict[str, Any]:
    df_base = pd.read_csv(DATA_PATH)
    df_input = preprocess_input(json_input, df_base)
    preprocessor, model = load_pipeline_and_model()
    predicted_pl = predict_profit_loss(df_input, preprocessor, model)

    entry_price = float(json_input.get("Entry Price", 0))
    target_price = float(json_input.get("Target Price", entry_price)) if "Target Price" in json_input else None

    if target_price is not None:
        actual_pl = target_price - entry_price
        conf_user, reason_user = calculate_confidence(actual_pl, predicted_pl)
        conf_user, reason_user = penalize_unrealistic_input(json_input, conf_user, reason_user)
    else:
        conf_user, reason_user = 0.0, "No target price provided by user"

    conf_model, reason_model = calculate_confidence(predicted_pl, predicted_pl)
    conf_model, reason_model = penalize_unrealistic_input(json_input, conf_model, reason_model)

    # ✅ THIS is the final return: full dictionary (not string)
    return display_summary(
        json_input,
        target_price,
        predicted_pl,
        conf_user,
        reason_user,
        conf_model,
        reason_model
    )
