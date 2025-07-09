from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import numpy as np
from collections import OrderedDict

# === Project modules ===
from prediction.predictor import (
    validate_input,
    apply_defaults,
    engineer_features,
    predict_from_pipeline,
    compute_confidence,
)
from config.paths import DATA_PATH, MODEL_PATH
from config.constants import (
    SYMBOL_NAME,
    DEFAULT_LOTS,
    REQUIRED_INPUT_FIELDS,
)

app = Flask(__name__)

# === Load model & base data ===
base_df = pd.read_csv(DATA_PATH)
pipeline = joblib.load(MODEL_PATH)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_input = request.get_json(force=True)
        print("✅ Received JSON input:", json_input)

        # === 1. Validate input ===
        validate_input(json_input)
        print("✅ Input validation passed")

        # === 2. Apply defaults for optional fields ===
        json_input, _ = apply_defaults(json_input, base_df)  # discard the second item
        print("🔧 Defaults applied. Final input keys:", list(json_input.keys()))

        # === 3. Check if required keys are still present ===
        required_fields = ["Entry Price", "Strategy", "Entry Timestamp", "Target Timestamp"]
        for field in required_fields:
            if field not in json_input:
                raise KeyError(f"Missing required field: '{field}'")

        # === 4. Feature Engineering ===
        df_features = engineer_features(json_input)
        print("✅ Feature engineering completed")

        # === 5. Prediction ===
        entry_price = float(json_input["Entry Price"])
        predicted_pl = predict_from_pipeline(df_features, pipeline)
        predicted_target = round(entry_price + predicted_pl, 2)
        target_price = float(json_input.get("Target Price", 0))
        actual_pl = target_price - entry_price if target_price else None

        # === 6. Confidence ===
        conf_user, reason_user = (None, None)
        if target_price:
            conf_user, reason_user = compute_confidence(actual_pl, predicted_pl, json_input)
        conf_model, reason_model = compute_confidence(predicted_pl, predicted_pl, json_input)

        # === 7. Duration Calculation ===
        duration_minutes = None
        try:
            entry_ts = pd.to_datetime(json_input.get("Entry Timestamp"), format="%d-%m-%Y %H:%M")
            target_ts = pd.to_datetime(json_input.get("Target Timestamp"), format="%d-%m-%Y %H:%M")
            duration_minutes = round((target_ts - entry_ts).total_seconds() / 60, 2)
        except Exception as duration_err:
            print("⚠️ Failed to calculate duration:", duration_err)

        # === 8. Fallback Note if no target ===
        target_price_msg = None
        if not json_input.get("Target Price"):
            target_price_msg = "Target Price was not provided by user — fallback value used for calculations."

        # === 9. Ordered JSON Response ===
        response = OrderedDict([
            ("Symbol", SYMBOL_NAME),
            ("Lot Size", DEFAULT_LOTS),
            ("Entry Price", entry_price),
            ("Predicted Target Price", predicted_target),
            ("Predicted Profit/Loss", round(predicted_pl, 2)),
            ("Target Price Message", target_price_msg),
            ("Confidence (Model Prediction)", conf_model),
            ("Confidence Reason (Model Prediction)", reason_model),
            ("Confidence (User Target)", conf_user),
            ("Confidence Reason (User Target)", reason_user),
            ("Strategy", json_input.get("Strategy")),
            ("Strategy Description", json_input.get("Strategy Description")),
            ("Trading Session", json_input.get("Trading Session")),
            ("Live Sentiment", json_input.get("Live Sentiment")),
            ("Stoploss", json_input.get("Stoploss")),
            ("Volatility", json_input.get("Volatility")),
            ("Volume", json_input.get("Volume")),
            ("Entry Timestamp", json_input.get("Entry Timestamp")),
            ("Target Timestamp", json_input.get("Target Timestamp")),
            ("Duration (minutes)", duration_minutes),
        ])

        # === 10. Clean numpy values before sending JSON ===
        response = convert_numpy_types(response)
        print("✅ Response prepared successfully")
        return jsonify(response), 200

    except Exception as e:
        print("❌ Error during prediction:", e)
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 400


if __name__ == "__main__":
    app.run(debug=True)