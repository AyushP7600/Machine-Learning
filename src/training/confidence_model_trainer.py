# === confidence_model_trainer.py ===

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from math import log

from config.paths import DATA_PATH, MODEL_PATH, PIPELINE_PATH, CONFIDENCE_MODEL_PATH
from config.constants import FEATURE_COLUMNS, DEFAULT_LOTS


def train_confidence_model():
    print("\n📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("❌ Dataset is empty. Please check your CSV file.")

    print("🧠 Feature engineering...")
    df['Entry Timestamp'] = pd.to_datetime(df['Entry Timestamp'], dayfirst=True, errors='coerce')
    df['Target Timestamp'] = pd.to_datetime(df['Target Timestamp'], dayfirst=True, errors='coerce')
    df['Hour'] = df['Entry Timestamp'].dt.hour
    df['DayOfWeek'] = df['Entry Timestamp'].dt.dayofweek
    df['Lots'] = DEFAULT_LOTS

    # Drop invalid rows
    df.dropna(subset=['Entry Price', 'Target Price'], inplace=True)
    df = df[df['Entry Price'] > 0]

    # Feature engineering
    df['Profit/Loss'] = df['Target Price'] - df['Entry Price']
    df['TargetPctGain'] = (df['Profit/Loss'] / df['Entry Price']) * 100
    df['LogEntry'] = df['Entry Price'].apply(lambda x: log(x))
    df['LogTarget'] = df['Target Price'].apply(lambda x: log(x))

    df.dropna(subset=FEATURE_COLUMNS + ['Profit/Loss'], inplace=True)

    if df.empty:
        raise ValueError("❌ Dataset is empty after cleaning.")

    print("⚙️ Loading main pipeline and model...")
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)

    print("🔄 Transforming features...")
    X = df[FEATURE_COLUMNS]
    X_transformed = pipeline.transform(X)

    print("🔍 Predicting Profit/Loss using main model...")
    predicted_pl = model.predict(X_transformed)
    actual_pl = df['Profit/Loss'].values

    print("📐 Calculating confidence based on prediction error...")
    errors = np.abs(predicted_pl - actual_pl)
    max_error = np.percentile(errors, 95) + 1e-6
    confidence_scores = 1 - (errors / max_error)
    confidence_scores = np.clip(confidence_scores, 0, 1)

    print("🎯 Training confidence model...")
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, confidence_scores, test_size=0.2, random_state=42)

    confidence_model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    confidence_model.fit(X_train, y_train)

    preds = confidence_model.predict(X_test)

    print("✅ Evaluation Metrics:")
    print(f"R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")

    print("💾 Saving trained confidence model...")
    joblib.dump(confidence_model, CONFIDENCE_MODEL_PATH)
    print(f"✅ Saved at {CONFIDENCE_MODEL_PATH}")


if __name__ == "__main__":
    train_confidence_model()
