# === model_trainer.py ===

import pandas as pd
import numpy as np
import joblib
from math import log
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor

from config.paths import DATA_PATH, MODEL_PATH, PIPELINE_PATH
from config.constants import FEATURE_COLUMNS, OPTIONAL_INPUT_FIELDS

# === Load Dataset ===
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# === Fill default values for optional fields if missing ===
fallback_defaults = {
    "Strategy Description": "Default strategy",
    "Trading Session": "London Session",
    "Live Sentiment": "Neutral",
    "Stoploss": 0.0,
    "Volume": 10000,
    "Volatility": 1.0
}
for col in OPTIONAL_INPUT_FIELDS:
    if col not in df.columns:
        df[col] = fallback_defaults.get(col, "N/A")

# === Parse datetime columns ===
df["Entry Timestamp"] = pd.to_datetime(df["Entry Timestamp"], format="%d-%m-%Y %H:%M")
df["Target Timestamp"] = pd.to_datetime(df["Target Timestamp"], format="%d-%m-%Y %H:%M")

# === Feature Engineering ===
df["Hour"] = df["Entry Timestamp"].dt.hour
df["DayOfWeek"] = df["Entry Timestamp"].dt.weekday
df["EntryPrice"] = df["Entry Price"].astype(float)
df["TargetPrice"] = df["Target Price"].astype(float)
df["Profit/Loss"] = df["TargetPrice"] - df["EntryPrice"]
df["LogEntry"] = df["EntryPrice"].apply(lambda x: log(x))

# === Prepare model training data ===
X = df[FEATURE_COLUMNS]  # Should NOT include 'Target Price' explicitly as input
y = df["Profit/Loss"]

# === Split feature types ===
numeric_cols = ["EntryPrice", "LogEntry", "Hour", "DayOfWeek", "Stoploss", "Volume", "Volatility"]
categorical_cols = ["Strategy", "Strategy Description", "Trading Session", "Live Sentiment"]

# === Define preprocessing pipeline ===
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# === Define model pipeline ===
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(n_estimators=150, learning_rate=0.08, random_state=42))
])

# === Train/Test Split and Fit ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# === Save Artifacts ===
joblib.dump(model_pipeline.named_steps["regressor"], MODEL_PATH)
joblib.dump(model_pipeline.named_steps["preprocessor"], PIPELINE_PATH)

print("✅ Model and pipeline saved successfully.")
print("📦 Model trained on Profit/Loss using input features excluding 'Target Price'.")
