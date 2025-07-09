import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
from math import log
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

from config.paths import DATA_PATH, MODEL_PATH, PIPELINE_PATH
from config.constants import REQUIRED_INPUT_FIELDS, OPTIONAL_INPUT_FIELDS, FEATURE_COLUMNS


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def fill_optional_fields(df: pd.DataFrame) -> pd.DataFrame:
    fallback_defaults = {
        "Strategy Description": "Default strategy",
        "Trading Session": "London Session",
        "Live Sentiment": "Neutral",
        "Stoploss": 0.0,
        "Volume": 10000,
        "Volatility": 1.0,
        "Target Price": df["Entry Price"] * 1.01 if "Entry Price" in df else 0.0
    }

    for col in OPTIONAL_INPUT_FIELDS:
        if col not in df.columns:
            df[col] = fallback_defaults.get(col, "N/A")
        df[col] = df[col].fillna(fallback_defaults.get(col))

    return df


def parse_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df["Entry Timestamp"] = pd.to_datetime(df["Entry Timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["Target Timestamp"] = pd.to_datetime(df["Target Timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["EntryPrice"] = pd.to_numeric(df["Entry Price"], errors="coerce")
    df["TargetPrice"] = pd.to_numeric(df["Target Price"], errors="coerce")
    df["Stoploss"] = pd.to_numeric(df["Stoploss"], errors="coerce")

    df.dropna(subset=["Entry Timestamp", "Target Timestamp", "EntryPrice", "TargetPrice", "Stoploss"], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["LogEntry"] = df["EntryPrice"].apply(lambda x: log(x) if x > 0 else np.nan)
    df["Hour"] = df["Entry Timestamp"].dt.hour
    df["DayOfWeek"] = df["Entry Timestamp"].dt.weekday
    df["IsMorningTrade"] = df["Hour"].between(8, 12).astype(int)
    df["IsNYSession"] = df["Trading Session"].str.contains("NY", case=False, na=False).astype(int)
    df["TradeDurationMin"] = (df["Target Timestamp"] - df["Entry Timestamp"]).dt.total_seconds() / 60

    df["RiskRewardRatio"] = np.where(
        (df["Stoploss"] > 0) & (abs(df["EntryPrice"] - df["Stoploss"]) > 0),
        abs(df["TargetPrice"] - df["EntryPrice"]) / abs(df["EntryPrice"] - df["Stoploss"]),
        1.0
    )

    df["IsHighVolatility"] = (df["Volatility"] > df["Volatility"].median()).astype(int)
    df["RiskBuffer"] = abs(df["EntryPrice"] - df["Stoploss"])

    df.sort_values("Entry Timestamp", inplace=True)
    df["EMA_9"] = ta.ema(df["EntryPrice"], length=9)
    df["EMA_21"] = ta.ema(df["EntryPrice"], length=21)
    df["RSI_14"] = ta.rsi(df["EntryPrice"], length=14)
    df["ATR_14"] = ta.atr(high=df["EntryPrice"], low=df["EntryPrice"], close=df["EntryPrice"], length=14)
    bb = ta.bbands(df["EntryPrice"], length=20)
    df["BB_Width"] = bb["BBU_20_2.0"] - bb["BBL_20_2.0"]
    macd = ta.macd(df["EntryPrice"], fast=12, slow=26, signal=9)
    df["MACD_Line"] = macd["MACD_12_26_9"]
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    df.dropna(subset=[
        "LogEntry", "TradeDurationMin", "EMA_9", "EMA_21", "RSI_14",
        "ATR_14", "BB_Width", "MACD_Line", "MACD_Hist"
    ], inplace=True)

    return df


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=40,
            class_weight='balanced',  # <-- Balanced to reduce false positives
            random_state=42,
            n_jobs=-1
        ))
    ])
    return model_pipeline


def plot_precision_recall(y_true, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend()
    plt.title("Precision-Recall vs Threshold")
    plt.grid(True)
    plt.show()


def train_and_save():
    df = load_data(DATA_PATH)
    df = fill_optional_fields(df)
    df = parse_and_clean(df)
    df = engineer_features(df)

    if df.empty:
        raise ValueError("❌ Dataset is empty after preprocessing.")

    df["Target"] = (df["TargetPrice"] > df["EntryPrice"]).astype(int)

    print("\n🔍 Class distribution:")
    print(df["Target"].value_counts())

    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    # Predict probabilities and use custom threshold
    y_pred_probs = pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.75  # Tune this threshold based on your precision-recall curve!
    y_pred = (y_pred_probs > threshold).astype(int)

    print("\n📊 Model Evaluation:")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📝 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Optional: Uncomment to visualize threshold tuning during dev
    # plot_precision_recall(y_test, y_pred_probs)

    # Save full pipeline (preprocessor + model) for easy inference
    joblib.dump(pipeline, MODEL_PATH)
    print("\n✅ Full pipeline saved successfully.")


if __name__ == "__main__":
    train_and_save()
