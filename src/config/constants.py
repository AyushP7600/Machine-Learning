"""
constants.py

Defines system-wide constants for input validation and response generation.
Follows SOLID principles and clean design for long-term scalability.
"""

# === 🔒 Immutable Defaults (Non-editable during runtime) ===

SYMBOL_NAME: str = "XAUUSD"
DEFAULT_LOTS: float = 0.01


# === 📥 Input Field Definitions ===

# Required fields that must be present in input JSON
REQUIRED_INPUT_FIELDS: list[str] = [
    "Entry Price",
    "Strategy",
    "Entry Timestamp",
    "Target Timestamp"
]

# Optional fields accepted in input JSON
OPTIONAL_INPUT_FIELDS: list[str] = [
    "Strategy Description",
    "Trading Session",
    "Live Sentiment",
    "Stoploss",
    "Volume",
    "Volatility",
    "Target Price"
]

# All fields that can be accepted from input JSON
ALL_INPUT_FIELDS: list[str] = REQUIRED_INPUT_FIELDS + OPTIONAL_INPUT_FIELDS


# === 📤 Output Summary Fields ===

# Maps internal or computed fields to clean user-facing labels
TRADE_SUMMARY_FIELDS: dict[str, str] = {
    "Symbol": SYMBOL_NAME,
    "Lots": DEFAULT_LOTS,
    "Entry Price": "Entry Price",
    "Predicted Target Price": "Predicted Target Price",
    "Profit/Loss": "Profit/Loss",
    "Target Price (Given)": "Target Price",
    "Actual Profit/Loss": "Actual Profit/Loss",

    "Confidence (User Target)": "Confidence (User Target)",
    "Confidence Reason (User Target)": "Reason (User Target)",
    "Confidence (Model Prediction)": "Confidence (Model Prediction)",
    "Confidence Reason (Model Prediction)": "Reason (Model Prediction)",

    "Strategy": "Strategy",
    "Strategy Description": "Strategy Description",
    "Trading Session": "Trading Session",
    "Live Sentiment": "Live Sentiment",
    "Stoploss": "Stoploss",
    "Volatility": "Volatility",
    "Volume": "Volume",
    "Entry Timestamp": "Entry Timestamp",
    "Target Timestamp": "Target Timestamp",
    "Duration": "Duration (minutes)"
}




# === 🧠 Final Features Used for Training/Prediction ===

FEATURE_COLUMNS: list[str] = [
    "EntryPrice", "LogEntry", "Hour", "DayOfWeek", "TradeDurationMin",
    "Stoploss", "Volume", "Volatility", "RiskRewardRatio",
    "IsMorningTrade", "IsNYSession", "IsHighVolatility",
    "EMA_9", "EMA_21", "RSI_14", "ATR_14", "BB_Width", "MACD_Line", "MACD_Hist",
    "Strategy", "Strategy Description", "Trading Session", "Live Sentiment"
]

