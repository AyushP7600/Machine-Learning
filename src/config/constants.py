# === constants.py ===

# Model Config
SYMBOL_NAME = "XAUUSD"
DEFAULT_LOTS = 0.01

# === Input JSON Fields ===
REQUIRED_INPUT_FIELDS = [
    "Entry Price",
    "Strategy",
    "Entry Timestamp",
    "Target Timestamp"
]

OPTIONAL_INPUT_FIELDS = [
    "Strategy Description",
    "Trading Session",
    "Live Sentiment",
    "Stoploss",
    "Volume",
    "Volatility"
]

# === All Input Fields (JSON)
# Note: Target Price can be accepted for logging or evaluation, but not used as input feature.
ALL_INPUT_FIELDS = REQUIRED_INPUT_FIELDS + OPTIONAL_INPUT_FIELDS + ["Target Price"]

# === Feature Columns used in ML model (after preprocessing)
# ✅ DO NOT include Target Price in training or prediction features
FEATURE_COLUMNS = [
    "EntryPrice",            # derived from "Entry Price"
    "LogEntry",              # log of EntryPrice
    "Hour",                  # derived from timestamp
    "DayOfWeek",             # derived from timestamp
    "Strategy",
    "Strategy Description",
    "Trading Session",
    "Live Sentiment",
    "Stoploss",
    "Volume",
    "Volatility"
]

# === Output Display Field Labels ===
TRADE_SUMMARY_FIELDS = {
    "Symbol": SYMBOL_NAME,
    "Lots": DEFAULT_LOTS,
    "Entry Price": "Entry Price",
    "Predicted Target Price": "Predicted Target Price",
    "Profit/Loss": "Profit/Loss",
    "Confidence": "Confidence",
    "Confidence Reason": "Reason",
    "Strategy": "Strategy",
    "Strategy Description": "Strategy Description",
    "Trading Session": "Trading Session",
    "Live Sentiment": "Live Sentiment",
    "Stoploss": "Stoploss",
    "Volume": "Volume",
    "Volatility": "Volatility",
    "Entry Timestamp": "Entry Timestamp",
    "Target Timestamp": "Target Timestamp",
    "Duration": "Duration (minutes)",
    "Target Price (Given)": "Target Price"
}
