import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../data/sample_XAUUSD_trading_data_with_lots.csv'))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../src/data/timestamped_trading_data_1500.csv'))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '../src/models/model.lgbm'))
PIPELINE_PATH = os.path.abspath(os.path.join(BASE_DIR, '../src/models/preprocessing_pipeline.joblib'))
CONFIDENCE_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '../src/models/confidence_model.lgbm'))
