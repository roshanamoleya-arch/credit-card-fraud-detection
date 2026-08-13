"""
config.py
---------
Central configuration for the fraud detection pipeline.
Keeping paths/constants here means every script (train, evaluate, predict)
stays in sync without hardcoding values in multiple places.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "transactions.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# --- Columns -------------------------------------------------------------
TARGET_COL = "is_fraud"
ID_COL = "transaction_id"

NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
]

BINARY_FEATURES = [
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
    "card_present",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# --- Modeling ------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # taken out of the remaining training data

# Decision threshold for classifying a transaction as fraud.
# Tuned in evaluate.py via precision-recall trade-off; stored here as default.
DEFAULT_THRESHOLD = 0.5
