"""
preprocess.py
-------------
Loads raw transaction data, engineers a couple of extra features,
builds a scikit-learn ColumnTransformer (scaling + one-hot encoding),
and splits the data into stratified train / validation / test sets.

This module is imported by train.py, evaluate.py and predict.py so the
exact same transformation logic is used everywhere (no train/serve skew).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_raw_data(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features that tend to help fraud models."""
    df = df.copy()

    # Night-time transactions (midnight-5am) are disproportionately risky.
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)

    # High-value transaction relative to the account's typical spend.
    df["high_value_flag"] = (df["ratio_to_median_purchase_price"] > 3).astype(int)

    # Card-not-present + not chip/pin is a classic risk combo.
    df["remote_no_verification"] = (
        (df["card_present"] == 0) & (df["used_chip"] == 0) & (df["used_pin_number"] == 0)
    ).astype(int)

    return df


def get_feature_columns():
    """Feature columns after engineering (used consistently everywhere)."""
    extra_binary = ["is_night", "high_value_flag", "remote_no_verification"]
    numeric = config.NUMERIC_FEATURES
    binary = config.BINARY_FEATURES + extra_binary
    categorical = config.CATEGORICAL_FEATURES
    return numeric, binary, categorical


def build_preprocessor() -> ColumnTransformer:
    numeric, binary, categorical = get_feature_columns()

    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("bin", "passthrough", binary),
            ("cat", categorical_pipeline, categorical),
        ]
    )
    return preprocessor


def split_data(df: pd.DataFrame):
    """Stratified split into train / val / test, preserving fraud ratio in each."""
    numeric, binary, categorical = get_feature_columns()
    feature_cols = numeric + binary + categorical

    X = df[feature_cols]
    y = df[config.TARGET_COL]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    val_fraction_of_train = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction_of_train,
        stratify=y_train_full,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare():
    """Convenience wrapper: raw CSV -> engineered features -> split."""
    df = load_raw_data()
    df = engineer_features(df)
    return split_data(df)
