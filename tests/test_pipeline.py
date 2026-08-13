"""
test_pipeline.py
-----------------
Lightweight sanity tests for the fraud-detection pipeline. These are not
exhaustive statistical tests -- they check that each stage of the
pipeline runs and produces outputs of the expected shape/type, which
catches the most common breakages (schema drift, path errors, etc.)

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_dataset
from src import config
from src.preprocess import build_preprocessor, engineer_features, get_feature_columns, split_data


@pytest.fixture(scope="module")
def sample_df():
    return generate_dataset(n_samples=5000, fraud_rate=0.02, seed=1)


def test_generate_dataset_shape_and_columns(sample_df):
    assert len(sample_df) == 5000
    expected_cols = {
        "transaction_id", "amount", "hour_of_day", "distance_from_home",
        "distance_from_last_transaction", "ratio_to_median_purchase_price",
        "repeat_retailer", "used_chip", "used_pin_number", "online_order",
        "card_present", "merchant_category", "is_fraud",
    }
    assert expected_cols.issubset(set(sample_df.columns))


def test_fraud_rate_within_tolerance(sample_df):
    rate = sample_df["is_fraud"].mean()
    # allow for label noise injection to shift rate slightly
    assert 0.01 < rate < 0.03


def test_no_nulls(sample_df):
    assert sample_df.isnull().sum().sum() == 0


def test_engineer_features_adds_columns(sample_df):
    out = engineer_features(sample_df)
    for col in ["is_night", "high_value_flag", "remote_no_verification"]:
        assert col in out.columns
        assert set(out[col].unique()).issubset({0, 1})


def test_split_data_is_stratified_and_disjoint(sample_df):
    df = engineer_features(sample_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(df)

    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    overall_rate = df["is_fraud"].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - overall_rate) < 0.02


def test_preprocessor_fits_and_transforms(sample_df):
    df = engineer_features(sample_df)
    numeric, binary, categorical = get_feature_columns()
    X = df[numeric + binary + categorical]

    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)

    assert X_t.shape[0] == len(X)
    assert X_t.shape[1] > len(numeric) + len(binary)  # one-hot expands categorical


def test_predictor_end_to_end(tmp_path):
    """Smoke test: train a tiny model on synthetic data and score a record."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    df = generate_dataset(n_samples=3000, fraud_rate=0.05, seed=2)
    df = engineer_features(df)
    numeric, binary, categorical = get_feature_columns()
    X = df[numeric + binary + categorical]
    y = df["is_fraud"]

    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_t, y)

    model_path = tmp_path / "model.joblib"
    prep_path = tmp_path / "prep.joblib"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, prep_path)

    from src.predict import FraudPredictor
    predictor = FraudPredictor(model_path=model_path, preprocessor_path=prep_path, threshold=0.5)

    record = df.iloc[0][numeric + binary + categorical].to_dict()
    result = predictor.predict_one(record)

    assert "fraud_probability" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["is_fraud_pred"] in (0, 1)
