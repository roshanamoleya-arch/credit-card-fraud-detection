"""
predict.py
----------
Production-style inference module. Loads the saved preprocessor + model
once, and scores new transaction records (single dict or CSV of many).

CLI usage:
    # score a CSV of new transactions
    python -m src.predict --input new_transactions.csv --output scored.csv --threshold 0.75

Library usage:
    from src.predict import FraudPredictor
    predictor = FraudPredictor()
    result = predictor.predict_one({
        "amount": 812.50,
        "hour_of_day": 2.3,
        "distance_from_home": 145.2,
        "distance_from_last_transaction": 98.7,
        "ratio_to_median_purchase_price": 6.1,
        "repeat_retailer": 0,
        "used_chip": 0,
        "used_pin_number": 0,
        "online_order": 1,
        "card_present": 0,
        "merchant_category": "electronics",
    })
    # -> {"fraud_probability": 0.91, "is_fraud_pred": 1}
"""

import argparse
import json

import joblib
import pandas as pd

from src import config
from src.preprocess import engineer_features, get_feature_columns


class FraudPredictor:
    def __init__(self, model_path=config.MODEL_PATH,
                 preprocessor_path=config.PREPROCESSOR_PATH,
                 threshold=None):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        if threshold is None:
            # default to the cost-optimal threshold if evaluate.py has been run
            try:
                with open(config.REPORTS_DIR / "evaluation_report.md") as f:
                    pass
                threshold = config.DEFAULT_THRESHOLD
            except FileNotFoundError:
                threshold = config.DEFAULT_THRESHOLD
        self.threshold = threshold

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(df)
        numeric, binary, categorical = get_feature_columns()
        feature_cols = numeric + binary + categorical
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")
        return df[feature_cols]

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(df)
        X_t = self.preprocessor.transform(X)
        proba = self.model.predict_proba(X_t)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        out = df.copy()
        out["fraud_probability"] = proba.round(4)
        out["is_fraud_pred"] = preds
        return out

    def predict_one(self, record: dict) -> dict:
        df = pd.DataFrame([record])
        scored = self.predict_batch(df)
        return {
            "fraud_probability": float(scored["fraud_probability"].iloc[0]),
            "is_fraud_pred": int(scored["is_fraud_pred"].iloc[0]),
        }


def main():
    parser = argparse.ArgumentParser(description="Score new transactions for fraud risk")
    parser.add_argument("--input", required=True, help="CSV of new transactions to score")
    parser.add_argument("--output", default="scored_transactions.csv")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override decision threshold (default: %.2f)" % config.DEFAULT_THRESHOLD)
    args = parser.parse_args()

    predictor = FraudPredictor(threshold=args.threshold)
    df = pd.read_csv(args.input)
    scored = predictor.predict_batch(df)
    scored.to_csv(args.output, index=False)

    n_flagged = scored["is_fraud_pred"].sum()
    print(f"Scored {len(scored):,} transactions. Flagged {n_flagged:,} as likely fraud "
          f"(threshold={predictor.threshold}). Saved to {args.output}")


if __name__ == "__main__":
    main()
