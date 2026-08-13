"""
train.py
--------
Trains and compares several classifiers for credit-card fraud detection,
handles severe class imbalance (~0.8% positive rate) with SMOTE
oversampling on the training fold only, selects the best model by
Precision-Recall AUC (the right metric for imbalanced classification,
since ROC-AUC is overly optimistic when negatives vastly outnumber
positives), and saves the winning model + preprocessor + metadata.

Run:
    python -m src.train
"""

import json
import time

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from src import config
from src.preprocess import build_preprocessor, load_and_prepare


def get_candidate_models():
    """Models chosen to represent 3 common tiers: linear baseline,
    bagging ensemble, and boosting ensemble."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def main():
    config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare()
    print(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    print(f"  train fraud rate: {y_train.mean()*100:.3f}%")

    print("Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    print("Applying SMOTE to training fold only (val/test stay untouched)...")
    smote = SMOTE(random_state=config.RANDOM_STATE, sampling_strategy=0.15)
    X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train)
    print(f"  resampled train fraud rate: {np.mean(y_train_res)*100:.2f}%  "
          f"({len(y_train_res):,} rows)")

    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train_res, y_train_res)
        elapsed = time.time() - start

        val_proba = model.predict_proba(X_val_t)[:, 1]
        pr_auc = average_precision_score(y_val, val_proba)
        roc_auc = roc_auc_score(y_val, val_proba)

        results[name] = {"pr_auc": pr_auc, "roc_auc": roc_auc, "train_seconds": round(elapsed, 1)}
        fitted_models[name] = model
        print(f"  {name}: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  ({elapsed:.1f}s)")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model = fitted_models[best_name]
    print(f"\nBest model by validation PR-AUC: {best_name} "
          f"(PR-AUC={results[best_name]['pr_auc']:.4f})")

    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    metadata = {
        "best_model": best_name,
        "all_results": results,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_fraud_rate": float(y_train.mean()),
        "smote_sampling_strategy": 0.15,
        "feature_columns": list(X_train.columns),
        "random_state": config.RANDOM_STATE,
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {config.MODEL_PATH}")
    print(f"Saved preprocessor to {config.PREPROCESSOR_PATH}")
    print(f"Saved metadata to {config.METADATA_PATH}")
    print("\nRun `python -m src.evaluate` next to get full test-set metrics and plots.")


if __name__ == "__main__":
    main()
