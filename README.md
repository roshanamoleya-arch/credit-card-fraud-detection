# Credit Card Fraud Detection — End-to-End ML Project

[![CI](https://github.com/roshanamoleya-arch/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/roshanamoleya-arch/credit-card-fraud-detection/actions/workflows/ci.yml)

A complete, production-style machine learning project for detecting
fraudulent credit card transactions. Covers the full lifecycle: data
generation, preprocessing/feature engineering, model training with
class-imbalance handling, evaluation with business-cost-aware
threshold tuning, and a reusable inference module — plus tests.

## Why this project matters (real-world framing)

Fraud detection is a canonical hard ML problem:
- **Severe class imbalance** — fraud is typically <1% of transactions.
- **Asymmetric costs** — missing a fraud is far more expensive than a
  false alarm, so accuracy is a misleading metric and the decision
  threshold must be tuned deliberately.
- **Concept drift** — fraud patterns evolve, so the pipeline is built
  to be retrained on fresh data, not just run once.
- **Latency constraints** — scoring has to be fast at inference time,
  which is why heavy feature engineering is avoided.

This project is built to reflect those constraints rather than just
maximize a leaderboard metric.

## Project structure

```
fraud_detection_project/
├── data/
│   ├── generate_data.py       # synthetic transaction data generator
│   └── transactions.csv       # generated dataset (created by running the script)
├── src/
│   ├── config.py               # paths, feature lists, constants
│   ├── preprocess.py           # feature engineering + train/val/test split
│   ├── train.py                 # trains & compares 3 models, saves the best
│   ├── evaluate.py              # test-set metrics, plots, cost-optimal threshold
│   └── predict.py               # FraudPredictor class + CLI for scoring new data
├── models/
│   ├── fraud_model.joblib       # trained model (created by train.py)
│   ├── preprocessor.joblib      # fitted sklearn ColumnTransformer
│   └── model_metadata.json      # training run metadata / model comparison
├── reports/
│   ├── evaluation_report.md     # generated evaluation report
│   └── evaluation_plots.png     # confusion matrix, PR/ROC curves, cost curve
├── tests/
│   └── test_pipeline.py         # pytest suite covering every stage
├── requirements.txt
└── README.md
```

## Dataset

Since a live proprietary transaction feed isn't available in this
environment, `data/generate_data.py` synthesizes a realistic dataset
using the same feature schema as well-known public fraud datasets:

| Feature | Description |
|---|---|
| `amount` | Transaction amount ($) |
| `hour_of_day` | Hour the transaction occurred |
| `distance_from_home` | Distance (km) from the cardholder's home |
| `distance_from_last_transaction` | Distance from the previous transaction |
| `ratio_to_median_purchase_price` | How unusual the amount is vs. the account's typical spend |
| `repeat_retailer` | Whether this retailer has been used before |
| `used_chip` / `used_pin_number` | Card-present verification method |
| `online_order` | Card-not-present online transaction |
| `card_present` | Physical card presented at time of purchase |
| `merchant_category` | Merchant type (grocery, electronics, travel, ...) |
| `is_fraud` | Target label |

Fraudulent transactions are generated with realistically shifted
distributions (higher amounts, farther distances, more likely to be
online/card-not-present, more common at night) plus label noise, so
the classification problem is genuinely hard — not trivially separable.

To swap in a real dataset (e.g. your own transaction logs, or a
Kaggle dataset), just replace `data/transactions.csv` with a file that
has the same columns and re-run the pipeline.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (skip if you're supplying your own transactions.csv)
python data/generate_data.py --n_samples 150000 --fraud_rate 0.006

# 2. Train and select the best model
python -m src.train

# 3. Evaluate on the held-out test set (produces reports/evaluation_report.md)
python -m src.evaluate

# 4. Score new transactions
python -m src.predict --input new_transactions.csv --output scored.csv --threshold 0.75

# 5. Run tests
pytest tests/ -v
```

## Modeling approach

**Models compared:** Logistic Regression (interpretable baseline),
Random Forest, and XGBoost — representing linear, bagging, and
boosting approaches respectively.

**Class imbalance:** SMOTE oversampling is applied to the *training
fold only* (never to validation/test, which would leak information
and produce falsely optimistic metrics), bringing the fraud rate from
~0.8% to ~13%. Models are also trained with `class_weight="balanced"`
where supported.

**Model selection metric:** PR-AUC (average precision), not accuracy
or ROC-AUC. With <1% positive rate, a model that predicts "not fraud"
for everything scores 99%+ accuracy while being useless. PR-AUC
focuses on how well the model ranks and precision/recall trade off
specifically for the rare positive class.

**Threshold selection:** `evaluate.py` doesn't just report metrics at
the default 0.5 threshold — it sweeps thresholds and picks the one
that minimizes expected business cost, using configurable costs for
false negatives (missed fraud) vs. false positives (false alarms).
This is the step most tutorials skip but that matters most in
production: the "right" threshold depends on your business, not on
a generic default.

## Extending this to a real deployment

- **Real data:** replace the synthetic generator with your actual
  transaction feed; keep the same column schema or update
  `src/config.py`.
- **Retraining cadence:** fraud patterns drift — wrap `train.py` in a
  scheduled job (e.g. weekly) and monitor PR-AUC on fresh validation
  data to detect degradation.
- **Serving:** `FraudPredictor` in `src/predict.py` is written to be
  imported directly into a web service (e.g. FastAPI) for real-time
  scoring — load it once at startup, call `.predict_one()` per request.
- **Monitoring:** track prediction distribution, flagged-transaction
  rate, and (once ground truth arrives) precision/recall in
  production to catch drift early.
- **Explainability:** for a compliance-sensitive domain like finance,
  consider adding SHAP values so flagged transactions can be
  explained to fraud analysts and, if needed, to regulators.
