"""
evaluate.py
-----------
Loads the saved model + preprocessor, scores the held-out test set
(never touched during training or model selection), and produces:
  - classification report at default (0.5) threshold
  - a cost-based optimal threshold (fraud misses are far costlier than
    false alarms, so 0.5 is rarely the right operating point)
  - confusion matrix, PR curve, and ROC curve plots
  - a markdown report summarizing everything

Run:
    python -m src.evaluate
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src import config
from src.preprocess import load_and_prepare

# --- Business cost assumptions -------------------------------------------
# Tune these to your actual business: cost of letting fraud through
# vs. cost of a false alarm (e.g. blocked legitimate transaction,
# customer friction, manual review labor).
COST_FALSE_NEGATIVE = 100.0  # missing a fraud costs, say, $100 on average
COST_FALSE_POSITIVE = 5.0    # a false alarm costs ~$5 (review labor / friction)


def find_cost_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_cost = 0.5, float("inf")
    costs = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
        costs.append(cost)
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold, best_cost, thresholds, costs


def main():
    print("Loading model, preprocessor, and test data...")
    model = joblib.load(config.MODEL_PATH)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    _, _, X_test, _, _, y_test = load_and_prepare()

    X_test_t = preprocessor.transform(X_test)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    # --- Default threshold metrics ---
    y_pred_default = (y_proba >= config.DEFAULT_THRESHOLD).astype(int)
    report_default = classification_report(y_test, y_pred_default, target_names=["legit", "fraud"])
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nTest set: {len(y_test):,} transactions, {y_test.sum():,} fraud "
          f"({y_test.mean()*100:.3f}%)")
    print(f"PR-AUC: {pr_auc:.4f}   ROC-AUC: {roc_auc:.4f}")
    print(f"\n--- Classification report @ threshold=0.5 ---\n{report_default}")

    # --- Cost-optimal threshold ---
    best_t, best_cost, thresholds, costs = find_cost_optimal_threshold(y_test, y_proba)
    y_pred_opt = (y_proba >= best_t).astype(int)
    report_opt = classification_report(y_test, y_pred_opt, target_names=["legit", "fraud"])
    print(f"\n--- Cost-optimal threshold: {best_t:.2f} "
          f"(assumes FN cost=${COST_FALSE_NEGATIVE:.0f}, FP cost=${COST_FALSE_POSITIVE:.0f}) ---")
    print(f"Estimated total cost at this threshold: ${best_cost:,.0f} "
          f"(vs ${costs[49]:,.0f} at threshold=0.5)")
    print(report_opt)

    # --- Plots ---
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_opt, display_labels=["legit", "fraud"], ax=axes[0, 0], cmap="Blues"
    )
    axes[0, 0].set_title(f"Confusion Matrix (threshold={best_t:.2f})")

    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[0, 1], name="model")
    axes[0, 1].set_title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1, 0], name="model")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 0].set_title(f"ROC Curve (AUC={roc_auc:.3f})")

    axes[1, 1].plot(thresholds, costs)
    axes[1, 1].axvline(best_t, color="red", linestyle="--", label=f"optimal t={best_t:.2f}")
    axes[1, 1].set_xlabel("Decision threshold")
    axes[1, 1].set_ylabel("Estimated business cost ($)")
    axes[1, 1].set_title("Cost vs. Threshold")
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = config.REPORTS_DIR / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=140)
    print(f"\nSaved plots to {plot_path}")

    # --- Markdown report ---
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    md = f"""# Fraud Detection Model — Evaluation Report

**Model:** {metadata['best_model']}
**Test set size:** {len(y_test):,} transactions ({y_test.sum():,} fraud, {y_test.mean()*100:.3f}%)

## Headline metrics
| Metric | Value |
|---|---|
| PR-AUC | {pr_auc:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## Classification report @ threshold = 0.50 (default)
```
{report_default}
```

## Classification report @ threshold = {best_t:.2f} (cost-optimal)
Assumes an average missed-fraud cost of ${COST_FALSE_NEGATIVE:.0f} and an average
false-alarm cost of ${COST_FALSE_POSITIVE:.0f}. Adjust `COST_FALSE_NEGATIVE` /
`COST_FALSE_POSITIVE` in `src/evaluate.py` to match your real business costs.

```
{report_opt}
```

Estimated total cost at optimal threshold: **${best_cost:,.0f}**
Estimated total cost at default 0.5 threshold: **${costs[49]:,.0f}**

## Model comparison (validation set, from training run)
| Model | PR-AUC | ROC-AUC | Train time (s) |
|---|---|---|---|
"""
    for name, r in metadata["all_results"].items():
        md += f"| {name} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['train_seconds']} |\n"

    md += "\n![Evaluation plots](evaluation_plots.png)\n"

    report_path = config.REPORTS_DIR / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"Saved markdown report to {report_path}")


if __name__ == "__main__":
    main()
