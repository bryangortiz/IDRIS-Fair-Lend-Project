"""
Stage 6 — evaluate baseline models: AUC, calibration, precision/recall,
and subgroup false-positive/false-negative rates. Produces the summary
table that is the core "accuracy-interpretability-disparity" comparison.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FIGURES, MODELS_DIR, REPORTS
from fairness_utils import subgroup_error_rates


def evaluate_model(name, pipe, X_test, y_test, demo_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model": name,
        "auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

    max_fpr_gap, max_fnr_gap = np.nan, np.nan
    for demo_col in demo_test.columns:
        sub = subgroup_error_rates(y_test, pred, demo_test[demo_col])
        sub.to_csv(REPORTS / f"subgroup_errors_{name}_{demo_col}.csv", index=False)
        if sub["fpr"].notna().any():
            gap = sub["fpr"].max() - sub["fpr"].min()
            max_fpr_gap = np.nanmax([max_fpr_gap, gap]) if not np.isnan(max_fpr_gap) else gap
        if sub["fnr"].notna().any():
            gap = sub["fnr"].max() - sub["fnr"].min()
            max_fnr_gap = np.nanmax([max_fnr_gap, gap]) if not np.isnan(max_fnr_gap) else gap

    metrics["max_subgroup_fpr_gap"] = max_fpr_gap
    metrics["max_subgroup_fnr_gap"] = max_fnr_gap

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed approval rate")
    ax.set_title(f"Calibration curve — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"calibration_{name}.png", dpi=150)
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    split = joblib.load(MODELS_DIR / "train_test_split.joblib")
    models = joblib.load(MODELS_DIR / "baseline_models.joblib")
    X_test, y_test, demo_test = split["X_test"], split["y_test"], split["demo_test"]

    results = [evaluate_model(name, pipe, X_test, y_test, demo_test) for name, pipe in models.items()]
    summary = pd.DataFrame(results).sort_values("auc", ascending=False)
    summary.to_csv(REPORTS / "baseline_model_comparison.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved comparison table -> {REPORTS / 'baseline_model_comparison.csv'}")
