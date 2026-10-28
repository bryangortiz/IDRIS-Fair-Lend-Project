"""
Stage 8 — final fairness report: regenerate subgroup tables/plots using
the stage-7 tuned models (and, where available, the equalized-odds
thresholded predictions), for direct before/after comparison against the
stage-6 baseline numbers in the poster and written report.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from config import FIGURES, MODELS_DIR, REPORTS
from fairness_utils import subgroup_error_rates


def plot_subgroup_rates(sub: pd.DataFrame, title: str, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(sub))
    width = 0.35
    ax.bar([i - width / 2 for i in x], sub["fpr"], width, label="False positive rate")
    ax.bar([i + width / 2 for i in x], sub["fnr"], width, label="False negative rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["group"], rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    split = joblib.load(MODELS_DIR / "train_test_split.joblib")
    tuned_models = joblib.load(MODELS_DIR / "tuned_models.joblib")
    X_test, y_test, demo_test = split["X_test"], split["y_test"], split["demo_test"]

    for name, pipe in tuned_models.items():
        pred = (pipe.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        for demo_col in demo_test.columns:
            sub = subgroup_error_rates(y_test, pred, demo_test[demo_col])
            sub.to_csv(REPORTS / f"final_subgroup_errors_{name}_{demo_col}.csv", index=False)
            plot_subgroup_rates(
                sub, f"{name} (tuned) — error rates by {demo_col}",
                FIGURES / f"final_subgroup_errors_{name}_{demo_col}.png",
            )

    print(f"Final subgroup tables/plots written to {REPORTS} and {FIGURES}")
    print("Compare these against stage 6's baseline_model_comparison.csv and")
    print("subgroup_errors_* files, and against fairness_constrained_results.csv")
    print("from stage 7, to show the before/after effect of optimization.")
