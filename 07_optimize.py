"""
Stage 7 — optimization. This is the core "AI + optimization" stage of the
project:

  7a. Hyperparameter optimization: grid/randomized search over each
      model's regularization strength / tree complexity, scored on AUC
      under stratified k-fold cross-validation.

  7b. Fairness-constrained optimization: (i) a global-threshold sweep
      that traces out accuracy vs. max subgroup FPR gap for each tuned
      model (a Pareto-style frontier), and (ii) fairlearn's
      ThresholdOptimizer, which fits group-specific decision thresholds
      to explicitly satisfy an equalized-odds constraint, so the
      accuracy cost of *actually* closing the disparity gap can be
      measured directly.

Outputs:
  - data/processed/models/tuned_models.joblib
  - reports/optimization_results.csv         (grid/random search summary)
  - reports/pareto_threshold_sweep.csv       (accuracy vs. disparity, all thresholds)
  - reports/figures/pareto_frontier.png
  - reports/fairness_constrained_results.csv (ThresholdOptimizer before/after)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import FIGURES, MODELS_DIR, RANDOM_STATE, REPORTS
from fairness_utils import worst_gap_across_demo

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def search_specs(preprocessor):
    """(name, search_estimator) pairs — GridSearchCV for small grids,
    RandomizedSearchCV for the larger random-forest grid."""
    specs = {}

    logistic_pipe = Pipeline([("prep", preprocessor),
                               ("clf", LogisticRegression(max_iter=2000, solver="liblinear",
                                                           random_state=RANDOM_STATE))])
    specs["logistic"] = GridSearchCV(
        logistic_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"]},
        scoring="roc_auc", cv=CV, n_jobs=-1,
    )
    specs["lasso_logistic"] = GridSearchCV(
        logistic_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1"]},
        scoring="roc_auc", cv=CV, n_jobs=-1,
    )
    specs["ridge_logistic"] = GridSearchCV(
        logistic_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"]},
        scoring="roc_auc", cv=CV, n_jobs=-1,
    )

    tree_pipe = Pipeline([("prep", preprocessor),
                           ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    specs["decision_tree"] = GridSearchCV(
        tree_pipe,
        {"clf__max_depth": [3, 5, 7, 10, None], "clf__min_samples_leaf": [1, 5, 10, 20]},
        scoring="roc_auc", cv=CV, n_jobs=-1,
    )

    rf_pipe = Pipeline([("prep", preprocessor),
                         ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    specs["random_forest"] = RandomizedSearchCV(
        rf_pipe,
        {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__max_features": ["sqrt", "log2"],
            "clf__min_samples_leaf": [1, 2, 5, 10],
        },
        n_iter=20, scoring="roc_auc", cv=CV, random_state=RANDOM_STATE, n_jobs=-1,
    )

    return specs


def run_hyperparameter_search(preprocessor, X_train, y_train):
    results = []
    tuned_models = {}
    for name, search in search_specs(preprocessor).items():
        print(f"Tuning {name} ...")
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        results.append({
            "model": name,
            "best_cv_auc": search.best_score_,
            "best_params": search.best_params_,
        })
        print(f"  best CV AUC={search.best_score_:.4f}  params={search.best_params_}")
    return tuned_models, pd.DataFrame(results)


def threshold_sweep(name, pipe, X_test, y_test, demo_test, thresholds=np.linspace(0.05, 0.95, 19)):
    proba = pipe.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "model": name,
            "threshold": t,
            "accuracy": accuracy_score(y_test, pred),
            "max_subgroup_fpr_gap": worst_gap_across_demo(y_test, pred, demo_test, "fpr"),
            "max_subgroup_fnr_gap": worst_gap_across_demo(y_test, pred, demo_test, "fnr"),
        })
    return pd.DataFrame(rows)


def plot_pareto_frontier(sweep_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, grp in sweep_df.groupby("model"):
        grp = grp.sort_values("threshold")
        ax.plot(grp["max_subgroup_fpr_gap"], grp["accuracy"], marker="o", label=name)
    ax.set_xlabel("Max subgroup FPR gap (disparity)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. disparity across decision thresholds\n(closer to top-left is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pareto_frontier.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    split = joblib.load(MODELS_DIR / "train_test_split.joblib")
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    demo_train, demo_test = split["demo_train"], split["demo_test"]

    # 7a. Hyperparameter optimization.
    tuned_models, search_summary = run_hyperparameter_search(preprocessor, X_train, y_train)
    search_summary.to_csv(REPORTS / "optimization_results.csv", index=False)
    joblib.dump(tuned_models, MODELS_DIR / "tuned_models.joblib")
    print(f"Saved tuned models -> {MODELS_DIR / 'tuned_models.joblib'}")

    # 7b(i). Global-threshold Pareto sweep.
    sweep_frames = [threshold_sweep(name, pipe, X_test, y_test, demo_test)
                     for name, pipe in tuned_models.items()]
    sweep_df = pd.concat(sweep_frames, ignore_index=True)
    sweep_df.to_csv(REPORTS / "pareto_threshold_sweep.csv", index=False)
    plot_pareto_frontier(sweep_df)
    print(f"Saved threshold sweep + Pareto plot -> {FIGURES / 'pareto_frontier.png'}")

    # 7b(ii). fairlearn ThresholdOptimizer (equalized odds), using the
    # first demographic column (e.g. derived_race) as the sensitive
    # feature for both fitting and evaluating the group-specific thresholds.
    try:
        from fairlearn.postprocessing import ThresholdOptimizer

        constrained_rows = []
        sensitive_col = demo_test.columns[0]
        for name, pipe in tuned_models.items():
            opt = ThresholdOptimizer(
                estimator=pipe, constraints="equalized_odds",
                predict_method="predict_proba", prefit=True,
            )
            opt.fit(X_train, y_train, sensitive_features=demo_train[sensitive_col])
            constrained_pred = opt.predict(X_test, sensitive_features=demo_test[sensitive_col],
                                            random_state=RANDOM_STATE)

            base_pred = (pipe.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
            constrained_rows.append({
                "model": name, "variant": "unconstrained_0.5_threshold",
                "accuracy": accuracy_score(y_test, base_pred),
                "max_subgroup_fpr_gap": worst_gap_across_demo(y_test, base_pred, demo_test, "fpr"),
                "max_subgroup_fnr_gap": worst_gap_across_demo(y_test, base_pred, demo_test, "fnr"),
            })
            constrained_rows.append({
                "model": name, "variant": "equalized_odds_thresholdoptimizer",
                "accuracy": accuracy_score(y_test, constrained_pred),
                "max_subgroup_fpr_gap": worst_gap_across_demo(y_test, constrained_pred, demo_test, "fpr"),
                "max_subgroup_fnr_gap": worst_gap_across_demo(y_test, constrained_pred, demo_test, "fnr"),
            })
            print(f"{name}: equalized-odds thresholding done.")

        pd.DataFrame(constrained_rows).to_csv(REPORTS / "fairness_constrained_results.csv", index=False)
        print(f"Saved fairness-constrained comparison -> {REPORTS / 'fairness_constrained_results.csv'}")
    except ImportError:
        print("fairlearn not installed (pip install fairlearn) — skipped ThresholdOptimizer step.")
