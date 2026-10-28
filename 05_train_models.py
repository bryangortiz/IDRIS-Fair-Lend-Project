"""
Stage 5 — fit baseline models (untuned / lightly regularized), for
comparison against the stage-7 optimized versions.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import MODELS_DIR, RANDOM_STATE

MODEL_SPECS = {
    "logistic": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "lasso_logistic": LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, max_iter=2000, random_state=RANDOM_STATE
    ),
    "ridge_logistic": LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, random_state=RANDOM_STATE
    ),
    "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


if __name__ == "__main__":
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    split = joblib.load(MODELS_DIR / "train_test_split.joblib")
    X_train, y_train = split["X_train"], split["y_train"]

    fitted = {}
    for name, estimator in MODEL_SPECS.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        print(f"Fit {name}")

    joblib.dump(fitted, MODELS_DIR / "baseline_models.joblib")
    print(f"Saved {len(fitted)} baseline models -> {MODELS_DIR / 'baseline_models.joblib'}")
