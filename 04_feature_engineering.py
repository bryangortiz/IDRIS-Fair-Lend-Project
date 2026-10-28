"""
Stage 4 — build the modeling matrix.

Demographic columns are split off into a separate frame for subgroup
evaluation later; they are never included in X.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from config import (
    CATEGORICAL_FEATURES,
    CLEAN_FILE,
    DEMOGRAPHIC_COLS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_FILE, low_memory=False)

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    demo_cols = [c for c in DEMOGRAPHIC_COLS if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET]
    demo = df[demo_cols]

    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demo, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(
        {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "demo_train": demo_train, "demo_test": demo_test,
            "feature_cols": feature_cols,
        },
        MODELS_DIR / "train_test_split.joblib",
    )

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Train approval rate: {y_train.mean():.1%} | Test approval rate: {y_test.mean():.1%}")
    print(f"Saved preprocessor + split -> {MODELS_DIR}")
