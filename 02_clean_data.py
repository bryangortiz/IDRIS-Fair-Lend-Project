"""
Stage 2 — clean the raw HMDA extract.

See README.md "Step 2" for the rationale behind each transformation.
"""

import numpy as np
import pandas as pd

from config import (
    APPROVED_CODES,
    CLEAN_FILE,
    DENIED_CODES,
    RAW_FILE,
    TARGET,
)

# HMDA reports debt-to-income as a bucketed string, not a raw number.
# Map to bucket midpoints so it can be used as a numeric feature.
DTI_BUCKET_MIDPOINTS = {
    "<20%": 15.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "36%": 36.0,
    "37%": 37.0,
    "38%": 38.0,
    "39%": 39.0,
    "40%": 40.0,
    "41%": 41.0,
    "42%": 42.0,
    "43%": 43.0,
    "44%": 44.0,
    "45%": 45.0,
    "46%": 46.0,
    "47%": 47.0,
    "48%": 48.0,
    "49%": 49.0,
    "50%-60%": 55.0,
    ">60%": 65.0,
}

EXEMPT_TOKENS = {"Exempt", "exempt", "NA", ""}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_FILE, low_memory=False)


def coerce_numeric(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    series = series.where(~series.isin(EXEMPT_TOKENS), np.nan)
    return pd.to_numeric(series, errors="coerce")


def winsorize(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Define the outcome; drop rows that aren't a completed approve/deny decision.
    df = df[df["action_taken"].isin(APPROVED_CODES | DENIED_CODES)].copy()
    df[TARGET] = df["action_taken"].isin(APPROVED_CODES).astype(int)

    # 2 & 3. Sentinel-code handling + numeric coercion for HMDA fields that
    # arrive as strings ("Exempt", blank, or bucketed).
    numeric_raw_cols = [
        "loan_amount", "loan_to_value_ratio", "interest_rate", "rate_spread",
        "loan_term", "property_value", "total_units", "income",
        "tract_minority_population_percent", "tract_to_msa_income_percentage",
        "tract_owner_occupied_units",
    ]
    for col in numeric_raw_cols:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])

    if "debt_to_income_ratio" in df.columns:
        dti_raw = df["debt_to_income_ratio"].astype(str).str.strip()
        df["debt_to_income_ratio"] = dti_raw.map(DTI_BUCKET_MIDPOINTS)
        df["debt_to_income_ratio"] = df["debt_to_income_ratio"].fillna(
            coerce_numeric(dti_raw)
        )

    # 4. Winsorize known long-tailed / outlier-prone fields.
    for col in ["loan_to_value_ratio", "debt_to_income_ratio", "income", "loan_amount"]:
        if col in df.columns:
            df[col] = winsorize(df[col])

    # 5. Demographics: derived_race / derived_ethnicity / derived_sex are
    # already CFPB-collapsed single-category fields — use them as-is.

    # 6. Document missingness by subgroup before dropping (limitations note).
    key_financial_cols = [c for c in ["income", "loan_amount", "debt_to_income_ratio",
                                       "loan_to_value_ratio"] if c in df.columns]
    if key_financial_cols:
        missing_by_group = (
            df.assign(any_missing=df[key_financial_cols].isna().any(axis=1))
            .groupby("derived_race")["any_missing"]
            .mean()
            .sort_values(ascending=False)
        )
        print("Share of rows with missing key financial fields, by derived_race:")
        print(missing_by_group.to_string())
        missing_by_group.to_csv(CLEAN_FILE.parent / "missingness_by_subgroup.csv")

    df = df.dropna(subset=[TARGET])
    df = df.dropna(subset=key_financial_cols, how="all")

    # 7. De-duplicate.
    dedupe_cols = [c for c in ["lei", "activity_year", "census_tract",
                                "loan_amount", "income"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_cols)
    print(f"Dropped {before - len(df):,} duplicate rows.")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    print(f"Loaded {len(raw):,} raw records.")
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_FILE, index=False)
    print(f"Wrote {len(cleaned):,} cleaned records -> {CLEAN_FILE}")
    print(f"Approval rate overall: {cleaned['approved'].mean():.1%}")
