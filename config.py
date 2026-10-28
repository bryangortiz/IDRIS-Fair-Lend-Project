"""Shared constants for the FairLend Newark pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = DATA_PROCESSED / "models"

for d in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_RAW / "hmda_essex_raw.csv"
CLEAN_FILE = DATA_PROCESSED / "hmda_essex_clean.csv"
FEATURES_FILE = DATA_PROCESSED / "hmda_essex_features.parquet"

# Essex County, NJ FIPS code (state 34 = NJ, county 013 = Essex)
STATE = "NJ"
COUNTY_FIPS = "34013"
ACTIVITY_YEARS = ["2022", "2023", "2024"]

# action_taken: 1=Originated, 2=Approved not accepted, 3=Denied,
# 4=Withdrawn, 5=Closed incomplete, 6=Purchased loan, 7/8=Preapproval only
APPROVED_CODES = {1, 2}
DENIED_CODES = {3}

HMDA_COLUMNS = [
    "activity_year", "lei", "state_code", "county_code", "census_tract",
    "action_taken", "rate_spread",
    "loan_amount", "loan_to_value_ratio", "interest_rate", "loan_term",
    "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "property_value", "total_units", "construction_method",
    "income", "debt_to_income_ratio", "applicant_credit_score_type",
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
    "tract_minority_population_percent", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
]

# Columns that must NEVER enter the model feature matrix (protected /
# audit-only attributes). Kept in the cleaned file for subgroup evaluation.
DEMOGRAPHIC_COLS = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]

ID_COLS = ["lei", "activity_year", "census_tract", "county_code", "state_code"]

TARGET = "approved"  # 1 = approved, 0 = denied

NUMERIC_FEATURES = [
    "loan_amount", "loan_to_value_ratio", "interest_rate", "loan_term",
    "property_value", "total_units", "income", "debt_to_income_ratio",
    "tract_minority_population_percent", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
]

CATEGORICAL_FEATURES = [
    "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "construction_method", "applicant_credit_score_type",
]

RANDOM_STATE = 42
