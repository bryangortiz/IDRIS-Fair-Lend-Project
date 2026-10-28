"""
Stage 1 — download HMDA LAR records for Essex County, NJ from the CFPB
HMDA Platform API (https://cfpb.github.io/hmda-platform/).

The API occasionally changes shape/parameters. If this request fails,
fall back to the manual HMDA Data Browser (ffiec.cfpb.gov/data-browser):
filter State=New Jersey, County=Essex County, the years in
config.ACTIVITY_YEARS, export CSV, and save it to config.RAW_FILE.
"""

import sys

import pandas as pd
import requests

from config import ACTIVITY_YEARS, COUNTY_FIPS, HMDA_COLUMNS, RAW_FILE

API_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"


def download() -> pd.DataFrame:
    params = {
        "counties": COUNTY_FIPS,
        "years": ",".join(ACTIVITY_YEARS),
    }
    print(f"Requesting HMDA data for county {COUNTY_FIPS}, years {ACTIVITY_YEARS} ...")
    resp = requests.get(API_URL, params=params, timeout=180)
    resp.raise_for_status()

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_bytes(resp.content)
    df = pd.read_csv(RAW_FILE, low_memory=False)
    print(f"Downloaded {len(df):,} raw records -> {RAW_FILE}")
    return df


if __name__ == "__main__":
    try:
        df = download()
    except Exception as exc:  # noqa: BLE001 - top-level script, report and exit
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Fall back to the manual HMDA Data Browser export "
            "(ffiec.cfpb.gov/data-browser) -> save CSV to "
            f"{RAW_FILE}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = [c for c in HMDA_COLUMNS if c not in df.columns]
    if missing:
        print(f"Note: {len(missing)} expected columns not present in this extract: {missing}")
