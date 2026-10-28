"""Shared subgroup-metric helpers used by stages 6, 7, and 8."""

import numpy as np
import pandas as pd


def subgroup_error_rates(y_true: pd.Series, y_pred, group_series: pd.Series) -> pd.DataFrame:
    """False-positive / false-negative rate for each value of group_series."""
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    rows = []
    for group_val, idx in group_series.groupby(group_series).groups.items():
        yt = y_true.loc[idx]
        yp = y_pred.loc[idx]
        negatives = yt == 0
        positives = yt == 1
        fpr = (yp[negatives] == 1).mean() if negatives.sum() else np.nan
        fnr = (yp[positives] == 0).mean() if positives.sum() else np.nan
        rows.append({"group": group_val, "n": len(idx), "fpr": fpr, "fnr": fnr})
    return pd.DataFrame(rows).sort_values("group")


def max_gap(y_true: pd.Series, y_pred, group_series: pd.Series, metric: str = "fpr") -> float:
    sub = subgroup_error_rates(y_true, y_pred, group_series)
    if sub[metric].notna().sum() < 2:
        return np.nan
    return float(sub[metric].max() - sub[metric].min())


def worst_gap_across_demo(y_true, y_pred, demo_df: pd.DataFrame, metric: str = "fpr") -> float:
    gaps = [max_gap(y_true, y_pred, demo_df[col], metric) for col in demo_df.columns]
    gaps = [g for g in gaps if not np.isnan(g)]
    return max(gaps) if gaps else np.nan
