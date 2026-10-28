"""
Stage 3 — exploratory analysis: baseline approval-rate disparities,
predictor distributions, missingness, and correlation structure.
Saves figures/tables to reports/figures/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLEAN_FILE, FIGURES, NUMERIC_FEATURES, TARGET

sns.set_theme(style="whitegrid")


def approval_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    return df.groupby(group_col)[TARGET].mean().sort_values(ascending=False)


def plot_approval_by_group(df: pd.DataFrame, group_col: str):
    rates = approval_rate_by_group(df, group_col)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Approval rate")
    ax.set_title(f"Approval rate by {group_col}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"approval_rate_by_{group_col}.png", dpi=150)
    plt.close(fig)
    rates.to_csv(FIGURES / f"approval_rate_by_{group_col}.csv")


def plot_minority_tract_quartile(df: pd.DataFrame):
    if "tract_minority_population_percent" not in df.columns:
        return
    d = df.dropna(subset=["tract_minority_population_percent"]).copy()
    d["minority_pct_quartile"] = pd.qcut(
        d["tract_minority_population_percent"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )
    plot_approval_by_group(d.rename(columns={"minority_pct_quartile": "tract_minority_quartile"}),
                            "tract_minority_quartile")


def plot_distributions(df: pd.DataFrame):
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, hue=TARGET, kde=False, bins=30, ax=axes[i],
                     element="step", stat="density", common_norm=False)
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.savefig(FIGURES / "predictor_distributions_by_outcome.png", dpi=150)
    plt.close(fig)


def plot_missingness(df: pd.DataFrame):
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    miss_by_group = (
        df[cols].isna().assign(derived_race=df["derived_race"])
        .groupby("derived_race").mean()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(miss_by_group, cmap="Reds", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Share missing, by field and derived_race")
    fig.tight_layout()
    fig.savefig(FIGURES / "missingness_heatmap.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame):
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Predictor correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    print(f"Loaded {len(df):,} cleaned records.")

    for col in ["derived_race", "derived_ethnicity", "derived_sex"]:
        if col in df.columns:
            plot_approval_by_group(df, col)

    plot_minority_tract_quartile(df)
    plot_distributions(df)
    plot_missingness(df)
    plot_correlation(df)

    print(f"EDA figures/tables written to {FIGURES}")
