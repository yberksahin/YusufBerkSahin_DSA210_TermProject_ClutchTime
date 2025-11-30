"""
NBA Last 3 Minutes - Hypothesis Testing (liveData v3)
Author: <Your Name>
DSA 210 - Fall 2025-2026

This script runs several hypothesis tests on the processed critical moments
dataset produced by exploratory_data_analysis.py.

All tests are based on REAL NBA liveData play-by-play events
(last 3 minutes of 4Q + all overtimes).
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "figures"
ALPHA = 0.05  # significance level


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def load_processed_data() -> pd.DataFrame | None:
    """
    Load processed_critical_moments.csv created by the EDA script.
    """
    path = os.path.join(PROCESSED_DIR, "processed_critical_moments.csv")
    if not os.path.exists(path):
        print("processed_critical_moments.csv not found.")
        print("Please run exploratory_data_analysis.py first.")
        return None

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} critical events from {path}")
    return df


def ensure_figures_dir() -> None:
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Hypothesis 1: 3PT vs 2PT in last 30 seconds when trailing by 3+
# ---------------------------------------------------------------------


def hypothesis_1_three_point_vs_two_point(df: pd.DataFrame) -> str:
    """
    H1: In the last 30 seconds when the HOME team is trailing by 3+ points,
        do 3PT attempts have a different success rate than 2PT attempts?

    We only look at shot events and use shotResult == 'Made' as success.
    """

    print("\n============================================================")
    print("Hypothesis 1: 3PT vs 2PT in late-game comeback context")
    print("============================================================")

    required_cols = {"time_remaining", "score_diff", "actionType"}
    if not required_cols.issubset(df.columns):
        print("Required columns missing; skipping Hypothesis 1.")
        return "Hypothesis 1: SKIPPED (missing columns)"

    # Filter for last 30 seconds, home trailing by 3+ points
    late = df[
        (df["time_remaining"] <= 30)
        & (df["score_diff"] <= -3)  # home - away <= -3 (home down by 3+)
        & (df["actionType"].isin(["shot", "freethrow", "freeThrow"]))
    ].copy()

    if late.empty:
        print("No matching shot events for Hypothesis 1.")
        return "Hypothesis 1: SKIPPED (no matching events)"

    # Normalize fields
    late["shotType"] = late.get("shotType", "").fillna("").astype(str).str.upper()
    late["shotResult"] = (
        late.get("shotResult", "").fillna("").astype(str).str.capitalize()
    )

    def categorize_shot(row) -> str:
        t = row["shotType"]
        a = row["actionType"].lower()
        if "3" in t:
            return "3PT"
        if a in ["freethrow", "freethrow", "freeThrow"] or "free" in t:
            return "FT"
        return "2PT"

    late["shot_category"] = late.apply(categorize_shot, axis=1)

    # Focus on 3PT vs 2PT only
    late = late[late["shot_category"].isin(["3PT", "2PT"])].copy()
    if late["shot_category"].nunique() < 2:
        print("Not enough variation between 3PT and 2PT shots.")
        return "Hypothesis 1: SKIPPED (need both 3PT and 2PT)"

    late["made"] = (late["shotResult"] == "Made").astype(int)

    # Success rates
    grouped = late.groupby("shot_category")["made"].agg(["mean", "sum", "count"])
    print("\nSuccess rates (last 30s, home down by 3+):")
    print(grouped)

    # Extract groups for t-test
    made_3pt = late[late["shot_category"] == "3PT"]["made"]
    made_2pt = late[late["shot_category"] == "2PT"]["made"]

    if len(made_3pt) < 5 or len(made_2pt) < 5:
        print("Sample sizes too small for a reliable t-test.")
        return "Hypothesis 1: INCONCLUSIVE (small sample)"

    t_stat, p_val = stats.ttest_ind(made_3pt, made_2pt, equal_var=False)

    print(f"\nTwo-sample t-test (3PT vs 2PT made rates):")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

    if p_val < ALPHA:
        conclusion = (
            "REJECT H0: 3PT and 2PT success rates differ significantly "
            "in late-game comeback situations."
        )
    else:
        conclusion = (
            "FAIL TO REJECT H0: No statistically significant difference "
            "between 3PT and 2PT success rates in this sample."
        )

    print(conclusion)
    return f"Hypothesis 1: p={p_val:.4f} → {conclusion}"


# ---------------------------------------------------------------------
# Hypothesis 2: Foul frequency in last 30s vs earlier in clutch time
# ---------------------------------------------------------------------


def hypothesis_2_foul_frequency(df: pd.DataFrame) -> str:
    """
    H2: Are fouls more frequent in the final 30 seconds than in the
        earlier part of clutch time (30–180 seconds)?

    We compare proportions of foul events among all events in two windows:
      - Window A: time_remaining <= 30
      - Window B: 30 < time_remaining <= 180
    """

    print("\n============================================================")
    print("Hypothesis 2: Foul frequency in last 30 seconds")
    print("============================================================")

    if "time_remaining" not in df.columns or "actionType" not in df.columns:
        print("Required columns missing; skipping Hypothesis 2.")
        return "Hypothesis 2: SKIPPED (missing columns)"

    # Tag foul events
    df["is_foul"] = df["actionType"].astype(str).str.lower().eq("foul")

    # Two time windows
    window_a = df[df["time_remaining"] <= 30]
    window_b = df[(df["time_remaining"] > 30) & (df["time_remaining"] <= 180)]

    if window_a.empty or window_b.empty:
        print("Not enough events in one or both windows.")
        return "Hypothesis 2: SKIPPED (insufficient events)"

    fouls_a = window_a["is_foul"].sum()
    total_a = len(window_a)

    fouls_b = window_b["is_foul"].sum()
    total_b = len(window_b)

    print(f"\nWindow A (<=30s): {fouls_a}/{total_a} fouls")
    print(f"Window B (31–180s): {fouls_b}/{total_b} fouls")

    counts = np.array([fouls_a, fouls_b])
    nobs = np.array([total_a, total_b])

    if min(counts) == 0 or min(nobs - counts) == 0:
        print("Some groups have zero fouls or zero non-fouls; z-test is unstable.")
        return "Hypothesis 2: INCONCLUSIVE (degenerate proportions)"

    z_stat, p_val = proportions_ztest(counts, nobs)

    print(f"\nProportions z-test:")
    print(f"z-statistic = {z_stat:.4f}, p-value = {p_val:.4f}")

    if p_val < ALPHA:
        conclusion = (
            "REJECT H0: Foul frequency in the final 30 seconds is "
            "significantly different from earlier clutch time."
        )
    else:
        conclusion = (
            "FAIL TO REJECT H0: No significant difference in foul frequency "
            "between final 30 seconds and earlier clutch time."
        )

    print(conclusion)
    return f"Hypothesis 2: p={p_val:.4f} → {conclusion}"


# ---------------------------------------------------------------------
# Hypothesis 3: Score differential and foul likelihood (score buckets)
# ---------------------------------------------------------------------


def hypothesis_3_score_diff_and_fouls(df: pd.DataFrame) -> str:
    """
    H3: Does the score differential affect the likelihood of committing fouls?

    We group events into score buckets based on score_diff (home - away)
    and test if the proportion of fouls differs across buckets using
    a chi-square test of independence.
    """

    print("\n============================================================")
    print("Hypothesis 3: Score differential vs foul likelihood")
    print("============================================================")

    if "score_diff" not in df.columns or "actionType" not in df.columns:
        print("Required columns missing; skipping Hypothesis 3.")
        return "Hypothesis 3: SKIPPED (missing columns)"

    df["is_foul"] = df["actionType"].astype(str).str.lower().eq("foul")

    df["score_bucket"] = pd.cut(
        df["score_diff"],
        bins=[-20, -6, -3, 0, 3, 6, 20],
        labels=["Home -7+", "Home -4 to -6", "Home -1 to -3",
                "Tied", "Home +1 to +3", "Home +4+"],
    )

    valid = df.dropna(subset=["score_bucket"])
    if valid.empty:
        print("No valid score buckets; skipping.")
        return "Hypothesis 3: SKIPPED (no valid score buckets)"

    contingency = pd.crosstab(valid["score_bucket"], valid["is_foul"])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print("Not enough variation across buckets or foul status.")
        return "Hypothesis 3: INCONCLUSIVE (degenerate crosstab)"

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    print("\nContingency table (score_bucket x is_foul):")
    print(contingency)

    print(f"\nChi-square test:")
    print(f"chi2 = {chi2:.4f}, dof = {dof}, p-value = {p_val:.4f}")

    if p_val < ALPHA:
        conclusion = (
            "REJECT H0: Foul likelihood depends on the score differential bucket."
        )
    else:
        conclusion = (
            "FAIL TO REJECT H0: Foul likelihood does not show a significant "
            "dependence on the score differential bucket in this sample."
        )

    print(conclusion)
    return f"Hypothesis 3: p={p_val:.4f} → {conclusion}"


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------


def generate_hypothesis_report(results: list[str]) -> None:
    """
    Save a text report summarizing hypothesis test outcomes.
    """
    ensure_figures_dir()
    report_path = os.path.join(FIGURES_DIR, "hypothesis_testing_report.txt")

    header = """
============================================================
NBA Last 3 Minutes - Hypothesis Testing Report (liveData v3)
============================================================

Significance level: alpha = 0.05

Summary of tests:
"""

    content = header + "\n" + "\n".join(f"- {r}" for r in results) + "\n"

    with open(report_path, "w") as f:
        f.write(content)

    print(f"\nHypothesis testing report saved to: {report_path}")
    print(content)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print("============================================================")
    print("NBA LAST 3 MINUTES - HYPOTHESIS TESTING (liveData v3)")
    print("============================================================\n")

    df = load_processed_data()
    if df is None or df.empty:
        return

    results = []

    r1 = hypothesis_1_three_point_vs_two_point(df)
    results.append(r1)

    r2 = hypothesis_2_foul_frequency(df)
    results.append(r2)

    r3 = hypothesis_3_score_diff_and_fouls(df)
    results.append(r3)

    generate_hypothesis_report(results)

    print("\nHypothesis testing complete.\n")


if __name__ == "__main__":
    main()
