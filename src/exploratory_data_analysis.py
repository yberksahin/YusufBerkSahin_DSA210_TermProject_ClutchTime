"""
NBA Last 3 Minutes - Exploratory Data Analysis (liveData v3)
Author: <Your Name>
DSA 210 - Fall 2025-2026

This script performs EDA on the critical moments dataset created by
data_collection.py (v3), which uses the NBA liveData JSON endpoint.

It:
- Loads the latest critical_moments_sample_*.csv from data/raw
- Computes basic summaries and derived features
- Produces visualizations and saves them to figures/
- Saves an enhanced processed dataset to data/processed/
- Generates a simple text EDA report
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "figures"


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def ensure_directories() -> None:
    """Make sure output directories exist."""
    for folder in [PROCESSED_DIR, FIGURES_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)


def load_latest_critical_data() -> pd.DataFrame | None:
    """
    Load the most recent critical_moments_sample_*.csv file from data/raw.
    """
    pattern = os.path.join(RAW_DIR, "critical_moments_sample_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("No critical_moments_sample_*.csv files found in data/raw.")
        print("Please run data_collection.py first.")
        return None

    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading data from: {latest_file}")
    df = pd.read_csv(latest_file)
    return df


# ---------------------------------------------------------------------
# Basic overview and feature engineering
# ---------------------------------------------------------------------


def basic_overview(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n====================================")
    print("BASIC DATASET OVERVIEW")
    print("====================================")

    print(f"\nShape: {df.shape}")

    if "GAME_ID" in df.columns:
        print(f"Unique games: {df['GAME_ID'].nunique()}")

    if "GAME_DATE" in df.columns:
        print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

    print("\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\nColumn dtypes:")
    print(df.dtypes.head(20))


def add_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time bin categories for analysis.
    Assumes time_remaining is in seconds.
    """
    if "time_remaining" not in df.columns:
        return df

    df["time_bin"] = pd.cut(
        df["time_remaining"],
        bins=[0, 30, 60, 90, 120, 150, 180],
        labels=["0-30s", "31-60s", "61-90s", "91-120s", "121-150s", "151-180s"],
        include_lowest=True,
    )
    return df


# ---------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------


def temporal_analysis(df: pd.DataFrame) -> None:
    """
    Temporal analysis of events over the last 3 minutes.
    """
    print("\n====================================")
    print("TEMPORAL ANALYSIS")
    print("====================================")

    if "time_remaining" not in df.columns:
        print("time_remaining column not found; skipping temporal analysis.")
        return

    df = add_time_bins(df)

    # Event count by time bin
    plt.figure(figsize=(10, 5))
    counts = df["time_bin"].value_counts().sort_index()
    counts.plot(kind="bar")
    plt.title("Event Count by Time Bin (Last 3 Minutes + OT)")
    plt.xlabel("Time Bin")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "temporal_event_counts.png"), dpi=300)
    plt.close()

    # Score differential vs time
    if "score_diff" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.scatter(df["time_remaining"], df["score_diff"], alpha=0.3, s=5)
        plt.axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.title("Score Differential vs Time Remaining")
        plt.xlabel("Time Remaining (seconds)")
        plt.ylabel("Score Differential (home - away)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURES_DIR, "score_diff_over_time.png"), dpi=300
        )
        plt.close()

        print("\nTemporal summary:")
        print(f"- Average score differential: {df['score_diff'].mean():.2f}")
        print(
            f"- Percentage of tied moments: {(df['score_diff'] == 0).mean() * 100:.1f}%"
        )


# ---------------------------------------------------------------------
# Shot selection analysis
# ---------------------------------------------------------------------


def shot_selection_analysis(df: pd.DataFrame) -> None:
    """
    Analyze shot selection in clutch time using liveData fields:
    - actionType (shot, freeThrow, etc.)
    - shotType  (2PT, 3PT)
    - shotResult (Made, Missed)
    """
    print("\n====================================")
    print("SHOT SELECTION ANALYSIS")
    print("====================================")

    if "actionType" not in df.columns:
        print("actionType column not found; skipping shot analysis.")
        return

    shots = df[df["actionType"].isin(["shot", "freethrow", "freeThrow"])].copy()
    if shots.empty:
        print("No shot events found.")
        return

    if "shotType" not in shots.columns:
        shots["shotType"] = ""
    else:
        shots["shotType"] = (
            shots["shotType"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

    shots["shotResult"] = (
        shots.get("shotResult", "")
        .astype(str)
        .fillna("")
        .str.capitalize()
    )

    shots["actionType"] = shots["actionType"].astype(str).str.lower()

    def classify_shot(row) -> str:
        t = row.get("shotType", "")
        a = row.get("actionType", "")
        if "3" in t:
            return "3PT"
        if a in ["freethrow", "freeThrow"] or "ft" in t or "free" in t:
            return "FT"
        return "2PT"

    shots["shot_category"] = shots.apply(classify_shot, axis=1)

    # Overall distribution
    plt.figure(figsize=(6, 6))
    counts = shots["shot_category"].value_counts()
    counts.plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.title("Shot Type Distribution (Last 3 Minutes + OT)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shot_type_distribution.png"), dpi=300)
    plt.close()

    print("\nShot type counts:")
    print(counts)


# ---------------------------------------------------------------------
# Game state heatmap
# ---------------------------------------------------------------------


def heatmap_game_states(df: pd.DataFrame) -> None:
    """
    Heatmap of how often different (time, score_diff) game states occur.
    """
    print("\n====================================")
    print("GAME STATE HEATMAP")
    print("====================================")

    if "time_remaining" not in df.columns or "score_diff" not in df.columns:
        print("time_remaining or score_diff missing; skipping heatmap.")
        return

    df = add_time_bins(df)

    df["score_bucket"] = pd.cut(
        df["score_diff"],
        bins=[-20, -6, -3, 0, 3, 6, 20],
        labels=["Home -7+", "Home -4 to -6", "Home -1 to -3",
                "Tied", "Home +1 to +3", "Home +4+"],
    )

    pivot = pd.crosstab(df["score_bucket"], df["time_bin"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of Events"},
    )
    plt.title("Frequency of Game States (Score vs Time)")
    plt.xlabel("Time Bin")
    plt.ylabel("Score Bucket (home - away)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "game_state_heatmap.png"), dpi=300)
    plt.close()

    print("\nMost common game states (top 5):")
    flat = []
    for score in pivot.index:
        for t in pivot.columns:
            flat.append((score, t, pivot.loc[score, t]))
    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)
    for row in flat_sorted[:5]:
        print(f"  - {row[0]} with {row[1]} remaining: {row[2]} events")


# ---------------------------------------------------------------------
# Team activity
# ---------------------------------------------------------------------


def team_clutch_activity(df: pd.DataFrame) -> None:
    """
    Simple measure: teams that appear most often in clutch events.
    Relies on MATCHUP column from LeagueGameFinder output.
    """
    print("\n====================================")
    print("TEAM CLUTCH ACTIVITY")
    print("====================================")

    if "MATCHUP" not in df.columns:
        print("MATCHUP column not found; skipping team analysis.")
        return

    # MATCHUP examples: "LAL @ BOS", "BOS vs. LAL"
    home = df["MATCHUP"].str.extract(r"^([A-Z]{2,3})\s")[0]
    away = df["MATCHUP"].str.extract(r"\s([A-Z]{2,3})$")[0]

    counts = pd.concat([home, away], ignore_index=True).value_counts()

    if counts.empty:
        print("Could not parse team abbreviations from MATCHUP; skipping.")
        return

    top10 = counts.head(10)

    plt.figure(figsize=(10, 5))
    top10.plot(kind="bar")
    plt.title("Top 10 Teams by Number of Clutch Events")
    plt.xlabel("Team")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "team_clutch_activity.png"), dpi=300)
    plt.close()

    print("\nTop 5 teams:")
    print(top10.head(5))


# ---------------------------------------------------------------------
# Event type distribution
# ---------------------------------------------------------------------


def event_type_distribution(df: pd.DataFrame) -> None:
    """
    Distribution of actionType in clutch time (shot, rebound, foul, etc.).
    """
    print("\n====================================")
    print("EVENT TYPE DISTRIBUTION")
    print("====================================")

    if "actionType" not in df.columns:
        print("actionType column not found; skipping event type analysis.")
        return

    counts = df["actionType"].value_counts()

    plt.figure(figsize=(10, 6))
    counts.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Event Types in Last 3 Minutes + OT")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "event_type_distribution.png"), dpi=300)
    plt.close()

    print("\nEvent type counts (top 10):")
    print(counts.head(10))


# ---------------------------------------------------------------------
# Save processed data & report
# ---------------------------------------------------------------------


def save_processed_data(df: pd.DataFrame) -> None:
    """
    Save processed dataset with added EDA features.
    """
    output_path = os.path.join(PROCESSED_DIR, "processed_critical_moments.csv")
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")


def generate_eda_report(df: pd.DataFrame) -> None:
    """
    Generate a simple human-readable EDA text report.
    """
    n_rows = len(df)
    n_games = df["GAME_ID"].nunique() if "GAME_ID" in df.columns else "N/A"
    date_min = df["GAME_DATE"].min() if "GAME_DATE" in df.columns else "N/A"
    date_max = df["GAME_DATE"].max() if "GAME_DATE" in df.columns else "N/A"
    avg_events = n_rows / n_games if isinstance(n_games, int) and n_games > 0 else 0

    time_bin_mode = (
        df["time_bin"].value_counts().idxmax() if "time_bin" in df.columns else "N/A"
    )
    score_diff_mean = df["score_diff"].mean() if "score_diff" in df.columns else 0.0
    tied_pct = (
        (df["score_diff"] == 0).mean() * 100 if "score_diff" in df.columns else 0.0
    )

    report = f"""
===========================================
NBA Last 3 Minutes - EDA Report (liveData v3)
===========================================

Dataset Overview
----------------
- Total records: {n_rows}
- Unique games: {n_games}
- Date range: {date_min} to {date_max}
- Average events per game: {avg_events:.1f}

Key Temporal Insights
---------------------
- Most frequent time bin: {time_bin_mode}
- Average score differential (home - away): {score_diff_mean:.2f}
- Percentage of tied moments: {tied_pct:.1f}%

Notes
-----
- Data limited to last 3 minutes of the 4th quarter plus any overtime.
- All values derived from NBA liveData play-by-play events.

"""

    report_path = os.path.join(FIGURES_DIR, "eda_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nEDA report saved to: {report_path}")
    print(report)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print("\n====================================")
    print("NBA LAST 3 MINUTES - EDA (liveData v3)")
    print("====================================\n")

    ensure_directories()

    df = load_latest_critical_data()
    if df is None or df.empty:
        return

    basic_overview(df)
    df = add_time_bins(df)

    temporal_analysis(df)
    shot_selection_analysis(df)
    heatmap_game_states(df)
    team_clutch_activity(df)
    event_type_distribution(df)

    save_processed_data(df)
    generate_eda_report(df)

    print("\nEDA complete.\n")


if __name__ == "__main__":
    main()
