"""
NBA Last 3 Minutes Data Collection (v3)
DSA 210 - Fall 2025-2026

This script:
1) Uses nba_api to get all NBA games for selected seasons.
2) Uses the NBA liveData play-by-play JSON endpoint (cdn.nba.com)
   to retrieve detailed play-by-play data for each game.
3) Extracts ONLY the last 3 minutes of the 4th quarter plus all overtimes.
4) Saves a sample of critical moments to data/raw.
"""

import os
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

PBP_URL_TEMPLATE = (
    "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
)

NBA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.1 Safari/537.36"
    ),
    "Referer": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


def get_all_games(seasons: list[str]) -> pd.DataFrame:
    """
    Retrieve all unique regular season games for the given seasons
    using LeagueGameFinder.
    """
    print(f"Collecting games for seasons: {seasons}")
    all_games = []

    nba_teams = teams.get_teams()

    for season in seasons:
        print(f"\nSeason: {season}")
        season_games = []

        for team in tqdm(nba_teams, desc=f"Teams in {season}"):
            try:
                gf = leaguegamefinder.LeagueGameFinder(
                    team_id_nullable=team["id"],
                    season_nullable=season,
                    season_type_nullable="Regular Season",
                )
                df = gf.get_data_frames()[0]
                season_games.append(df)
                time.sleep(0.4)  # rate limiting
            except Exception as exc:
                print(f"Error for team {team['full_name']} in {season}: {exc}")

        if season_games:
            season_df = pd.concat(season_games, ignore_index=True)
            season_df = season_df.drop_duplicates(subset=["GAME_ID"])
            print(f"  Unique games in {season}: {len(season_df)}")
            all_games.append(season_df)

    if not all_games:
        print("No games found.")
        return pd.DataFrame()

    all_games_df = pd.concat(all_games, ignore_index=True)
    print(f"\nTotal unique games across seasons: {len(all_games_df)}")
    return all_games_df


def clock_to_seconds(clock_str: str) -> int:
    """
    Convert NBA 'clock' string to seconds remaining in the period.
    Supports formats like '02:34' and 'PT2M34.00S'.
    """
    if not isinstance(clock_str, str) or clock_str.strip() == "":
        return 0

    clock_str = clock_str.strip()
    try:
        # ISO8601 format, e.g. 'PT2M34.00S'
        if clock_str.startswith("PT"):
            import re

            minutes_match = re.search(r"(\d+)M", clock_str)
            seconds_match = re.search(r"(\d+(\.\d+)?)S", clock_str)

            minutes = int(minutes_match.group(1)) if minutes_match else 0
            seconds = float(seconds_match.group(1)) if seconds_match else 0.0
            return int(round(minutes * 60 + seconds))

        # Standard 'MM:SS' format
        if ":" in clock_str:
            minutes, seconds = clock_str.split(":")
            return int(minutes) * 60 + int(float(seconds))

        # Fallback: try direct cast
        return int(float(clock_str))
    except Exception:
        return 0


def fetch_pbp_live(game_id: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch play-by-play data from NBA liveData endpoint for a single game.

    Returns a DataFrame with one row per action/event.
    """
    url = PBP_URL_TEMPLATE.format(game_id=game_id)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=NBA_HEADERS, timeout=15)
            resp.raise_for_status()

            data = resp.json()
            actions = data.get("game", {}).get("actions", [])

            if not actions:
                print(f"  No actions found for game {game_id}")
                return pd.DataFrame()

            df = pd.DataFrame(actions)
            df["GAME_ID"] = game_id
            return df

        except Exception as exc:
            print(
                f"Error retrieving PBP for {game_id} "
                f"(attempt {attempt}/{max_retries}): {exc}"
            )
            time.sleep(1.0)

    return pd.DataFrame()


def extract_last_3_minutes(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter play-by-play to:
      - last 3 minutes of the 4th quarter
      - all overtime periods (PERIOD > 4)
    Also compute score differential and time remaining.
    """
    if pbp_df.empty:
        return pbp_df

    df = pbp_df.copy()

    # Ensure required columns exist
    # liveData typically uses 'period', 'clock', 'homeScore', 'awayScore'
    if "period" not in df.columns or "clock" not in df.columns:
        print("  Warning: period/clock columns missing in PBP data.")
        return pd.DataFrame()

    # Convert clock to seconds remaining in the period
    df["time_remaining"] = df["clock"].apply(clock_to_seconds)

    # Score differential (home - away)
    home_col = "homeScore"
    away_col = "awayScore"
    if home_col in df.columns and away_col in df.columns:
        df[home_col] = df[home_col].fillna(0).astype(int)
        df[away_col] = df[away_col].fillna(0).astype(int)
        df["score_diff"] = df[home_col] - df[away_col]
    else:
        df["score_diff"] = 0

    # Last 3 minutes of 4th quarter + all overtimes
    mask_last3_q4 = (df["period"] == 4) & (df["time_remaining"] <= 180)
    mask_ot = df["period"] > 4
    critical = df[mask_last3_q4 | mask_ot].copy()

    return critical


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to CSV and pickle under data/raw."""
    if df.empty:
        print(f"DataFrame '{filename}' is empty. Nothing to save.")
        return

    csv_path = os.path.join(RAW_DIR, f"{filename}.csv")
    pkl_path = os.path.join(RAW_DIR, f"{filename}.pkl")

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"Saved CSV  : {csv_path}")
    print(f"Saved PKL  : {pkl_path}")


def extract_critical_moments(
    games_df: pd.DataFrame, limit: int | None = 100
) -> pd.DataFrame:
    """
    For a set of games, fetch play-by-play and extract critical moments.

    Parameters
    ----------
    games_df : DataFrame
        Output of get_all_games().
    limit : int or None
        Number of games to process (for experimentation).
        Use None to process all games (can be very slow).
    """
    if games_df.empty:
        return pd.DataFrame()

    if limit is not None:
        games_df = games_df.head(limit)

    print(f"\nExtracting critical moments from {len(games_df)} games...")

    all_critical = []

    for _, game_row in tqdm(
        games_df.iterrows(), total=len(games_df), desc="Games"
    ):
        game_id = game_row["GAME_ID"]

        pbp_raw = fetch_pbp_live(game_id)
        if pbp_raw.empty:
            continue

        critical = extract_last_3_minutes(pbp_raw)
        if critical.empty:
            continue

        # Attach metadata
        critical["GAME_DATE"] = game_row["GAME_DATE"]
        critical["MATCHUP"] = game_row["MATCHUP"]

        all_critical.append(critical)

        # be nice to the API
        time.sleep(0.3)

    if not all_critical:
        print("No critical moments extracted.")
        return pd.DataFrame()

    critical_df = pd.concat(all_critical, ignore_index=True)
    print(f"\nTotal critical events collected: {len(critical_df)}")
    return critical_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    print("============================================================")
    print("NBA LAST 3 MINUTES - DATA COLLECTION (v3)")
    print("============================================================\n")

    ensure_directories()

    # Step 1: game list
    print("Step 1: Fetching all game IDs...")
    games_df = get_all_games(SEASONS)
    if games_df.empty:
        print("No games found. Exiting.")
        return

    save_dataframe(
        games_df,
        f"all_games_{datetime.now().strftime('%Y%m%d')}",
    )

    # Step 2: critical moments
    print("\nStep 2: Extracting critical moments (last 3 minutes + OT)...")
    critical_df = extract_critical_moments(games_df, limit=100)

    if not critical_df.empty:
        filename = f"critical_moments_sample_{datetime.now().strftime('%Y%m%d')}"
        save_dataframe(critical_df, filename)

        print("\nSample preview:\n")
        preview_cols = [
            "GAME_ID",
            "GAME_DATE",
            "MATCHUP",
            "period",
            "clock",
            "time_remaining",
            "homeScore",
            "awayScore",
            "score_diff",
            "description",
        ]
        existing_cols = [c for c in preview_cols if c in critical_df.columns]
        print(critical_df[existing_cols].head(10))

    print("\nData collection complete.\n")


if __name__ == "__main__":
    main()
