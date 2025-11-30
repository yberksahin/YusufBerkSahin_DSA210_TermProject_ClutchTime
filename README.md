# README.md

### NBA Last 3 Minutes Strategy Optimizationssss

**DSA 210 – Fall 2025–2026**

---

## 1. Project Overview

This project investigates **decision-making in the final 3 minutes of NBA games**, a period where small choices often determine the outcome. Using **real play-by-play data from the 2020–2024 seasons**, the goal is to understand how offensive and defensive strategies perform under different game states (score margin, foul situation, time pressure, player involvement).

The project delivers:

* Automated data collection with `nba_api`
* Feature engineering to reconstruct game states
* Exploratory Data Analysis (EDA) of clutch-time behavior
* Six hypothesis tests powered entirely by real NBA data

The November 28 milestone includes data acquisition, EDA, and hypothesis testing. Machine learning components will follow in the January phase.

---

## 2. Data Sources

All datasets are obtained programmatically using the official **nba_api**.

### Primary Endpoints

* `playbyplayv2`: Full, event-level logs for every game
* `leaguegamefinder`: Retrieve every game ID between 2020–2024
* `boxscoretraditionalv2`: Player boxscores used to identify top scorers

### Derived Real-Data Features

These fields are created directly from raw play-by-play sequences:

* **Game free-throw accuracy** (computed from real FT attempts)
* **Possession duration** (based on timestamp differences)
* **Team foul counts** → used to approximate bonus status
* **Final score differential** (from last scoring event)
* **Top scorer per team** (via boxscore data)

No synthetic variables or simulations are used.

---

## 3. Feature Engineering

| Feature                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `time_remaining`        | Seconds left in the period                     |
| `score_diff`            | Real-time score margin (home positive)         |
| `shot_type`             | 3PT / 2PT / FT classification                  |
| `team_foul_count`       | Cumulative fouls at each moment                |
| `in_bonus`              | Bonus approximation from foul counts           |
| `shot_clock_used`       | Time elapsed since previous event by same team |
| `final_score_diff`      | Actual final margin of the game                |
| `top_scorer_id`         | Highest-scoring player on each team            |
| `is_clutch_player_shot` | Whether the shooter is the top scorer          |

These variables allow the project to analyze high-pressure decision outcomes using genuine historical behavior.

---

## 4. Exploratory Data Analysis (EDA)

The EDA explores clutch-time patterns across four NBA seasons:

* Event frequency across time intervals (0–30s, 31–60s, etc.)
* Score margin evolution and tie situations
* Shot selection behavior in different score states
* Heatmaps showing distribution of game situations
* Team-specific clutch event counts
* Event-type distributions (shots, fouls, turnovers, timeouts)

All figures are saved in `/figures`, and an overall summary is generated in `eda_report.txt`.

---

## 5. Hypothesis Testing (Real Data)

Six hypotheses are tested using only real NBA data.

### **H1 — Three-Point vs Two-Point Strategy (Trailing by 3+)**

Compares success rates of 3PT vs 2PT attempts in the last 30 seconds using a two-sample t-test.

### **H2 — Intentional Foul Effectiveness**

Uses real game FT% and final score outcomes to evaluate whether fouling is beneficial at specific FT accuracy thresholds.

### **H3 — Two-for-One Strategy Value**

Classifies quick (<7s) vs deliberate shots using real possession timing and compares win outcomes.

### **H4 — Impact of Bonus Situation**

Examines whether shot success changes when teams are near or in the bonus (using real foul counts).

### **H5 — Score Differential Breakpoints**

Analyzes whether shot success varies nonlinearly across score margins.

### **H6 — Clutch Player Involvement**

Tests whether shots taken by a team’s top scorer behave differently under clutch pressure.

---

## 6. Repository Structure

```
/data
    /raw
    /processed
/figures
/src
    data_collection.py
    exploratory_data_analysis.py
    hypothesis_testing.py
README.md
requirements.txt
```

---

## 7. Reproducibility

To run the project:

```bash
pip install -r requirements.txt
python src/data_collection.py
python src/exploratory_data_analysis.py
python src/hypothesis_testing.py
```

---

## 8. Limitations

* Bonus status is approximated using available fouls (full-period data not provided).
* Exact shot-clock timestamps are not included in play-by-play logs.
* Lineup tracking is limited; the clutch player is defined as the team's top scorer.

Despite these constraints, all analyses rely strictly on **real NBA events** without simulated input.

---

## 9. Project Status (28 November)

* ✔ Data collected (2020–2024)
* ✔ Real feature engineering completed
* ✔ EDA completed
* ✔ Six hypothesis tests completed
* ✔ Repository fully aligned with DSA210 requirements
