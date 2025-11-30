# NBA Last 3 Minutes Strategy Optimization

## DSA 210 – Fall 2025–2026

---

## **Project Overview**

This project analyzes the final three minutes ("clutch time") of NBA games using play-by-play data from recent seasons. The objective is to understand which offensive and defensive actions are most effective depending on time remaining, score differential, and game context.

Using a custom data pipeline built on the NBA liveData endpoint (V3), the project:

* Collects detailed event logs for the last 180 seconds of regulation and all overtime periods.
* Performs exploratory data analysis (EDA) to understand patterns in shot selection, fouls, and game flow.
* Conducts statistical hypothesis tests to evaluate late‑game strategic assumptions.

---

## **Data Pipeline (V3 LiveData)**

The V3 pipeline uses a more robust approach than the standard nba_api play-by-play endpoint. Key features:

* Retrieves all games for selected seasons.
* Downloads play-by-play data using the official **liveData** API:
  `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json`
* Extracts events only from:

  * **Last 3 minutes of 4th quarter**
  * **All overtime periods**
* Engineers the following features:

  * `time_remaining` (seconds)
  * `score_diff` (homeScore – awayScore)
  * `event_type`, `shot_category`, `foul_indicator`
  * game metadata (date, teams)

Generated datasets are stored under `data/` but are ignored by Git for reproducibility and size considerations.

---

## **Exploratory Data Analysis (EDA)**

The EDA module examines how teams behave in clutch time. The following visuals are produced and stored in `figures/`:

### **1. Temporal Analysis**

* Event frequency across 30‑second bins
* Score differential as a function of time

### **2. Shot Selection**

* Distribution of 2PT, 3PT, and FT attempts
* Shot type vs. score differential

### **3. Game State Heatmap**

* Frequency of (score bucket × time bucket) combinations

### **4. Team-Level Activity**

* Teams appearing most frequently in clutch scenarios

### **5. Event Type Trends**

* Rebounds, turnovers, fouls, substitutions, timeouts, etc.

A text summary (`eda_report.txt`) is also generated.

---

## **Hypothesis Testing (Implemented Tests)**

For the November deliverable, three statistically valid hypothesis tests were implemented.

### **Hypothesis 1 — 3PT vs 2PT Efficiency When Trailing by ≥3**

**H0:** No difference in success rate between 3PT and 2PT attempts in final 30 seconds when trailing by 3+.

**Method:** Two-sample t-test.

**Result:** Insufficient sample size → test skipped.

---

### **Hypothesis 2 — Foul Frequency in Final 30 Seconds**

**H0:** Foul frequency in the last 30 seconds equals foul frequency during 31–180 seconds.

**Method:** Two-proportion z-test.

**Result:** No significant difference at α = 0.05.

---

### **Hypothesis 3 — Score Differential vs. Foul Likelihood**

**H0:** Foul likelihood is independent of score differential bucket.

**Method:** Chi-square independence test.

**Result:** Crosstab degenerate → inconclusive.

---

## **Results Summary**

* Shot behavior and event frequency vary strongly across time bins.
* No conclusive evidence of foul escalation in the final 30 seconds.
* Score differential relationships require more data for robust testing.
* Some late‑game strategies appear underused or highly situational.

---

## **Future Work**

The initial project design included six hypothesis tests. Three are implemented; the remaining three are scheduled for the next phase.

### **1. Bonus Situation Impact**

Evaluate how being in the bonus affects shot selection, foul behavior, and possession strategy.

### **2. Two-for-One Strategy Value**

Quantify expected value differences between quick shots and deliberate late-clock possessions.

### **3. Clutch Player Influence**

Examine whether elite clutch performers shift optimal late-game decision patterns.

These analyses will be incorporated into an ML‑based late‑game decision engine.

---

## **Repository Structure**

```
project/
│
├── data/               # raw + processed (ignored by Git)
├── figures/            # saved plots (ignored by Git)
├── src/
│   ├── data_collection.py
│   ├── exploratory_data_analysis.py
│   └── hypothesis_testing.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## **Reproduction Instructions**

```
pip install -r requirements.txt
python src/data_collection.py
python src/exploratory_data_analysis.py
python src/hypothesis_testing.py
```

---

## **Author**

**Yusuf Berk Şahin**
Sabancı University – DSA210 Term Project
