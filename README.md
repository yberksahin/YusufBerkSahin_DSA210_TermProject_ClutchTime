# NBA Last 3 Minutes Strategy Optimization

## Project Overview
A data-driven decision support system that recommends optimal strategies for the critical last 3 minutes of NBA games based on current game state.

## Motivation  
"Clutch time" - basketball's most thrilling moments. 24 seconds left, down by 2, ball in your hands... Should you attempt a three? Drive for a guaranteed two and foul? Or play fast for a two-for-one opportunity?

NBA history's most unforgettable moments happened in these critical minutes. Ray Allen's 2013 Finals three-pointer, Michael Jordan's "The Shot", Damian Lillard's playoff buzzer-beaters... But were these decisions truly optimal, or just lucky?

Statistics show that 35% of NBA games change hands in the final 3 minutes. The decisions coaches make in these moments determine the difference between championship glory and early vacation. In this project, I will analyze 50,000+ critical moments from 2020-2024 seasons to find the mathematically optimal strategy for every situation. Coaches will no longer rely on gut feelings but on data-driven decisions to win games.

## Data Sources

### Primary Data:
- **NBA Stats API** (https://github.com/swar/nba_api)
- 2020-2024 seasons play-by-play data
- Every possession detail: score, time, foul situation, shot clock

### Enrichment Data:
- **NBA.com Clutch Stats** 
- Teams: https://www.nba.com/stats/teams/clutch-traditional
- Players: https://www.nba.com/stats/players/clutch-traditional
- Clutch time shooting percentages, offensive/defensive ratings

## Strategy Categories

### Offensive Strategies (Ball in Our Possession):

**1. Quick 3PT (Quick Three-Pointer)**
- Three-point attempt within 10 seconds or less
- Usually utilized in transition or early offense
- Advantage: Can create 2-for-1 opportunity, tempo advantage
- Risk: Lower shooting percentage

**2. Set 3PT (Set Play Three-Pointer)**
- Open three-point shot created through organized offense
- Generated through screens and ball movement
- Advantage: High-quality shot, better shooting percentage
- Risk: Uses time, turnover risk

**3. Quick 2PT (Quick Two-Pointer)**
- Within 10 seconds, usually drive to basket or post-up
- Used in fast break or mismatch situations
- Advantage: High shooting percentage, chance to draw foul
- Risk: Only 2 points, insufficient when trailing

**4. Safe 2PT (Guaranteed Two-Pointer)**
- Patient play for high-percentage position
- Post play, pick & roll, or quality isolation
- Advantage: Highest success rate
- Risk: Time consuming, only 2 points

**5. Draw Foul (Foul Drawing Focus)**
- Aggressive drive to basket focusing on drawing contact
- Effective in bonus situation or when opponent in foul trouble
- Advantage: Free throws + adds foul to opponent
- Risk: Referees reluctant to call fouls in final moments

### Defensive Strategies (Ball with Opponent):

**1. Intentional Foul (Deliberate Foul)**
- Immediate foul before shot attempt
- Especially on poor free throw shooters ("Hack-a-Shaq")
- Advantage: Stops clock, gives only 2 FT maximum
- Risk: Opponent might make free throws

**2. No Foul Defense (Clean Defense)**
- Normal defense, absolutely no fouling
- Hands up, vertical contest
- Advantage: Forces difficult shot
- Risk: Might give up easy basket

**3. Defend 3PT (Three-Point Prevention)**
- Focus on perimeter defense, willing to give up 2
- Close out all shooters, leave paint open
- Advantage: Prevents 3-point comeback
- Risk: Easy 2 points or foul

**4. Full Pressure (Full Court Press)**
- Aggressive, full court pressure defense
- Aim for steal or 8-second violation
- Advantage: Can create turnover, disrupts opponent
- Risk: Can give up easy basket, high foul risk

## Methodology

1. **Data Collection**: Play-by-play data from NBA API (2020-2024)
2. **Feature Engineering**: Creating game state variables
3. **Exploratory Data Analysis**: Which strategies succeed in which situations?
4. **Hypothesis Testing**: Statistically significant strategies
5. **ML Models**: Optimal strategy prediction with RandomForest and XGBoost
6. **Evaluation**: Win probability and expected value calculations

## Expected Outputs

- Top 3 strategy recommendations for each game state
- Success probability for each strategy (%)
- Expected point differential
- Interactive dashboard for real-time strategy recommendations
