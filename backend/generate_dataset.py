"""
Mock T20 Dataset Generator
Generates realistic synthetic IPL match data with a deliberate bias
towards RCB playing at M. Chinnaswamy Stadium for testing.
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

TEAMS = [
    "Royal Challengers Bangalore",
    "Mumbai Indians",
    "Chennai Super Kings",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

# Venue name → (par_score_mean, par_score_std, chase_win_rate)
VENUES = {
    "M. Chinnaswamy Stadium, Bengaluru":     (182, 12, 0.55),
    "Wankhede Stadium, Mumbai":              (176, 14, 0.52),
    "MA Chidambaram Stadium, Chennai":       (158, 10, 0.42),
    "Eden Gardens, Kolkata":                 (170, 13, 0.48),
    "Arun Jaitley Stadium, Delhi":           (172, 11, 0.50),
    "Sawai Mansingh Stadium, Jaipur":        (166, 12, 0.47),
    "Rajiv Gandhi Intl Stadium, Hyderabad":  (168, 11, 0.49),
    "IS Bindra Stadium, Mohali":             (171, 13, 0.51),
    "Narendra Modi Stadium, Ahmedabad":      (164, 10, 0.46),
    "BRSABV Ekana Stadium, Lucknow":         (163, 11, 0.45),
}

TOSS_DECISIONS = ["bat", "field"]


def _generate_single_match(rng: np.random.Generator, teams: list, venue: str, venue_stats: tuple):
    """Generate one synthetic T20 match record."""
    par_mean, par_std, chase_win_base = venue_stats

    # Pick two distinct teams
    team1, team2 = rng.choice(teams, size=2, replace=False)

    # Toss
    toss_winner = rng.choice([team1, team2])
    toss_decision = rng.choice(TOSS_DECISIONS, p=[0.35, 0.65])  # T20 skews field-first

    # Determine batting order based on toss
    if toss_decision == "bat":
        batting_first, chasing = toss_winner, (team2 if toss_winner == team1 else team1)
    else:
        chasing, batting_first = toss_winner, (team2 if toss_winner == team1 else team1)

    # ── First Innings ──
    innings_1_score = int(rng.normal(par_mean, par_std))
    innings_1_score = max(90, min(260, innings_1_score))  # clamp realistic range

    powerplay_1 = int(rng.normal(innings_1_score * 0.30, 8))
    powerplay_1 = max(20, min(90, powerplay_1))

    wickets_pp_1 = int(rng.choice([0, 1, 1, 2, 2, 3, 3, 4]))  # weighted

    # Top-order contribution (top 3 batsmen)
    top_order_pct_1 = rng.beta(5, 3)  # skewed towards 0.5-0.7
    top_order_runs_1 = int(innings_1_score * top_order_pct_1)

    innings_1_wickets = int(rng.choice(range(4, 11), p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05]))

    # ── Second Innings (Chase) ──
    # Chase success probability influenced by venue, toss, and target
    target = innings_1_score + 1
    difficulty = (target - par_mean) / par_std  # z-score of target vs venue par

    # Adjust chase probability
    chase_prob = chase_win_base - difficulty * 0.12
    if toss_decision == "field":
        chase_prob += 0.05  # advantage for choosing to chase
    chase_prob = np.clip(chase_prob, 0.15, 0.85)

    chasing_won = int(rng.random() < chase_prob)

    if chasing_won:
        innings_2_score = target + int(rng.integers(0, 15))
        innings_2_wickets = int(rng.choice(range(0, 8), p=[0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.07, 0.03]))
    else:
        innings_2_score = target - int(rng.integers(1, 40))
        innings_2_wickets = int(rng.choice(range(6, 11), p=[0.10, 0.20, 0.30, 0.25, 0.15]))

    innings_2_score = max(60, innings_2_score)

    powerplay_2 = int(rng.normal(innings_2_score * 0.30, 8))
    powerplay_2 = max(15, min(90, powerplay_2))

    wickets_pp_2 = int(rng.choice([0, 0, 1, 1, 2, 2, 3, 4]))

    top_order_pct_2 = rng.beta(5, 3)
    top_order_runs_2 = int(innings_2_score * top_order_pct_2)

    return {
        "venue": venue,
        "team_batting_first": batting_first,
        "team_chasing": chasing,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "innings_1_score": innings_1_score,
        "innings_1_wickets": innings_1_wickets,
        "powerplay_score_1": powerplay_1,
        "wickets_in_pp_1": wickets_pp_1,
        "top_order_runs_1": top_order_runs_1,
        "innings_2_score": innings_2_score,
        "innings_2_wickets": innings_2_wickets,
        "powerplay_score_2": powerplay_2,
        "wickets_in_pp_2": wickets_pp_2,
        "top_order_runs_2": top_order_runs_2,
        "chasing_team_won": chasing_won,
    }


def generate_dataset(n_matches: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic T20 match dataset.

    ~30% of matches feature RCB, ~25% are at Chinnaswamy.
    """
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_matches):
        # Bias: 25% of matches at Chinnaswamy
        if rng.random() < 0.25:
            venue = "M. Chinnaswamy Stadium, Bengaluru"
        else:
            venue = rng.choice(list(VENUES.keys()))

        venue_stats = VENUES[venue]

        # Bias: 30% of matches include RCB
        if rng.random() < 0.30:
            other_teams = [t for t in TEAMS if t != "Royal Challengers Bangalore"]
            teams_for_match = ["Royal Challengers Bangalore", rng.choice(other_teams)]
        else:
            teams_for_match = TEAMS  # random pair from all

        record = _generate_single_match(rng, teams_for_match, venue, venue_stats)
        records.append(record)

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "data", "t20_matches.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = generate_dataset(n_matches=2000)
    df.to_csv(output_path, index=False)

    print(f"[+] Generated {len(df)} matches -> {output_path}")
    print(f"\n[+] Dataset Summary:")
    print(f"   Unique venues:  {df['venue'].nunique()}")
    print(f"   RCB matches:    {df[df['team_batting_first'].str.contains('RCB|Royal Challengers') | df['team_chasing'].str.contains('RCB|Royal Challengers')].shape[0]}")
    print(f"   Chinnaswamy:    {df[df['venue'].str.contains('Chinnaswamy')].shape[0]}")
    print(f"   Chase win rate: {df['chasing_team_won'].mean():.1%}")
    print(f"\n[+] First 3 rows:")
    print(df.head(3).to_string())

