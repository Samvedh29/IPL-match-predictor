"""
Feature Engineering Module
Computes the three key features for the T20 prediction model:
  1. Venue Par Score      — historical average first-innings total at the venue
  2. Toss Decision Impact — encoded toss advantage (field-first boost on high-scoring grounds)
  3. Top-Order Dependency — ratio of top-3 batsmen's runs to total innings score
"""

import pandas as pd
import numpy as np


def compute_venue_par_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: Venue Par Score
    Rolling average of first-innings totals at each venue.
    For the first match at a venue, uses the overall mean as a prior.
    """
    overall_mean = df["innings_1_score"].mean()

    venue_par = (
        df.groupby("venue")["innings_1_score"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["venue_par_score"] = venue_par.fillna(overall_mean).round(1)
    return df


def compute_toss_decision_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: Toss Decision Impact
    Combines toss decision with venue characteristics:
      - field-first on a high-scoring venue → positive impact
      - bat-first on a low-scoring venue   → positive impact
      - Otherwise                          → neutral/negative

    Encoded as a continuous score in [-1, 1].
    """
    median_par = df["venue_par_score"].median()

    def _impact(row):
        chose_field = 1 if row["toss_decision"] == "field" else -1
        venue_factor = (row["venue_par_score"] - median_par) / median_par
        toss_won_by_chaser = 1 if row["toss_winner"] == row["team_chasing"] else -1

        # Core formula: field-first on high-scoring venues is advantageous
        impact = chose_field * (0.4 + 0.6 * venue_factor) * toss_won_by_chaser
        return np.clip(impact, -1.0, 1.0)

    df["toss_decision_impact"] = df.apply(_impact, axis=1).round(3)
    return df


def compute_top_order_dependency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: Top-Order Dependency
    Ratio of top-3 batsmen's runs to the total innings score.
    High dependency (>0.65) indicates fragile lower-order — risky for chasing.
    Computed for both innings.
    """
    df["top_order_dep_1"] = (df["top_order_runs_1"] / df["innings_1_score"].clip(lower=1)).round(3)
    df["top_order_dep_2"] = (df["top_order_runs_2"] / df["innings_2_score"].clip(lower=1)).round(3)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all three feature transformations in sequence."""
    df = compute_venue_par_score(df)
    df = compute_toss_decision_impact(df)
    df = compute_top_order_dependency(df)
    return df


# ──────────────────────────────────────────────────────────
# Feature columns used by the model
# ──────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "venue_par_score",
    "toss_decision_impact",
    "top_order_dep_1",
    "innings_1_score",
    "powerplay_score_1",
    "wickets_in_pp_1",
    "innings_1_wickets",
    "powerplay_score_2",
    "wickets_in_pp_2",
    "top_order_dep_2",
]

TARGET_COLUMN = "chasing_team_won"


if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "data", "t20_matches.csv")
    if not os.path.exists(csv_path):
        print("❌ Dataset not found. Run generate_dataset.py first.")
        exit(1)

    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    print("✅ Features engineered successfully!")
    print(f"\n📐 New columns: venue_par_score, toss_decision_impact, top_order_dep_1, top_order_dep_2")
    print(f"\n📊 Feature statistics:")
    print(df[FEATURE_COLUMNS].describe().round(2).to_string())
