"""
Validate model predictions against IPL 2026 matches played AFTER May 1 
(matches not in the original Cricsheet dataset).
Data sourced from ESPNcricinfo, iplt20.com, and news reports.
"""
import os
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATS_PATH = os.path.join(MODEL_DIR, "latest_stats.json")

# Matches after May 1, 2026 (not in our Cricsheet dataset)
# Format: (date, team1, team2, venue, toss_winner, toss_decision, actual_winner)
NEW_MATCHES = [
    # May 2 — MI vs CSK (source: bignewsnetwork)
    ("2026-05-02", "Mumbai Indians", "Chennai Super Kings",
     "MA Chidambaram Stadium, Chepauk, Chennai",
     "Mumbai Indians", "bat", "Chennai Super Kings"),
    
    # May 3 — PBKS vs GT (source: iplt20.com match 48)
    ("2026-05-03", "Punjab Kings", "Gujarat Titans",
     "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
     "Punjab Kings", "field", "Punjab Kings"),

    # May 4 — RCB vs CSK (source: iplt20.com match 49)
    ("2026-05-04", "Royal Challengers Bengaluru", "Chennai Super Kings",
     "M Chinnaswamy Stadium, Bengaluru",
     "Royal Challengers Bengaluru", "field", "Royal Challengers Bengaluru"),

    # May 5 — CSK vs PBKS (source: iplt20.com) — CSK won by 28 runs
    ("2026-05-05", "Chennai Super Kings", "Punjab Kings",
     "M Chinnaswamy Stadium, Bengaluru",
     "Chennai Super Kings", "field", "Chennai Super Kings"),

    # May 5 — KKR vs LSG (source: iplt20.com) — KKR won by 98 runs
    ("2026-05-05", "Kolkata Knight Riders", "Lucknow Super Giants",
     "Himachal Pradesh Cricket Association Stadium, Dharamsala",
     "Kolkata Knight Riders", "field", "Kolkata Knight Riders"),

    # May 6 — MI vs SRH (source: iplt20.com) — MI won by 7 wickets
    ("2026-05-06", "Mumbai Indians", "Sunrisers Hyderabad",
     "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
     "Mumbai Indians", "field", "Mumbai Indians"),

    # May 7 — DC vs RR (source: iplt20.com) — DC won by 20 runs
    ("2026-05-07", "Delhi Capitals", "Rajasthan Royals",
     "Arun Jaitley Stadium, Delhi",
     "Delhi Capitals", "field", "Delhi Capitals"),

    # May 8 — SRH vs LSG (source: iplt20.com) — SRH won by 10 wickets
    ("2026-05-08", "Sunrisers Hyderabad", "Lucknow Super Giants",
     "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
     "Sunrisers Hyderabad", "field", "Sunrisers Hyderabad"),

    # May 9 — RR vs GT (source: iplt20.com) — GT won by 77 runs
    ("2026-05-09", "Rajasthan Royals", "Gujarat Titans",
     "Sawai Mansingh Stadium, Jaipur",
     "Gujarat Titans", "field", "Gujarat Titans"),

    # May 10 — CSK vs LSG (source: iplt20.com) — CSK won by 5 wickets
    ("2026-05-10", "Chennai Super Kings", "Lucknow Super Giants",
     "MA Chidambaram Stadium, Chepauk, Chennai",
     "Chennai Super Kings", "field", "Chennai Super Kings"),

    # May 10 — MI vs RCB (source: iplt20.com) — RCB won by 2 wickets
    ("2026-05-10", "Mumbai Indians", "Royal Challengers Bengaluru",
     "Wankhede Stadium, Mumbai",
     "Mumbai Indians", "field", "Royal Challengers Bengaluru"),

    # May 11 — PBKS vs DC (source: tribuneindia) — DC won by 3 wickets
    ("2026-05-11", "Punjab Kings", "Delhi Capitals",
     "Himachal Pradesh Cricket Association Stadium, Dharamsala",
     "Delhi Capitals", "field", "Delhi Capitals"),

    # May 12 — GT vs SRH (source: iplt20.com) — GT won by 82 runs
    ("2026-05-12", "Gujarat Titans", "Sunrisers Hyderabad",
     "Narendra Modi Stadium, Ahmedabad",
     "Gujarat Titans", "field", "Gujarat Titans"),

    # May 13 — RCB vs KKR (source: iplt20.com) — RCB won by 6 wickets
    ("2026-05-13", "Royal Challengers Bengaluru", "Kolkata Knight Riders",
     "M Chinnaswamy Stadium, Bengaluru",
     "Royal Challengers Bengaluru", "field", "Royal Challengers Bengaluru"),

    # May 14 — PBKS vs MI (source: mid-day) — MI won by 6 wickets
    ("2026-05-14", "Punjab Kings", "Mumbai Indians",
     "Himachal Pradesh Cricket Association Stadium, Dharamsala",
     "Mumbai Indians", "field", "Mumbai Indians"),

    # May 15 — LSG vs CSK (source: sportskeeda) — LSG won by 7 wickets
    ("2026-05-15", "Lucknow Super Giants", "Chennai Super Kings",
     "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
     "Lucknow Super Giants", "field", "Lucknow Super Giants"),
]


def run_validation():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_post_toss.pkl"))
    venue_le = joblib.load(os.path.join(MODEL_DIR, "venue_le.pkl"))
    team_le = joblib.load(os.path.join(MODEL_DIR, "team_le.pkl"))

    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

    known_venues = set(venue_le.classes_)

    print(f"\n[*] Validating model on {len(NEW_MATCHES)} NEW IPL 2026 matches (May 2 – May 15)")
    print(f"    These matches are NOT in the training/test data.\n")

    print("=" * 100)
    print(f"{'Date':<12} | {'Matchup':<45} | {'Model Prediction':<30} | {'Actual':<8}")
    print("-" * 100)

    correct = 0
    total = 0

    for date, t1, t2, venue, toss_w, toss_d, actual_winner in NEW_MATCHES:
        # Encode venue
        v_clean = venue if venue in known_venues else venue_le.classes_[0]
        v_enc = venue_le.transform([v_clean])[0]

        t1_enc = team_le.transform([t1])[0]
        t2_enc = team_le.transform([t2])[0]

        toss_winner_is_t1 = 1 if toss_w == t1 else 0
        toss_decision_bat = 1 if toss_d.lower() == "bat" else 0

        t1_elo = stats["team_avg_player_elo"].get(t1, 1500.0)
        t2_elo = stats["team_avg_player_elo"].get(t2, 1500.0)
        t1_form = stats["team_recent_form"].get(t1, 0.5)
        t2_form = stats["team_recent_form"].get(t2, 0.5)
        global_bias = stats["global_chase_bias"]
        venue_bias = stats["venue_chase_bias"].get(venue, global_bias)

        feature_cols = [
            "venue_enc", "t1_enc", "t2_enc",
            "toss_winner_is_t1", "toss_decision_bat",
            "t1_player_form_elo", "t2_player_form_elo",
            "t1_recent_form", "t2_recent_form",
            "global_chase_bias", "venue_chase_bias",
            "is_impact_player_era",
        ]

        X_dict = {
            "venue_enc": v_enc, "t1_enc": t1_enc, "t2_enc": t2_enc,
            "toss_winner_is_t1": toss_winner_is_t1,
            "toss_decision_bat": toss_decision_bat,
            "t1_player_form_elo": t1_elo, "t2_player_form_elo": t2_elo,
            "t1_recent_form": t1_form, "t2_recent_form": t2_form,
            "global_chase_bias": global_bias, "venue_chase_bias": venue_bias,
            "is_impact_player_era": 1,
        }

        X = pd.DataFrame([X_dict])[feature_cols]
        prob_t1 = float(model.predict_proba(X)[0][1])
        pred_t1_wins = prob_t1 >= 0.5
        predicted_winner = t1 if pred_t1_wins else t2
        conf = prob_t1 * 100 if pred_t1_wins else (1 - prob_t1) * 100

        is_correct = predicted_winner == actual_winner
        if is_correct:
            correct += 1
        total += 1

        matchup = f"{t1} vs {t2}"
        res_str = "OK" if is_correct else "WRONG"
        pred_str = f"{predicted_winner} ({conf:.1f}%)"

        print(f"{date:<12} | {matchup:<45} | {pred_str:<30} | {res_str}")

    print("=" * 100)
    print(f"\nNEW MATCHES ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
    
    # Combined with original 42 matches
    orig_correct = 26
    orig_total = 42
    combined_correct = orig_correct + correct
    combined_total = orig_total + total
    print(f"COMBINED (all {combined_total} matches): {combined_correct}/{combined_total} = {combined_correct/combined_total*100:.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    run_validation()
