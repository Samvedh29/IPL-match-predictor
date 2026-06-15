import os
import json
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from ipl_2026_squads import IPL_2026_SQUADS

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_post_toss.pkl")
VENUE_LE_PATH = os.path.join(MODEL_DIR, "venue_le.pkl")
TEAM_LE_PATH = os.path.join(MODEL_DIR, "team_le.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "latest_stats.json")

class PredictRequest(BaseModel):
    venue: str = Field(..., description="Match venue name")
    team1: str = Field(..., description="First team")
    team2: str = Field(..., description="Second team")
    toss_winner: str = Field(..., description="Team that won the toss")
    toss_decision: str = Field(..., description="'bat' or 'field'")
    team1_players: Optional[list] = Field(None, description="Optional list of 11 player names for Team 1")
    team2_players: Optional[list] = Field(None, description="Optional list of 11 player names for Team 2")

class PredictResponse(BaseModel):
    win_probability: float = Field(..., description="Team 1 win probability (0-100%)")
    team1: str
    team2: str
    top_features: list
    input_summary: dict
    diagnosis: Optional[dict] = Field(None, description="Detailed breakdown of prediction factors")
    projected_score_t1: Optional[int] = Field(None)
    projected_score_t2: Optional[int] = Field(None)
    run_progression_t1: Optional[list] = Field(None)
    run_progression_t2: Optional[list] = Field(None)
    match_description: Optional[str] = Field(None)

model = None
venue_le = None
team_le = None
latest_stats = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, venue_le, team_le, latest_stats
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    venue_le = joblib.load(VENUE_LE_PATH)
    team_le = joblib.load(TEAM_LE_PATH)
    
    with open(STATS_PATH, "r") as f:
        latest_stats = json.load(f)
        
    print(f"[+] Advanced Pre-Match XGBoost model and stats loaded")

    yield
    print("[*] Shutting down...")

app = FastAPI(
    title="Advanced T20 Pre-Match Predictor",
    description="AI-powered 0-ball prediction with Synergy and Chase Bias",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_team_elo(team_name: str, players: list):
    if not players:
        return latest_stats['team_avg_player_elo'].get(team_name, 1500.0)
    elos = [latest_stats['player_elos'].get(p, 1500.0) for p in players]
    return sum(elos) / len(elos)

import random
import numpy as np

def simulate_match_progression(t1_name, t2_name, t1_elo, t2_elo, t1_form, t2_form, venue_bias, toss_winner, toss_decision, win_prob):
    base_score = 165
    venue_impact = (venue_bias - 0.5) * 30
    elo_diff_t1 = (t1_elo - 1500) / 10
    elo_diff_t2 = (t2_elo - 1500) / 10
    form_diff_t1 = (t1_form - 0.5) * 20
    form_diff_t2 = (t2_form - 0.5) * 20
    
    if (toss_winner == t1_name and toss_decision == 'bat') or (toss_winner == t2_name and toss_decision == 'field'):
        batting_first = t1_name
        chasing = t2_name
    else:
        batting_first = t2_name
        chasing = t1_name
        
    t1_batting_first = (batting_first == t1_name)
    
    proj_t1 = int(base_score + elo_diff_t1 + form_diff_t1 + (0 if t1_batting_first else venue_impact))
    proj_t2 = int(base_score + elo_diff_t2 + form_diff_t2 + (0 if not t1_batting_first else venue_impact))
    
    if win_prob > 55 and t1_batting_first and proj_t1 <= proj_t2:
        proj_t1 = proj_t2 + random.randint(5, 15)
    elif win_prob > 55 and not t1_batting_first and proj_t2 <= proj_t1:
        proj_t1 = proj_t2 + random.randint(2, 8)
    elif win_prob < 45 and t1_batting_first and proj_t2 <= proj_t1:
        proj_t2 = proj_t1 + random.randint(2, 8)
    elif win_prob < 45 and not t1_batting_first and proj_t1 <= proj_t2:
        proj_t2 = proj_t1 + random.randint(5, 15)
        
    def generate_progression(total_score):
        weights = [8]*6 + [7.5]*9 + [10]*5
        weights = np.array(weights)
        weights = weights / sum(weights)
        runs = []
        cumulative = 0
        for i in range(19):
            target_over = total_score * weights[i]
            over_score = max(2, int(np.random.normal(target_over, 2)))
            cumulative += over_score
            runs.append(cumulative)
        runs.append(total_score)
        for i in range(1, 20):
            if runs[i] < runs[i-1]:
                runs[i] = runs[i-1] + random.randint(1, 4)
        runs[-1] = total_score
        return [int(x) for x in runs]
        
    prog_t1 = generate_progression(proj_t1)
    prog_t2 = generate_progression(proj_t2)
    
    desc = ""
    if win_prob > 60:
        desc += f"{t1_name} enters this match as strong favorites ({win_prob:.1f}% probability). "
    elif win_prob < 40:
        desc += f"{t2_name} is heavily favored to win ({100-win_prob:.1f}% probability). "
    else:
        desc += f"This is a tightly contested matchup ({win_prob:.1f}% vs {100-win_prob:.1f}%). "
        
    if form_diff_t1 > 5 and form_diff_t2 < -5:
         desc += f"{t1_name}'s excellent recent form gives them a major edge. "
    elif form_diff_t2 > 5 and form_diff_t1 < -5:
         desc += f"{t2_name}'s recent momentum is a key positive factor for them. "
         
    if venue_impact > 5:
        desc += f"The venue heavily favors the chasing team, suggesting that {chasing} will have an advantage in the second innings."
    elif venue_impact < -5:
        desc += f"The venue is statistically better for defending totals, playing into {batting_first}'s hands."
    else:
        desc += f"The toss decision to {toss_decision} by {toss_winner} sets up an intriguing strategic battle."
        
    return proj_t1, proj_t2, prog_t1, prog_t2, desc

def build_feature_vector(req: PredictRequest):
    try:
        v_enc = venue_le.transform([req.venue])[0]
    except ValueError:
        v_enc = 0
        
    t1_enc = team_le.transform([req.team1])[0]
    t2_enc = team_le.transform([req.team2])[0]
    
    toss_winner_is_t1 = 1 if req.toss_winner == req.team1 else 0
    toss_decision_bat = 1 if req.toss_decision.lower() == 'bat' else 0
    
    t1_elo = get_team_elo(req.team1, req.team1_players)
    t2_elo = get_team_elo(req.team2, req.team2_players)
    
    t1_form = latest_stats['team_recent_form'].get(req.team1, 0.5)
    t2_form = latest_stats['team_recent_form'].get(req.team2, 0.5)
    
    global_bias = latest_stats['global_chase_bias']
    venue_bias = latest_stats['venue_chase_bias'].get(req.venue, global_bias)
    
    is_impact_player_era = 1 

    feature_cols = [
        'venue_enc', 't1_enc', 't2_enc', 
        'toss_winner_is_t1', 'toss_decision_bat',
        't1_player_form_elo', 't2_player_form_elo',
        't1_recent_form', 't2_recent_form',
        'global_chase_bias', 'venue_chase_bias',
        'is_impact_player_era'
    ]
    
    X_dict = {
        'venue_enc': v_enc, 't1_enc': t1_enc, 't2_enc': t2_enc,
        'toss_winner_is_t1': toss_winner_is_t1,
        'toss_decision_bat': toss_decision_bat,
        't1_player_form_elo': t1_elo,
        't2_player_form_elo': t2_elo,
        't1_recent_form': t1_form,
        't2_recent_form': t2_form,
        'global_chase_bias': global_bias,
        'venue_chase_bias': venue_bias,
        'is_impact_player_era': is_impact_player_era
    }
    
    return pd.DataFrame([X_dict])[feature_cols]

@app.get("/meta")
def get_metadata():
    active_teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad"
    ]
    active_venues = [
        "Arun Jaitley Stadium, Delhi", "Barsapara Cricket Stadium, Guwahati",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
        "Eden Gardens, Kolkata", "Himachal Pradesh Cricket Association Stadium, Dharamsala",
        "M Chinnaswamy Stadium, Bengaluru", "MA Chidambaram Stadium, Chepauk, Chennai",
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
        "Narendra Modi Stadium, Ahmedabad", "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
        "Sawai Mansingh Stadium, Jaipur", "Wankhede Stadium, Mumbai"
    ]
    # Build last XI only from stats (for auto-fill), but use official 2026 squads for dropdowns
    team_last_xi = latest_stats.get('team_last_xi', {}) if latest_stats else {}
    # Filter last_xi to only include players in the official 2026 squad
    filtered_last_xi = {}
    for team, xi in team_last_xi.items():
        if team in IPL_2026_SQUADS:
            filtered_last_xi[team] = [p for p in xi if p in IPL_2026_SQUADS[team]]
        else:
            filtered_last_xi[team] = xi

    return {
        "teams": active_teams,
        "venues": active_venues,
        "team_last_xi": filtered_last_xi,
        "team_squads": IPL_2026_SQUADS,
        "all_players": sorted(list(latest_stats.get('player_elos', {}).keys())) if latest_stats else []
    }

@app.post("/predict", response_model=PredictResponse)
def predict_match(req: PredictRequest):
    if req.team1 == req.team2:
        raise HTTPException(status_code=400, detail="Team 1 and Team 2 must be different.")
    if req.toss_winner not in [req.team1, req.team2]:
        raise HTTPException(status_code=400, detail="Toss winner must be one of the playing teams.")

    X_df_forward = build_feature_vector(req)
    prob_forward = float(model.predict_proba(X_df_forward)[0][1])
    
    req_reverse = PredictRequest(
        venue=req.venue, team1=req.team2, team2=req.team1,
        toss_winner=req.toss_winner, toss_decision=req.toss_decision,
        team1_players=req.team2_players, team2_players=req.team1_players
    )
    X_df_reverse = build_feature_vector(req_reverse)
    prob_reverse = float(model.predict_proba(X_df_reverse)[0][1])
    
    t1_win_prob = ((prob_forward + (1.0 - prob_reverse)) / 2.0) * 100 
    
    importances = model.feature_importances_
    features_sorted = sorted(zip(X_df_forward.columns, importances), key=lambda x: x[1], reverse=True)
    top_features = [{"feature": f, "importance": float(i)} for f, i in features_sorted[:5]]

    # Build diagnosis for the frontend
    t1_elo = get_team_elo(req.team1, req.team1_players)
    t2_elo = get_team_elo(req.team2, req.team2_players)
    t1_form = latest_stats['team_recent_form'].get(req.team1, 0.5)
    t2_form = latest_stats['team_recent_form'].get(req.team2, 0.5)
    global_bias = latest_stats['global_chase_bias']
    venue_bias = latest_stats['venue_chase_bias'].get(req.venue, global_bias)

    diagnosis = {
        "t1_elo": round(t1_elo, 1),
        "t2_elo": round(t2_elo, 1),
        "elo_edge": round(t1_elo - t2_elo, 1),
        "t1_form": round(t1_form * 100, 1),
        "t2_form": round(t2_form * 100, 1),
        "venue_chase_bias": round(venue_bias * 100, 1),
        "global_chase_bias": round(global_bias * 100, 1),
        "toss_winner": req.toss_winner,
        "toss_decision": req.toss_decision,
        "toss_advantage": req.toss_winner == req.team1,
        "venue": req.venue,
        "batting_first": req.team1 if (req.toss_winner == req.team1 and req.toss_decision == 'bat') or (req.toss_winner == req.team2 and req.toss_decision == 'field') else req.team2,
        "chasing": req.team2 if (req.toss_winner == req.team1 and req.toss_decision == 'bat') or (req.toss_winner == req.team2 and req.toss_decision == 'field') else req.team1,
    }

    proj_t1, proj_t2, prog_t1, prog_t2, desc = simulate_match_progression(
        req.team1, req.team2, t1_elo, t2_elo, t1_form, t2_form, venue_bias, req.toss_winner, req.toss_decision, t1_win_prob
    )

    return PredictResponse(
        win_probability=t1_win_prob,
        team1=req.team1,
        team2=req.team2,
        top_features=top_features,
        input_summary=req.dict(),
        diagnosis=diagnosis,
        projected_score_t1=proj_t1,
        projected_score_t2=proj_t2,
        run_progression_t1=prog_t1,
        run_progression_t2=prog_t2,
        match_description=desc
    )

