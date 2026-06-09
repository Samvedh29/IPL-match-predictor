import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "cricsheet")
MODEL_DIR = os.path.join(BASE_DIR, "model")

IPL_CSV = os.path.join(DATA_DIR, "ipl_metadata.csv")
T20I_CSV = os.path.join(DATA_DIR, "t20i_metadata.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

class PlayerEloSystem:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor
        self.ratings = {}

    def get_rating(self, player_name):
        return self.ratings.get(player_name, 1500)

    def get_team_avg_rating(self, players):
        if not players: return 1500
        return sum(self.get_rating(p) for p in players) / len(players)

    def update_ratings(self, team1_players, team2_players, team1_won):
        r1 = self.get_team_avg_rating(team1_players)
        r2 = self.get_team_avg_rating(team2_players)
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1
        a1 = 1 if team1_won else 0
        a2 = 0 if team1_won else 1
        for p in team1_players:
            self.ratings[p] = self.get_rating(p) + self.k_factor * (a1 - e1)
        for p in team2_players:
            self.ratings[p] = self.get_rating(p) + self.k_factor * (a2 - e2)

def prepare_data():
    print("[*] Loading Cricsheet data...")
    ipl_df = pd.read_csv(IPL_CSV)
    t20i_df = pd.read_csv(T20I_CSV)
    
    combined = pd.concat([ipl_df, t20i_df], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.dropna(subset=['date'])
    combined = combined.sort_values('date').reset_index(drop=True)
    
    player_elo = PlayerEloSystem()
    
    # Trackers for recent form and batting-heavy meta (Chase Bias)
    team_recent_history = {} # key: team_name, val: list of 1s (wins) and 0s (losses)
    global_chase_history = []
    venue_chase_history = {} # key: venue, val: list of 1s (chase win) and 0s (defend win)
    
    features = []
    
    print("[*] Processing matches chronologically to build recent form features...")
    for _, row in combined.iterrows():
        t1 = row['team1']
        t2 = row['team2']
        winner = row['winner']
        venue = row['venue']
        
        if not isinstance(winner, str) or winner not in [t1, t2]:
            continue
            
        t1_won = (winner == t1)
        
        t1_xi = str(row['team1_xi']).split(',') if pd.notna(row['team1_xi']) else []
        t2_xi = str(row['team2_xi']).split(',') if pd.notna(row['team2_xi']) else []
        
        # Pre-match Player Forms
        pre_t1_p_elo = player_elo.get_team_avg_rating(t1_xi)
        pre_t2_p_elo = player_elo.get_team_avg_rating(t2_xi)
        
        # Pre-match Team Recent Form (Last 10 matches)
        if t1 not in team_recent_history: team_recent_history[t1] = []
        if t2 not in team_recent_history: team_recent_history[t2] = []
        
        t1_history = team_recent_history[t1][-10:]
        t2_history = team_recent_history[t2][-10:]
        
        t1_recent_form = sum(t1_history) / len(t1_history) if len(t1_history) > 0 else 0.5
        t2_recent_form = sum(t2_history) / len(t2_history) if len(t2_history) > 0 else 0.5
        
        # Batting-Heavy Meta Tracker (Chase Bias)
        if row['toss_decision'] == 'field':
            chasing_team = row['toss_winner']
        else:
            chasing_team = t1 if row['toss_winner'] == t2 else t2
            
        chase_successful = 1 if winner == chasing_team else 0
        
        # Pre-match global chase bias (last 50 matches)
        global_chase_bias = sum(global_chase_history[-50:]) / min(len(global_chase_history), 50) if global_chase_history else 0.5
        
        # Pre-match venue chase bias (last 15 matches)
        if venue not in venue_chase_history: venue_chase_history[venue] = []
        v_hist = venue_chase_history[venue][-15:]
        venue_chase_bias = sum(v_hist) / len(v_hist) if len(v_hist) > 0 else 0.5
        
        if row['match_type'] == 'ipl':
            toss_winner = 1 if row['toss_winner'] == t1 else 0
            toss_decision = 1 if row['toss_decision'] == 'bat' else 0
            
            features.append({
                'date': row['date'],
                'venue': venue,
                'team1': t1,
                'team2': t2,
                'toss_winner_is_t1': toss_winner,
                'toss_decision_bat': toss_decision,
                't1_player_form_elo': pre_t1_p_elo,
                't2_player_form_elo': pre_t2_p_elo,
                't1_recent_form': t1_recent_form,
                't2_recent_form': t2_recent_form,
                'global_chase_bias': global_chase_bias,
                'venue_chase_bias': venue_chase_bias,
                'target_t1_win': 1 if t1_won else 0
            })
            
        # Post-match updates
        player_elo.update_ratings(t1_xi, t2_xi, t1_won)
        
        team_recent_history[t1].append(1 if t1_won else 0)
        team_recent_history[t2].append(0 if t1_won else 1)
        
        global_chase_history.append(chase_successful)
        venue_chase_history[venue].append(chase_successful)
        
    df_train = pd.DataFrame(features)
    return df_train, player_elo

def train_model():
    df, player_elo = prepare_data()
    
    venue_le = LabelEncoder()
    df['venue_enc'] = venue_le.fit_transform(df['venue'])
    
    team_le = LabelEncoder()
    all_teams = pd.concat([df['team1'], df['team2']]).unique()
    team_le.fit(all_teams)
    df['t1_enc'] = team_le.transform(df['team1'])
    df['t2_enc'] = team_le.transform(df['team2'])
    
    max_date = df['date'].max()
    days_diff = (max_date - df['date']).dt.days
    df['sample_weight'] = np.exp(-days_diff / 1000)
    
    impact_player_date = pd.to_datetime('2023-03-31')
    df['is_impact_player_era'] = (df['date'] >= impact_player_date).astype(int)

    feature_cols = [
        'venue_enc', 't1_enc', 't2_enc', 
        'toss_winner_is_t1', 'toss_decision_bat',
        't1_player_form_elo', 't2_player_form_elo',
        't1_recent_form', 't2_recent_form',
        'global_chase_bias', 'venue_chase_bias',
        'is_impact_player_era'
    ]
    
    X = df[feature_cols]
    y = df['target_t1_win']
    weights = df['sample_weight']
    
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train, w_test = weights.iloc[:split_idx], weights.iloc[split_idx:]
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], verbose=False)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print("\n" + "="*50)
    print(f"[+] Recent Form Model Accuracy: {acc*100:.1f}%")
    print(f"[+] Recent Form Model AUC: {auc:.3f}")
    print("="*50)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_post_toss.pkl"))
    joblib.dump(venue_le, os.path.join(MODEL_DIR, "venue_le.pkl"))
    joblib.dump(team_le, os.path.join(MODEL_DIR, "team_le.pkl"))
    joblib.dump(player_elo.ratings, os.path.join(MODEL_DIR, "player_elos.pkl"))
    
    print("\n[+] Top Features Importance:")
    importances = model.feature_importances_
    feats = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values('Importance', ascending=False)
    for _, row in feats.iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f}")

if __name__ == "__main__":
    train_model()
