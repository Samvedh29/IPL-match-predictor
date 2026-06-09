import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

def evaluate_2026():
    # We will import the prepare_data function from our advanced script
    # to reconstruct the feature set with the proper Elo ratings.
    from train_advanced_model import prepare_data
    
    print("[*] Rebuilding features for 2026 evaluation...")
    df, _ = prepare_data()
    
    # Filter for IPL 2026 matches (matches starting from late March 2026)
    df_2026 = df[df['date'] >= '2026-03-20'].copy()
    
    if len(df_2026) == 0:
        print("No 2026 matches found in the dataset.")
        return
        
    print(f"\n[*] Found {len(df_2026)} matches for IPL 2026 in the dataset.")
    
    # Load model and encoders
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_post_toss.pkl"))
    venue_le = joblib.load(os.path.join(MODEL_DIR, "venue_le.pkl"))
    team_le = joblib.load(os.path.join(MODEL_DIR, "team_le.pkl"))
    
    # Encode features
    # Handle unseen venues safely
    known_venues = set(venue_le.classes_)
    df_2026['venue_clean'] = df_2026['venue'].apply(lambda x: x if x in known_venues else venue_le.classes_[0])
    df_2026['venue_enc'] = venue_le.transform(df_2026['venue_clean'])
    
    df_2026['t1_enc'] = team_le.transform(df_2026['team1'])
    df_2026['t2_enc'] = team_le.transform(df_2026['team2'])
    
    # The impact player era is True for all 2026
    df_2026['is_impact_player_era'] = 1
    
    feature_cols = [
        'venue_enc', 't1_enc', 't2_enc', 
        'toss_winner_is_t1', 'toss_decision_bat',
        't1_player_form_elo', 't2_player_form_elo',
        't1_recent_form', 't2_recent_form',
        'global_chase_bias', 'venue_chase_bias',
        'is_impact_player_era'
    ]
    
    X = df_2026[feature_cols]
    y_true = df_2026['target_t1_win']
    
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    correct = 0
    print("\n" + "="*80)
    print(f"{'Date':<12} | {'Matchup':<40} | {'Toss Dec':<10} | {'Predicted Winner':<25} | {'Result':<10}")
    print("-" * 80)
    
    for idx, row in df_2026.reset_index(drop=True).iterrows():
        t1 = row['team1']
        t2 = row['team2']
        t1_won = row['target_t1_win'] == 1
        
        prob_t1 = probs[idx]
        pred_t1_wins = preds[idx] == 1
        
        predicted_winner = t1 if pred_t1_wins else t2
        actual_winner = t1 if t1_won else t2
        
        toss_dec = "Bat" if row['toss_decision_bat'] else "Field"
        
        is_correct = (pred_t1_wins == t1_won)
        if is_correct:
            correct += 1
            
        matchup = f"{t1} vs {t2}"
        date_str = row['date'].strftime('%Y-%m-%d')
        res_str = "OK" if is_correct else "WRONG"
        
        pred_str = f"{predicted_winner} ({prob_t1*100:.1f}%)" if pred_t1_wins else f"{predicted_winner} ({(1-prob_t1)*100:.1f}%)"
        
        print(f"{date_str:<12} | {matchup:<40} | {toss_dec:<10} | {pred_str:<25} | {res_str:<10}")

    acc = correct / len(df_2026)
    print("=" * 80)
    print(f"OVERALL ACCURACY FOR IPL 2026: {correct}/{len(df_2026)} = {acc*100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_2026()
