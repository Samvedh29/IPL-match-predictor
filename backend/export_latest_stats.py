import os
import json
import joblib
from train_advanced_model import prepare_data

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

def export():
    print("Running prepare_data to export latest stats...")
    df, player_elo = prepare_data()
    
    # 1. Global Chase Bias
    global_chase_bias = float(df['global_chase_bias'].iloc[-1])
    
    # 2. Team Recent Form
    # We look at the last time a team played to get their current form
    team_recent_form = {}
    team_avg_player_elo = {}
    for team in pd.concat([df['team1'], df['team2']]).unique():
        # Find last match for this team
        t1_matches = df[df['team1'] == team]
        t2_matches = df[df['team2'] == team]
        
        last_t1_idx = t1_matches.index[-1] if len(t1_matches) > 0 else -1
        last_t2_idx = t2_matches.index[-1] if len(t2_matches) > 0 else -1
        
        if last_t1_idx > last_t2_idx:
            form = float(t1_matches.iloc[-1]['t1_recent_form'])
            avg_elo = float(t1_matches.iloc[-1]['t1_player_form_elo'])
        elif last_t2_idx > -1:
            form = float(t2_matches.iloc[-1]['t2_recent_form'])
            avg_elo = float(t2_matches.iloc[-1]['t2_player_form_elo'])
        else:
            form = 0.5
            avg_elo = 1500.0
        team_recent_form[team] = form
        team_avg_player_elo[team] = avg_elo

        
    # 3. Venue Chase Bias
    venue_chase_bias = {}
    for venue in df['venue'].unique():
        v_matches = df[df['venue'] == venue]
        if len(v_matches) > 0:
            venue_chase_bias[venue] = float(v_matches.iloc[-1]['venue_chase_bias'])
            
    # Read the raw metadata to get the actual players
    from train_advanced_model import IPL_CSV, T20I_CSV
    ipl_df = pd.read_csv(IPL_CSV)
    t20i_df = pd.read_csv(T20I_CSV)
    combined = pd.concat([ipl_df, t20i_df], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    active_teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad"
    ]
    
    current_year = combined['date'].dt.year.max()
    
    team_last_xi = {}
    team_squads = {}
    
    # Calculate player last played dates globally to filter out retired players
    player_last_date = {}
    for _, row in combined.iterrows():
        p1 = [p.strip() for p in str(row['team1_xi']).split(',') if pd.notna(row['team1_xi'])]
        p2 = [p.strip() for p in str(row['team2_xi']).split(',') if pd.notna(row['team2_xi'])]
        for p in p1 + p2:
            if p not in player_last_date or row['date'] > player_last_date[p]:
                player_last_date[p] = row['date']

    for team in active_teams:
        # Find last match for pre-fill
        t1_matches = combined[combined['team1'] == team]
        t2_matches = combined[combined['team2'] == team]
        last_t1_idx = t1_matches.index[-1] if len(t1_matches) > 0 else -1
        last_t2_idx = t2_matches.index[-1] if len(t2_matches) > 0 else -1
        
        if last_t1_idx > last_t2_idx:
            xi_str = t1_matches.iloc[-1]['team1_xi']
        elif last_t2_idx > -1:
            xi_str = t2_matches.iloc[-1]['team2_xi']
        else:
            xi_str = ""
        team_last_xi[team] = [p.strip() for p in str(xi_str).split(',')] if pd.notna(xi_str) and xi_str else []

        # Squad: Players who have played for this franchise in the last 2 years
        squad = set()
        t_matches = combined[((combined['team1'] == team) | (combined['team2'] == team)) & (combined['date'].dt.year >= current_year - 1)]
        
        for _, row in t_matches.iterrows():
            if row['team1'] == team:
                if pd.notna(row['team1_xi']):
                    squad.update([p.strip() for p in str(row['team1_xi']).split(',')])
            else:
                if pd.notna(row['team2_xi']):
                    squad.update([p.strip() for p in str(row['team2_xi']).split(',')])
        
        # Further filter: Player must have played ANY match in the last 2 seasons globally
        # (This handles cases where a player was in the squad last year but has since retired/disappeared)
        active_squad = []
        for p in squad:
            if p in player_last_date and player_last_date[p].year >= current_year - 1:
                active_squad.append(p)
                
        team_squads[team] = sorted(active_squad)
            
    # 4. Player Elos
    player_elos = player_elo.ratings
    
    stats = {
        'global_chase_bias': global_chase_bias,
        'team_recent_form': team_recent_form,
        'team_avg_player_elo': team_avg_player_elo,
        'team_last_xi': team_last_xi,
        'team_squads': team_squads,
        'venue_chase_bias': venue_chase_bias,
        'player_elos': player_elos
    }
    
    out_path = os.path.join(MODEL_DIR, "latest_stats.json")
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Exported latest stats to {out_path}")

if __name__ == "__main__":
    import pandas as pd
    export()
