import os
import urllib.request
import zipfile
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cricsheet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "ipl": "https://cricsheet.org/downloads/ipl_json.zip",
    "t20i": "https://cricsheet.org/downloads/t20s_json.zip"
}

def download_and_extract(name, url):
    zip_path = DATA_DIR / f"{name}.zip"
    extract_dir = DATA_DIR / name
    
    if not extract_dir.exists():
        print(f"[*] Downloading {name} data from Cricsheet...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[*] Extracting {name} data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)
    else:
        print(f"[+] {name} data already exists.")
    return extract_dir

def build_match_metadata(extract_dir, match_type):
    """Parses JSON files to extract match metadata and playing XIs."""
    print(f"[*] Parsing {match_type} metadata...")
    matches = []
    
    # Process files
    files = list(extract_dir.glob("*.json"))
    for file_path in tqdm(files, desc=f"Parsing {match_type}"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info = data.get("info", {})
            
            # Skip matches with no result
            if "outcome" in info and "result" in info["outcome"] and info["outcome"]["result"] in ["no result", "tie"]:
                # We can handle ties later, but skip no results
                if info["outcome"]["result"] == "no result":
                    continue
            
            date = info.get("dates", [""])[0]
            venue = info.get("venue", "Unknown")
            teams = info.get("teams", [])
            toss = info.get("toss", {})
            toss_winner = toss.get("winner", "")
            toss_decision = toss.get("decision", "")
            
            # Get winner
            winner = ""
            if "outcome" in info and "winner" in info["outcome"]:
                winner = info["outcome"]["winner"]
            elif "outcome" in info and "result" in info["outcome"] and info["outcome"]["result"] == "tie":
                # If tied and went to super over, winner is in eliminator
                winner = info["outcome"].get("eliminator", "Tie")
            
            if len(teams) != 2 or not winner or winner == "Tie":
                continue
                
            # Get Playing XI
            players = info.get("players", {})
            team1 = teams[0]
            team2 = teams[1]
            team1_xi = players.get(team1, [])
            team2_xi = players.get(team2, [])
            
            matches.append({
                "match_id": file_path.stem,
                "match_type": match_type,
                "date": date,
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "team1_xi": ",".join(team1_xi),
                "team2_xi": ",".join(team2_xi)
            })
        except Exception as e:
            continue
            
    df = pd.DataFrame(matches)
    df = df.sort_values("date").reset_index(drop=True)
    out_path = DATA_DIR / f"{match_type}_metadata.csv"
    df.to_csv(out_path, index=False)
    print(f"[+] Saved {len(df)} matches to {out_path}")
    return df

if __name__ == "__main__":
    ipl_dir = download_and_extract("ipl", URLS["ipl"])
    t20i_dir = download_and_extract("t20i", URLS["t20i"])
    
    build_match_metadata(ipl_dir, "ipl")
    build_match_metadata(t20i_dir, "t20i")
    print("[+] Data ingestion complete.")
