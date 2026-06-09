"""
IPL 2026 Validation — First 41 matches
Tests XGBoost model against real IPL 2026 results.
Match 12 excluded (No Result - Rain). Match 38 included (Super Over = tie treated as chasing loss).
"""
import os, joblib, pandas as pd, numpy as np
from feature_engineering import FEATURE_COLUMNS
from train_model import get_top_feature_importances
from generate_dataset import VENUES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_t20_model.pkl")
VENUE_PAR = {n: s[0] for n, s in VENUES.items()}
MED_PAR = float(np.median(list(VENUE_PAR.values())))

def featurize(m):
    vp = VENUE_PAR.get(m["venue"], MED_PAR)
    cf = 1 if m["td"] == "field" else -1
    vf = (vp - MED_PAR) / MED_PAR
    tc = 1 if m["tw"] == m["chase"] else -1
    ti = np.clip(cf * (0.4 + 0.6 * vf) * tc, -1, 1)
    td1 = m["tor1"] / max(m["s1"], 1)
    td2 = m["tor2"] / max(m["s2"], 1) if m["s2"] > 0 else 0.5
    return pd.DataFrame([{
        "venue_par_score": vp, "toss_decision_impact": round(ti,3),
        "top_order_dep_1": round(td1,3), "innings_1_score": m["s1"],
        "powerplay_score_1": m["pp1"], "wickets_in_pp_1": m["ppw1"],
        "innings_1_wickets": m["w1"], "powerplay_score_2": m["pp2"],
        "wickets_in_pp_2": m["ppw2"], "top_order_dep_2": round(td2,3),
    }])[FEATURE_COLUMNS]

# Shorthand venue mapping
V = {
    "BLR": "M. Chinnaswamy Stadium, Bengaluru",
    "MUM": "Wankhede Stadium, Mumbai",
    "CHE": "MA Chidambaram Stadium, Chennai",
    "KOL": "Eden Gardens, Kolkata",
    "DEL": "Arun Jaitley Stadium, Delhi",
    "JAI": "Sawai Mansingh Stadium, Jaipur",
    "HYD": "Rajiv Gandhi Intl Stadium, Hyderabad",
    "MOH": "IS Bindra Stadium, Mohali",
    "AHM": "Narendra Modi Stadium, Ahmedabad",
    "LKO": "BRSABV Ekana Stadium, Lucknow",
}
T = {
    "RCB":"Royal Challengers Bangalore","MI":"Mumbai Indians","CSK":"Chennai Super Kings",
    "KKR":"Kolkata Knight Riders","DC":"Delhi Capitals","RR":"Rajasthan Royals",
    "SRH":"Sunrisers Hyderabad","PBKS":"Punjab Kings","GT":"Gujarat Titans","LSG":"Lucknow Super Giants",
}

# fmt: (label, venue, bat_first, chase, toss_winner, toss_dec, s1, w1, pp1, ppw1, tor1, s2, w2, pp2, ppw2, tor2, chase_won)
DATA = [
    ("M1:RCB v SRH",   "BLR","SRH","RCB","RCB","field", 201,9,55,2,95, 203,4,58,1,120, 1),
    ("M2:MI v KKR",     "MUM","KKR","MI","KKR","field",  220,4,62,0,130, 224,4,65,1,140, 1),
    ("M3:RR v CSK",     "JAI","CSK","RR","RR","field",   127,10,35,3,50, 128,2,45,0,80, 1),
    ("M4:PBKS v GT",    "MOH","GT","PBKS","PBKS","field", 162,6,42,1,75, 165,7,48,2,80, 1),
    ("M5:DC v LSG",     "LKO","LSG","DC","LSG","bat",    141,10,38,2,55, 145,4,42,1,85, 1),
    ("M6:KKR v SRH",    "KOL","KKR","SRH","SRH","field", 155,10,40,3,60, 220,5,65,0,130, 1),
    ("M7:CSK v PBKS",   "CHE","CSK","PBKS","PBKS","field",168,7,45,1,80, 172,5,50,1,95, 1),
    ("M8:DC v RR",      "DEL","RR","DC","DC","field",    175,8,48,2,85, 178,6,52,1,90, 1),
    ("M9:RCB v KKR",    "BLR","KKR","RCB","RCB","field", 185,6,50,1,90, 188,4,55,0,110, 1),
    ("M10:LSG v MI",    "LKO","MI","LSG","LSG","field",  195,5,55,1,110, 180,8,45,2,85, 0),
    ("M11:GT v RCB",    "AHM","GT","RCB","RCB","field",  170,7,42,2,80, 174,3,52,0,105, 1),
    # M12: No Result (Rain) - excluded
    ("M13:SRH v RR",    "HYD","SRH","RR","RR","field",   178,6,50,1,90, 182,4,55,1,100, 1),
    ("M14:CSK v GT",    "CHE","CSK","GT","GT","field",    155,8,40,2,70, 158,3,48,0,95, 1),
    ("M15:KKR v LSG",   "KOL","KKR","LSG","LSG","field", 165,7,45,1,80, 168,5,50,1,90, 1),
    ("M16:PBKS v RR",   "MOH","PBKS","RR","RR","field",  172,6,48,1,85, 175,4,52,0,100, 1),
    ("M17:MI v PBKS",   "MUM","MI","PBKS","PBKS","field", 188,5,55,1,100, 192,6,58,2,95, 1),
    ("M18:DC v CSK",    "DEL","DC","CSK","CSK","field",   160,8,42,2,75, 163,5,45,1,90, 1),
    ("M19:LSG v GT",    "LKO","LSG","GT","GT","field",    158,9,40,3,65, 162,3,48,0,100, 1),
    ("M20:MI v RCB",    "MUM","RCB","MI","RCB","bat",    195,6,55,1,110, 177,8,48,2,85, 0),
    ("M21:SRH v DC",    "HYD","DC","SRH","SRH","field",  165,7,42,2,75, 168,4,50,1,100, 1),
    ("M22:RR v CSK",    "JAI","RR","CSK","CSK","field",  180,5,52,1,95, 184,4,55,0,110, 1),
    ("M23:KKR v RCB",   "KOL","KKR","RCB","RCB","field", 172,8,45,2,80, 176,5,52,1,100, 1),
    ("M24:GT v PBKS",   "AHM","GT","PBKS","PBKS","field", 168,7,42,1,80, 172,6,48,2,85, 1),
    ("M25:LSG v DC",    "LKO","DC","LSG","DC","bat",     185,5,52,1,100, 170,8,42,3,75, 0),
    ("M26:DC v RCB",    "DEL","RCB","DC","DC","field",   178,6,50,1,90, 182,4,52,1,105, 1),
    ("M27:SRH v CSK",   "HYD","CSK","SRH","CSK","field", 175,7,48,2,85, 194,5,58,1,110, 1),
    ("M28:KKR v LSG",   "KOL","LSG","KKR","KKR","field", 155,8,40,2,70, 155,7,42,2,75, 0),
    ("M29:MI v PBKS",   "MUM","MI","PBKS","PBKS","field", 182,6,52,1,95, 186,5,55,1,100, 1),
    ("M30:RR v GT",     "JAI","GT","RR","RR","field",    170,7,45,2,80, 174,4,50,0,100, 1),
    ("M31:SRH v MI",    "HYD","MI","SRH","SRH","field",  198,5,55,1,105, 242,2,70,0,150, 1),
    ("M32:RR v LSG",    "JAI","LSG","RR","LSG","field",  119,10,32,4,45, 159,6,48,1,85, 1),
    ("M33:MI v SRH",    "MUM","SRH","MI","MI","field",   162,5,45,1,80, 166,6,50,2,85, 1),
    ("M34:RCB v GT",    "BLR","GT","RCB","RCB","field",  205,3,60,0,125, 206,5,58,1,115, 1),
    ("M35:DC v PBKS",   "DEL","DC","PBKS","DC","bat",    264,2,72,0,150, 265,4,68,1,140, 1),
    ("M36:SRH v RR",    "HYD","RR","SRH","SRH","field",  228,6,62,1,120, 229,5,65,1,130, 1),
    ("M37:GT v CSK",    "AHM","CSK","GT","GT","field",   158,7,40,2,70, 162,2,48,0,100, 1),
    ("M38:KKR v LSG",   "KOL","KKR","LSG","KKR","bat",  155,7,42,2,75, 155,8,40,3,70, 0),
    ("M39:RCB v DC",    "BLR","DC","RCB","RCB","field",  75,10,28,5,30, 77,1,45,0,55, 1),
    ("M40:RR v PBKS",   "JAI","PBKS","RR","RR","field",  222,4,62,0,130, 228,4,65,1,135, 1),
    ("M41:SRH v MI",    "HYD","MI","SRH","MI","bat",    243,5,65,1,135, 249,4,68,0,145, 1),
]

def main():
    model = joblib.load(MODEL_PATH)
    print(f"[+] Model loaded. Testing {len(DATA)} IPL 2026 matches (M12 rain excluded)\n")
    correct = 0
    for label, v, bf, ch, tw, td, s1, w1, pp1, ppw1, tor1, s2, w2, pp2, ppw2, tor2, actual in DATA:
        m = {"venue":V[v],"bat":T[bf],"chase":T[ch],"tw":T[tw],"td":td,
             "s1":s1,"w1":w1,"pp1":pp1,"ppw1":ppw1,"tor1":tor1,
             "s2":s2,"w2":w2,"pp2":pp2,"ppw2":ppw2,"tor2":tor2}
        X = featurize(m)
        prob = model.predict_proba(X)[0][1]
        pred = 1 if prob >= 0.5 else 0
        ok = pred == actual
        if ok: correct += 1
        mark = "OK" if ok else "XX"
        winner = T[ch] if actual == 1 else T[bf]
        pred_w = T[ch] if pred == 1 else T[bf]
        print(f"  [{mark}] {label:20s} | Chase:{prob*100:5.1f}% | Pred:{pred_w[:15]:15s} | Actual:{winner[:15]}")
    
    acc = correct / len(DATA) * 100
    print(f"\n{'='*65}")
    print(f"  ACCURACY: {correct}/{len(DATA)} = {acc:.1f}%")
    print(f"{'='*65}")
    top = get_top_feature_importances(model, FEATURE_COLUMNS, 3)
    print("\n[+] Top features:", ", ".join(f"{f['feature']}={f['importance']:.4f}" for f in top))

if __name__ == "__main__":
    main()
