# 🏏 IPL Match Predictor

An AI-powered IPL match win probability predictor built with **XGBoost**, **FastAPI**, and **React**. Predicts pre-match win probabilities using player Elo ratings, team recent form, toss data, venue chase bias, and more.

---

## ✨ Features

- 🤖 **ML-powered predictions** using an XGBoost model trained on historical IPL data
- 📊 **Player Elo ratings** — select your playing XI to get squad-adjusted predictions
- 🔁 **Symmetry-corrected probabilities** — forward + reverse inference for balanced output
- 🏟️ **Venue-aware** — accounts for venue-specific chase bias
- 📈 **Feature importance** breakdown for every prediction
- 🔍 **Diagnosis panel** — see ELO edge, form, toss advantage, and more
- ⚡ **Real-time API** with FastAPI backend and React + Vite frontend

---

## 🛠️ Tech Stack

| Layer    | Technology                          |
|----------|--------------------------------------|
| Frontend | React 19, Vite, Tailwind CSS        |
| Backend  | FastAPI, Uvicorn                    |
| ML Model | XGBoost, scikit-learn, pandas       |
| AI       | Google Generative AI (Gemini API)   |

---

## 📁 Project Structure

```
t20-predictor/
├── backend/
│   ├── main.py                  # FastAPI app & prediction endpoints
│   ├── train_model.py           # Base model training
│   ├── train_advanced_model.py  # Advanced XGBoost training
│   ├── feature_engineering.py   # Feature construction
│   ├── generate_dataset.py      # Dataset generation from raw data
│   ├── ingest_cricsheet.py      # Cricsheet data ingestion
│   ├── export_latest_stats.py   # Export player/team stats
│   ├── ipl_2026_squads.py       # Official 2026 IPL squad data
│   ├── evaluate_2026.py         # 2026 season evaluation
│   ├── validate_real_matches.py # Real match validation
│   ├── requirements.txt
│   └── model/                   # Trained model artifacts (not tracked)
├── frontend/
│   ├── src/                     # React components & pages
│   ├── public/
│   └── package.json
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo GEMINI_API_KEY=your_key_here > .env

# Start the backend server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Swagger docs: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## 🔌 API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/meta`    | Get teams, venues, squads & players |
| POST   | `/predict` | Get win probability prediction      |

### Example Prediction Request

```json
POST /predict
{
  "venue": "Wankhede Stadium, Mumbai",
  "team1": "Mumbai Indians",
  "team2": "Chennai Super Kings",
  "toss_winner": "Mumbai Indians",
  "toss_decision": "bat",
  "team1_players": ["Rohit Sharma", "Ishan Kishan", "..."],
  "team2_players": ["MS Dhoni", "Ruturaj Gaikwad", "..."]
}
```

---

## 📊 Model Details

The XGBoost model uses the following features:

- `venue_enc` — Encoded venue
- `t1_enc` / `t2_enc` — Encoded team identifiers
- `toss_winner_is_t1` — Whether Team 1 won the toss
- `toss_decision_bat` — Whether the toss winner chose to bat
- `t1_player_form_elo` / `t2_player_form_elo` — Squad-averaged Elo ratings
- `t1_recent_form` / `t2_recent_form` — Recent win rate (last N matches)
- `venue_chase_bias` — Historical chase win % at this venue
- `global_chase_bias` — Overall IPL chase win %
- `is_impact_player_era` — Flag for impact player rule era

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Samvedh K M** — [@Samvedh29](https://github.com/Samvedh29)
