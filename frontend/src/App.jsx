import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? 'http://localhost:8001' : 'https://samvedh29-t20-predictor.hf.space')

/* ─────────────────────────────────────────────
   Win Probability Gauge
   ───────────────────────────────────────────── */
function WinGauge({ probability, team1, team2 }) {
  const p = Math.round(probability ?? 50)
  const circumference = 2 * Math.PI * 90
  const offset = circumference - (p / 100) * circumference

  const getColor = (v) => {
    if (v >= 70) return '#a6e3a1'
    if (v >= 50) return '#f9e2af'
    if (v >= 30) return '#fab387'
    return '#f38ba8'
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-56 h-56">
        <svg viewBox="0 0 200 200" className="w-full h-full -rotate-90">
          <circle cx="100" cy="100" r="90" fill="none" stroke="#313244" strokeWidth="14" />
          <circle
            cx="100" cy="100" r="90" fill="none"
            stroke={getColor(p)}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.5s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-bold" style={{ color: getColor(p) }}>
            {p}%
          </span>
          <span className="text-xs text-text-muted mt-1 uppercase">WIN PROBABILITY</span>
        </div>
      </div>
      <div className="text-center space-y-1">
        <p className="text-sm text-text-muted">
          <span className="font-semibold text-text">{team1 || 'Team 1'}</span> winning
        </p>
        <p className="text-xs text-text-muted">
          vs {team2 || 'Team 2'}
        </p>
      </div>
    </div>
  )
}


/* ─────────────────────────────────────────────
   Feature Importance Bars
   ───────────────────────────────────────────── */
function FeatureBars({ features }) {
  if (!features || features.length === 0) return null
  const maxImp = Math.max(...features.map(f => f.importance))

  const friendlyNames = {
    't2_recent_form': 'Team 2 Recent Form',
    't1_recent_form': 'Team 1 Recent Form',
    't1_player_form_elo': 'Team 1 Player Forms',
    't2_player_form_elo': 'Team 2 Player Forms',
    'global_chase_bias': 'Global Batting Meta',
    'venue_chase_bias': 'Venue Batting Meta',
    'is_impact_player_era': 'Impact Player Era',
    'toss_winner_is_t1': 'Toss Winner',
    'toss_decision_bat': 'Toss Decision',
    'venue_enc': 'Venue Impact',
    't1_enc': 'Team 1 Identity',
    't2_enc': 'Team 2 Identity'
  }

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-text-muted uppercase tracking-wider">
        Key Predictive Factors
      </h3>
      {features.map((f, i) => (
        <div key={i} className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-text">{friendlyNames[f.feature] || f.feature}</span>
            <span className="text-text-muted">{(f.importance * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-surface-lighter rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${(f.importance / maxImp) * 100}%`,
                background: `linear-gradient(90deg, var(--color-primary), var(--color-accent))`,
                transition: 'width 0.8s ease',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}





/* ─────────────────────────────────────────────
   Match Projection Panel
   ───────────────────────────────────────────── */
function MatchProjection({ prediction }) {
  if (!prediction || !prediction.run_progression_t1) return null;

  const { team1, team2, projected_score_t1, projected_score_t2, run_progression_t1, run_progression_t2, match_description } = prediction;

  const chartData = [];
  for (let i = 0; i < 20; i++) {
    chartData.push({
      over: i + 1,
      [team1]: run_progression_t1[i],
      [team2]: run_progression_t2[i]
    });
  }

  return (
    <div className="bg-surface rounded-3xl border border-surface-light/30 p-8 shadow-xl shadow-bg/50 mt-6 space-y-6">
      <h3 className="text-sm font-semibold text-text-muted uppercase tracking-wider">
        Match Projection & Analysis
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-lighter rounded-2xl p-4 flex flex-col items-center justify-center border border-surface-light/20">
            <span className="text-xs text-text-muted mb-1">{team1} Projected</span>
            <span className="text-3xl font-bold text-primary">{projected_score_t1}</span>
        </div>
        <div className="bg-surface-lighter rounded-2xl p-4 flex flex-col items-center justify-center border border-surface-light/20">
            <span className="text-xs text-text-muted mb-1">{team2} Projected</span>
            <span className="text-3xl font-bold text-accent">{projected_score_t2}</span>
        </div>
      </div>

      <div className="text-sm text-text leading-relaxed bg-surface-lighter/50 rounded-2xl p-4 border border-surface-light/20">
        <span className="font-semibold text-text-muted mr-2">Analysis:</span>
        {match_description}
      </div>

      <div className="h-[300px] w-full pt-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#313244" vertical={false} />
            <XAxis dataKey="over" stroke="#a6adc8" fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke="#a6adc8" fontSize={11} tickLine={false} axisLine={false} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e1e2e', border: '1px solid #313244', borderRadius: '12px' }}
              itemStyle={{ fontSize: '12px' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line type="monotone" dataKey={team1} stroke="var(--color-primary)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
            <Line type="monotone" dataKey={team2} stroke="var(--color-accent)" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}


/* ─────────────────────────────────────────────
   Prediction Reasoning Panel
   ───────────────────────────────────────────── */
function PredictionReasoning({ prediction }) {
  if (!prediction?.diagnosis) return null;

  const d = prediction.diagnosis;
  const winProb = Math.round(prediction.win_probability);
  const team1 = prediction.team1;
  const team2 = prediction.team2;
  const favored = winProb >= 50 ? team1 : team2;
  const favoredProb = winProb >= 50 ? winProb : 100 - winProb;

  // Generate reasoning bullets
  const reasons = [];

  // 1. ELO Analysis
  const eloEdge = d.elo_edge;
  if (Math.abs(eloEdge) > 30) {
    const stronger = eloEdge > 0 ? team1 : team2;
    reasons.push({
      icon: '⚡',
      title: 'Squad Strength (Player Elo)',
      detail: `${stronger} has a significantly stronger squad (Elo edge: ${Math.abs(eloEdge).toFixed(0)} pts). Their players have performed better across recent T20 history.`,
      impact: eloEdge > 0 ? 'favors_t1' : 'favors_t2',
      values: { t1: d.t1_elo, t2: d.t2_elo }
    });
  } else if (Math.abs(eloEdge) > 10) {
    const stronger = eloEdge > 0 ? team1 : team2;
    reasons.push({
      icon: '⚡',
      title: 'Squad Strength (Player Elo)',
      detail: `${stronger} holds a slight squad quality advantage (Elo edge: ${Math.abs(eloEdge).toFixed(0)} pts). Individual player form ratings give them a marginal edge.`,
      impact: eloEdge > 0 ? 'favors_t1' : 'favors_t2',
      values: { t1: d.t1_elo, t2: d.t2_elo }
    });
  } else {
    reasons.push({
      icon: '⚡',
      title: 'Squad Strength (Player Elo)',
      detail: `Both squads are evenly matched in player quality (Elo diff: ${Math.abs(eloEdge).toFixed(0)} pts). Neither team has a clear individual talent edge.`,
      impact: 'neutral',
      values: { t1: d.t1_elo, t2: d.t2_elo }
    });
  }

  // 2. Recent Form
  const formDiff = d.t1_form - d.t2_form;
  if (Math.abs(formDiff) > 20) {
    const hotTeam = formDiff > 0 ? team1 : team2;
    const hotForm = formDiff > 0 ? d.t1_form : d.t2_form;
    const coldForm = formDiff > 0 ? d.t2_form : d.t1_form;
    reasons.push({
      icon: '🔥',
      title: 'Recent Form (Last 10 Matches)',
      detail: `${hotTeam} is in excellent form (${hotForm}% win rate in recent matches vs ${coldForm}%). Momentum is a proven factor in T20 cricket.`,
      impact: formDiff > 0 ? 'favors_t1' : 'favors_t2'
    });
  } else if (Math.abs(formDiff) > 10) {
    const betterTeam = formDiff > 0 ? team1 : team2;
    reasons.push({
      icon: '🔥',
      title: 'Recent Form (Last 10 Matches)',
      detail: `${betterTeam} has a slight form advantage (${formDiff > 0 ? d.t1_form : d.t2_form}% vs ${formDiff > 0 ? d.t2_form : d.t1_form}%). Recent momentum provides a minor edge.`,
      impact: formDiff > 0 ? 'favors_t1' : 'favors_t2'
    });
  } else {
    reasons.push({
      icon: '🔥',
      title: 'Recent Form (Last 10 Matches)',
      detail: `Both teams are in similar form (${d.t1_form}% vs ${d.t2_form}%). Neither has a significant momentum advantage heading into this match.`,
      impact: 'neutral'
    });
  }

  // 3. Venue Chase Bias
  const chaseBias = d.venue_chase_bias;
  if (chaseBias > 60) {
    reasons.push({
      icon: '🏟️',
      title: 'Venue Bias — Chase-Friendly',
      detail: `This venue heavily favors the chasing team (${chaseBias}% chase win rate historically). ${d.chasing} benefits from chasing here, with dew and batting-friendly conditions typically aiding the second innings.`,
      impact: d.chasing === team1 ? 'favors_t1' : 'favors_t2'
    });
  } else if (chaseBias < 40) {
    reasons.push({
      icon: '🏟️',
      title: 'Venue Bias — Defend-Friendly',
      detail: `This venue historically favors defending (only ${chaseBias}% chase win rate). ${d.batting_first} benefits from batting first, as this ground tends to slow down in the second innings.`,
      impact: d.batting_first === team1 ? 'favors_t1' : 'favors_t2'
    });
  } else {
    reasons.push({
      icon: '🏟️',
      title: 'Venue Analysis',
      detail: `This venue is balanced for both chasing and defending (${chaseBias}% chase win rate). Neither batting first nor chasing provides a decisive statistical edge at this ground.`,
      impact: 'neutral'
    });
  }

  // 4. Toss Impact
  const tossTeam = d.toss_winner;
  const tossChoice = d.toss_decision;
  const tossAligned = (tossChoice === 'field' && chaseBias > 55) || (tossChoice === 'bat' && chaseBias < 45);
  if (tossAligned) {
    reasons.push({
      icon: '🪙',
      title: 'Toss Decision — Well Aligned',
      detail: `${tossTeam} won the toss and chose to ${tossChoice}, which aligns well with this venue's historical trend. This is a strategically sound call based on ${chaseBias > 55 ? 'the chase-friendly nature' : 'the defend-friendly nature'} of this ground.`,
      impact: tossTeam === team1 ? 'favors_t1' : 'favors_t2'
    });
  } else {
    reasons.push({
      icon: '🪙',
      title: 'Toss Decision',
      detail: `${tossTeam} won the toss and chose to ${tossChoice}. Given the venue's chase bias of ${chaseBias}%, this is a ${Math.abs(chaseBias - 50) < 10 ? 'neutral' : 'potentially suboptimal'} decision.`,
      impact: 'neutral'
    });
  }

  // 5. Global Chase Meta
  const globalBias = d.global_chase_bias;
  reasons.push({
    icon: '📊',
    title: 'Global T20 Meta Trend',
    detail: `The current global chase win rate stands at ${globalBias}%, reflecting the ${globalBias > 52 ? 'batting-heavy meta where chasing teams have an advantage' : globalBias < 48 ? 'bowling-friendly meta favoring defending teams' : 'balanced meta between chasing and defending'} across recent T20 matches worldwide.`,
    impact: 'neutral'
  });

  const getImpactColor = (impact) => {
    if (impact === 'favors_t1') return '#6366f1';
    if (impact === 'favors_t2') return '#f59e0b';
    return '#a6adc8';
  };

  const getImpactLabel = (impact) => {
    if (impact === 'favors_t1') return team1;
    if (impact === 'favors_t2') return team2;
    return 'Neutral';
  };

  return (
    <div className="bg-surface rounded-3xl border border-surface-light/30 p-8 shadow-xl shadow-bg/50 mt-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
            <span className="text-white text-sm">🧠</span>
          </div>
          <div>
            <h3 className="text-sm font-bold text-text uppercase tracking-wider">
              AI Reasoning
            </h3>
            <p className="text-[11px] text-text-muted">Based on historical data analysis</p>
          </div>
        </div>
        <div className="verdict-badge px-4 py-2 rounded-full text-sm font-bold" style={{ background: 'linear-gradient(135deg, var(--color-primary), var(--color-accent))', color: 'white' }}>
          {favored} — {favoredProb}%
        </div>
      </div>

      {/* ELO Comparison Bar */}
      {reasons[0]?.values && (
        <div className="bg-surface-lighter/50 rounded-2xl p-4 border border-surface-light/20 space-y-3">
          <div className="flex justify-between text-xs font-semibold">
            <span className="text-primary">{team1} — {reasons[0].values.t1.toFixed(0)} Elo</span>
            <span className="text-accent">{team2} — {reasons[0].values.t2.toFixed(0)} Elo</span>
          </div>
          <div className="h-3 bg-surface rounded-full overflow-hidden flex">
            <div
              className="h-full elo-bar-fill rounded-l-full"
              style={{
                width: `${(reasons[0].values.t1 / (reasons[0].values.t1 + reasons[0].values.t2)) * 100}%`,
                background: 'linear-gradient(90deg, var(--color-primary), var(--color-primary-light))'
              }}
            />
            <div
              className="h-full elo-bar-fill rounded-r-full"
              style={{
                width: `${(reasons[0].values.t2 / (reasons[0].values.t1 + reasons[0].values.t2)) * 100}%`,
                background: 'linear-gradient(90deg, var(--color-accent), var(--color-accent-light))'
              }}
            />
          </div>
        </div>
      )}

      {/* Reasoning Cards */}
      <div className="space-y-3 reasoning-scroll" style={{ maxHeight: '500px', overflowY: 'auto' }}>
        {reasons.map((r, i) => (
          <div key={i} className="reasoning-card bg-surface-lighter/40 rounded-2xl p-4 border border-surface-light/20 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-lg">{r.icon}</span>
                <span className="text-sm font-semibold text-text">{r.title}</span>
              </div>
              <span
                className="text-[10px] font-bold uppercase px-2 py-1 rounded-full"
                style={{
                  background: getImpactColor(r.impact) + '20',
                  color: getImpactColor(r.impact),
                  border: `1px solid ${getImpactColor(r.impact)}40`
                }}
              >
                {r.impact === 'neutral' ? '≈ Neutral' : `↑ ${getImpactLabel(r.impact)}`}
              </span>
            </div>
            <p className="text-xs text-text-muted leading-relaxed">{r.detail}</p>
          </div>
        ))}
      </div>

      {/* Verdict Summary */}
      <div className="bg-gradient-to-r from-primary/10 to-accent/10 rounded-2xl p-5 border border-primary/20">
        <div className="flex items-start gap-3">
          <span className="text-2xl mt-0.5">⚖️</span>
          <div className="space-y-1">
            <h4 className="text-sm font-bold text-text">Oracle Verdict</h4>
            <p className="text-xs text-text-muted leading-relaxed">
              {favoredProb >= 65
                ? `${favored} is the clear favorite at ${favoredProb}%. The combination of ${eloEdge > 20 ? 'superior squad quality' : 'venue conditions and form'} gives them a strong edge. The model sees this as a relatively one-sided contest based on pre-match data.`
                : favoredProb >= 55
                ? `${favored} edges this matchup at ${favoredProb}%. While they hold the advantage, the margin is thin — a single key performance (toss, powerplay, or dew factor) could swing the result. Historically, matches at this confidence level go either way about ${100 - favoredProb}% of the time.`
                : `This is a genuine coin-flip at ${favoredProb}%. The model cannot separate these two sides significantly. Expect the outcome to hinge on in-match factors like powerplay execution, death bowling, and fielding quality — elements that pre-match models cannot fully capture.`
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}


/* ─────────────────────────────────────────────
   How It Works Section
   ───────────────────────────────────────────── */
function HowItWorks() {
  const steps = [
    {
      num: '01',
      icon: '📡',
      title: 'Data Ingestion',
      desc: 'We ingest thousands of T20 matches from Cricsheet — ball-by-ball data from IPL, T20I, and major T20 leagues. Every delivery, dismissal, and milestone is captured.',
      detail: '5,000+ matches processed'
    },
    {
      num: '02',
      icon: '⚡',
      title: 'Player Elo Tracking',
      desc: 'Each player is assigned a dynamic Elo rating (like chess). Ratings update after every match based on win/loss outcomes, creating a real-time measure of player form and quality.',
      detail: 'Tracks 3,000+ players'
    },
    {
      num: '03',
      icon: '🔥',
      title: 'Recent Form Engine',
      desc: 'Team momentum is computed from the last 10 matches. The model weighs recent results exponentially — a team on a 5-match winning streak gets far more credit than one from 3 months ago.',
      detail: 'Rolling 10-match window'
    },
    {
      num: '04',
      icon: '🏟️',
      title: 'Venue Chase Bias',
      desc: "Every venue has a historical chase/defend win ratio. Wankhede is different from Chepauk. The model knows if a ground favors batting second (dew, flat pitches) or defending (spin-friendly, slow tracks).",
      detail: '40+ venues profiled'
    },
    {
      num: '05',
      icon: '🧮',
      title: 'XGBoost Prediction',
      desc: 'All features feed into a gradient-boosted decision tree (XGBoost). The model was trained on 12 engineered features with time-decayed sample weights — recent seasons matter more.',
      detail: '~70% prediction accuracy'
    },
    {
      num: '06',
      icon: '🧠',
      title: 'Symmetry Correction',
      desc: 'To eliminate team-ordering bias, we predict both Team1-vs-Team2 AND Team2-vs-Team1, then average the results. This ensures a truly fair probability regardless of input order.',
      detail: 'Dual-pass calibration'
    }
  ];

  return (
    <div className="mb-8 space-y-8">
      {/* Hero Intro */}
      <div className="text-center space-y-4 py-8">
        <div className="inline-flex items-center gap-2 bg-surface-lighter/50 px-4 py-2 rounded-full border border-surface-light/50 text-xs text-text-muted">
          <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse"></span>
          AI-Powered Cricket Analytics
        </div>
        <h2 className="text-3xl font-bold text-text">
          How the Oracle <span style={{ background: 'linear-gradient(90deg, var(--color-primary), var(--color-accent))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Predicts Matches</span>
        </h2>
        <p className="text-sm text-text-muted max-w-2xl mx-auto leading-relaxed">
          Our 0-Ball Prediction Engine analyzes pre-match signals — player quality, team momentum, venue characteristics, and toss dynamics — to forecast match outcomes before a single ball is bowled. Here's the science behind it.
        </p>
      </div>

      {/* Steps Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {steps.map((step, i) => (
          <div key={i} className="step-card bg-surface rounded-2xl border border-surface-light/30 p-6 space-y-4 shadow-lg shadow-bg/30">
            <div className="flex items-center justify-between">
              <span className="step-number text-3xl font-black">{step.num}</span>
              <span className="text-2xl">{step.icon}</span>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-bold text-text">{step.title}</h3>
              <p className="text-xs text-text-muted leading-relaxed">{step.desc}</p>
            </div>
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase text-primary bg-primary/10 px-3 py-1.5 rounded-lg border border-primary/20 w-fit">
              {step.detail}
            </div>
          </div>
        ))}
      </div>

      {/* Model Accuracy Badge */}
      <div className="flex justify-center">
        <div className="bg-surface rounded-2xl border border-surface-light/30 px-8 py-5 flex items-center gap-6 shadow-xl shadow-bg/50">
          <div className="text-center">
            <span className="text-3xl font-black" style={{ background: 'linear-gradient(90deg, var(--color-primary), var(--color-accent))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>70%</span>
            <p className="text-[10px] text-text-muted uppercase mt-1">Overall Accuracy</p>
          </div>
          <div className="w-px h-10 bg-surface-lighter"></div>
          <div className="text-center">
            <span className="text-3xl font-black" style={{ background: 'linear-gradient(90deg, var(--color-primary), var(--color-accent))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>12</span>
            <p className="text-[10px] text-text-muted uppercase mt-1">Input Features</p>
          </div>
        </div>
      </div>
    </div>
  );
}


/* ─────────────────────────────────────────────
   Main App
   ───────────────────────────────────────────── */
function App() {
  // ── State ──
  const [venues, setVenues] = useState([])
  const [teams, setTeams] = useState([])

  const [venue, setVenue] = useState('')
  const [team1, setTeam1] = useState('')
  const [team2, setTeam2] = useState('')
  const [tossWinner, setTossWinner] = useState('')
  const [tossDecision, setTossDecision] = useState('field')

  const [teamLastXi, setTeamLastXi] = useState({})
  const [teamSquads, setTeamSquads] = useState({})
  const [allPlayers, setAllPlayers] = useState([])
  
  const [team1Players, setTeam1Players] = useState(Array(11).fill(''))
  const [team2Players, setTeam2Players] = useState(Array(11).fill(''))
  const [showSquadBuilder, setShowSquadBuilder] = useState(false)

  // Auto-sync toss winner to avoid 400 Bad Request from stale data
  useEffect(() => {
    if (tossWinner !== team1 && tossWinner !== team2) {
      if (team1) setTossWinner(team1)
    }
  }, [team1, team2, tossWinner])

  // Auto-fill squad when team changes
  useEffect(() => {
    if (teamLastXi[team1]) {
      // pad with empty strings up to 12 slots (11 players + 1 impact player)
      const xi = [...teamLastXi[team1]]
      while(xi.length < 12) xi.push('')
      setTeam1Players(xi.slice(0, 12))
    }
  }, [team1, teamLastXi])

  useEffect(() => {
    if (teamLastXi[team2]) {
      const xi = [...teamLastXi[team2]]
      while(xi.length < 12) xi.push('')
      setTeam2Players(xi.slice(0, 12))
    }
  }, [team2, teamLastXi])

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  // ── Load teams & venues on mount ──
  useEffect(() => {
    fetch(`${API_BASE}/meta`).then(r => r.json()).then(d => {
      setTeams(d.teams)
      setVenues(d.venues)
      setTeamLastXi(d.team_last_xi || {})
      setTeamSquads(d.team_squads || {})
      setAllPlayers(d.all_players || [])
      if (d.teams.length >= 2) {
        setTeam1(d.teams[0])
        setTeam2(d.teams[1])
        setTossWinner(d.teams[0])
        if (d.team_last_xi) {
            const xi1 = [...(d.team_last_xi[d.teams[0]] || [])]
            while(xi1.length < 12) xi1.push('')
            setTeam1Players(xi1.slice(0, 12))
            
            const xi2 = [...(d.team_last_xi[d.teams[1]] || [])]
            while(xi2.length < 12) xi2.push('')
            setTeam2Players(xi2.slice(0, 12))
        }
      }
      if (d.venues.length > 0) setVenue(d.venues[0])
    }).catch(() => {})
  }, [])



  // ── Predict ──
  const handlePredict = async () => {
    if (!venue || !team1 || !team2) return
    if (team1 === team2) {
      alert("Please select two different teams.")
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          venue,
          team1,
          team2,
          toss_winner: tossWinner,
          toss_decision: tossDecision,
          team1_players: team1Players.filter(Boolean).length === 11 ? team1Players : null,
          team2_players: team2Players.filter(Boolean).length === 11 ? team2Players : null,
        }),
      })
      const data = await res.json()
      if (res.ok) {
        setPrediction(data)
        // Chat is now user-initiated only — no auto-fire
      } else {
        const errorMsg = typeof data.detail === 'string' 
            ? data.detail 
            : JSON.stringify(data.detail) || 'Prediction failed'
        alert(errorMsg)
      }
    } catch (e) {
      alert('Cannot reach the backend. Is the server running?')
    }
    setLoading(false)
  }



  // ── Select helpers ──
  const SelectField = ({ label, value, onChange, options, id }) => (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-xs font-medium text-text-muted uppercase tracking-wider">{label}</label>
      <select
        id={id}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-surface-lighter border border-surface-lighter text-text text-sm rounded-lg px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all appearance-none cursor-pointer"
      >
        {options.map(o => (
          <option key={typeof o === 'string' ? o : o.value} value={typeof o === 'string' ? o : o.value}>
            {typeof o === 'string' ? o : o.label}
          </option>
        ))}
      </select>
    </div>
  )

  return (
    <div className="min-h-screen bg-bg">
      {/* ── Header ── */}
      <header className="border-b border-surface-light/50 bg-surface/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
              <span className="text-white text-lg">🏏</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-text">Pre-Match Oracle</h1>
              <p className="text-xs text-text-muted">0-Ball Synergy & Chase Bias AI Model</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs font-medium bg-surface-lighter px-3 py-1.5 rounded-full border border-surface-light">
            <span className="inline-block w-2 h-2 rounded-full bg-success animate-pulse shadow-[0_0_8px_rgba(166,227,161,0.6)]"></span>
            <span className="text-text">70% Accuracy Live</span>
          </div>
        </div>
      </header>

      {/* ── Main Grid ── */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* ── Left: Input Controls ── */}
          <div className="lg:col-span-4 space-y-5">
            <div className="bg-surface rounded-3xl border border-surface-light/30 p-6 space-y-6 shadow-xl shadow-bg/50">
              <div className="flex items-center gap-2">
                 <div className="w-8 h-8 rounded-full bg-surface-lighter flex items-center justify-center border border-surface-light/50">
                    <span className="text-sm">🏟️</span>
                 </div>
                 <h2 className="text-sm font-bold text-text uppercase tracking-wider">
                   Match Conditions
                 </h2>
              </div>

              <SelectField
                label="Venue" id="venue-select"
                value={venue} onChange={setVenue}
                options={venues}
              />

              <SelectField
                label="Team 1" id="team1-select"
                value={team1} onChange={setTeam1}
                options={teams}
              />

              <SelectField
                label="Team 2" id="team2-select"
                value={team2} onChange={setTeam2}
                options={teams}
              />

              <div className="grid grid-cols-2 gap-4">
                <SelectField
                  label="Toss Winner" id="toss-winner-select"
                  value={tossWinner} onChange={setTossWinner}
                  options={[team1, team2].filter(Boolean)}
                />
                <SelectField
                  label="Toss Decision" id="toss-decision-select"
                  value={tossDecision} onChange={setTossDecision}
                  options={[{ value: 'bat', label: 'Bat First' }, { value: 'field', label: 'Field First' }]}
                />
              </div>
            </div>

            {/* Squad Builder Toggle */}
            <button
              onClick={() => setShowSquadBuilder(!showSquadBuilder)}
              className="w-full bg-surface-lighter hover:bg-surface-light border border-surface-light text-text-muted text-sm font-medium py-3 rounded-2xl transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <span className="text-lg">👥</span>
              {showSquadBuilder ? 'Hide Squad Builder' : 'Edit Playing XIs'}
            </button>

            {/* Squad Builder UI */}
            {showSquadBuilder && (
              <div className="bg-surface rounded-3xl border border-surface-light/30 p-6 space-y-6 shadow-xl shadow-bg/50">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-bold text-text uppercase tracking-wider">
                    Custom Squad Builder
                  </h2>
                  <span className="text-[10px] text-text-muted bg-surface-lighter px-2 py-1 rounded-full border border-surface-light/50">
                    Auto-filled with latest match XI
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-8">
                  {/* Team 1 Squad Selection */}
                  <div className="space-y-4">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-xs font-bold text-primary uppercase tracking-tight truncate" title={team1}>{team1 || 'Team 1'}</h3>
                      <div className="flex justify-between items-center">
                         <span className="text-[10px] text-text-muted font-mono">{team1Players.filter(p => p !== '').length} / 12 selected</span>
                         <button 
                            onClick={() => setTeam1Players(Array(12).fill(''))}
                            className="text-[9px] text-accent hover:underline uppercase font-bold"
                          >
                            Clear
                          </button>
                      </div>
                    </div>
                    
                    <div className="flex flex-col gap-2">
                        <input 
                            type="text"
                            placeholder="Search or add any player..."
                            className="w-full bg-surface-lighter border border-surface-light text-text text-[11px] rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-primary"
                            onChange={(e) => {
                                const val = e.target.value.toLowerCase();
                                if (val.length > 2) {
                                    const found = allPlayers.find(p => p.toLowerCase() === val);
                                    if (found && !team1Players.includes(found)) {
                                        const firstEmpty = team1Players.indexOf('');
                                        if (firstEmpty !== -1) {
                                            const newPlayers = [...team1Players];
                                            newPlayers[firstEmpty] = found;
                                            setTeam1Players(newPlayers);
                                            e.target.value = '';
                                        }
                                    }
                                }
                            }}
                        />
                        <div className="grid grid-cols-1 gap-1.5 max-h-[350px] overflow-y-auto pr-2 custom-scrollbar border-t border-surface-light/20 pt-2">
                          {(teamSquads[team1] || []).map(player => {
                            const isSelected = team1Players.includes(player);
                            const slotIdx = team1Players.indexOf(player);
                            return (
                              <button
                                key={player}
                                onClick={() => {
                                  if (isSelected) {
                                    const newPlayers = [...team1Players];
                                    newPlayers[slotIdx] = '';
                                    setTeam1Players(newPlayers);
                                  } else {
                                    const firstEmpty = team1Players.indexOf('');
                                    if (firstEmpty !== -1) {
                                      const newPlayers = [...team1Players];
                                      newPlayers[firstEmpty] = player;
                                      setTeam1Players(newPlayers);
                                    }
                                  }
                                }}
                                className={`flex items-center justify-between px-3 py-2 rounded-xl text-[11px] transition-all border ${
                                  isSelected 
                                    ? 'bg-primary/20 border-primary text-primary shadow-lg shadow-primary/10' 
                                    : 'bg-surface-lighter border-surface-light text-text-muted hover:border-text-muted'
                                }`}
                              >
                                <span className="truncate max-w-[120px]">{player}</span>
                                {isSelected && (
                                  <span className="bg-primary text-bg font-bold px-1.5 py-0.5 rounded text-[9px]">
                                    {slotIdx === 0 ? 'C' : slotIdx === 11 ? 'IMP' : slotIdx + 1}
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>
                    </div>
                  </div>

                  {/* Team 2 Squad Selection */}
                  <div className="space-y-4">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-xs font-bold text-accent uppercase tracking-tight truncate" title={team2}>{team2 || 'Team 2'}</h3>
                      <div className="flex justify-between items-center">
                         <span className="text-[10px] text-text-muted font-mono">{team2Players.filter(p => p !== '').length} / 12 selected</span>
                         <button 
                            onClick={() => setTeam2Players(Array(12).fill(''))}
                            className="text-[9px] text-accent hover:underline uppercase font-bold"
                          >
                            Clear
                          </button>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                        <input 
                            type="text"
                            placeholder="Search or add any player..."
                            className="w-full bg-surface-lighter border border-surface-light text-text text-[11px] rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-accent"
                            onChange={(e) => {
                                const val = e.target.value.toLowerCase();
                                if (val.length > 2) {
                                    const found = allPlayers.find(p => p.toLowerCase() === val);
                                    if (found && !team2Players.includes(found)) {
                                        const firstEmpty = team2Players.indexOf('');
                                        if (firstEmpty !== -1) {
                                            const newPlayers = [...team2Players];
                                            newPlayers[firstEmpty] = found;
                                            setTeam2Players(newPlayers);
                                            e.target.value = '';
                                        }
                                    }
                                }
                            }}
                        />
                        <div className="grid grid-cols-1 gap-1.5 max-h-[350px] overflow-y-auto pr-2 custom-scrollbar border-t border-surface-light/20 pt-2">
                          {(teamSquads[team2] || []).map(player => {
                            const isSelected = team2Players.includes(player);
                            const slotIdx = team2Players.indexOf(player);
                            return (
                              <button
                                key={player}
                                onClick={() => {
                                  if (isSelected) {
                                    const newPlayers = [...team2Players];
                                    newPlayers[slotIdx] = '';
                                    setTeam2Players(newPlayers);
                                  } else {
                                    const firstEmpty = team2Players.indexOf('');
                                    if (firstEmpty !== -1) {
                                      const newPlayers = [...team2Players];
                                      newPlayers[firstEmpty] = player;
                                      setTeam2Players(newPlayers);
                                    }
                                  }
                                }}
                                className={`flex items-center justify-between px-3 py-2 rounded-xl text-[11px] transition-all border ${
                                  isSelected 
                                    ? 'bg-accent/20 border-accent text-accent shadow-lg shadow-accent/10' 
                                    : 'bg-surface-lighter border-surface-light text-text-muted hover:border-text-muted'
                                }`}
                              >
                                <span className="truncate max-w-[120px]">{player}</span>
                                {isSelected && (
                                  <span className="bg-accent text-bg font-bold px-1.5 py-0.5 rounded text-[9px]">
                                    {slotIdx === 0 ? 'C' : slotIdx === 11 ? 'IMP' : slotIdx + 1}
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Predict Button */}
            <button
              id="predict-button"
              onClick={handlePredict}
              disabled={loading || !venue || !team1 || !team2}
              className="group relative w-full py-4 rounded-2xl font-bold text-white transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer overflow-hidden"
              style={{
                background: loading
                  ? 'var(--color-surface-lighter)'
                  : 'linear-gradient(135deg, var(--color-primary), var(--color-accent))',
              }}
            >
              {!loading && (
                  <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out"></div>
              )}
              <div className="relative flex items-center justify-center gap-2">
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Analyzing Model...
                  </>
                ) : 'Run Oracle Prediction'}
              </div>
            </button>
            
            <p className="text-center text-xs text-text-muted mt-2">
              Model uses player Elo tracking, recent form, and venue chase bias.
            </p>
          </div>

          {/* ── Results: Gauge + Features ── */}
          <div className="lg:col-span-8 space-y-6">
            {!prediction && <HowItWorks />}

            {prediction && (
              <>
                <div className="bg-surface rounded-3xl border border-surface-light/30 p-8 flex flex-col items-center shadow-xl shadow-bg/50">
                  <WinGauge
                    probability={prediction?.win_probability}
                    team1={prediction?.team1 || team1}
                    team2={prediction?.team2 || team2}
                  />
                </div>

                {prediction?.top_features && (
                  <div className="bg-surface rounded-3xl border border-surface-light/30 p-8 shadow-xl shadow-bg/50">
                    <FeatureBars features={prediction.top_features} />
                  </div>
                )}

                <PredictionReasoning prediction={prediction} />
                
                <MatchProjection prediction={prediction} />
              </>
            )}
          </div>

        </div>
      </main>
    </div>
  )
}

export default App
