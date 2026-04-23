import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Workflow, ChevronRight, Cpu, Network, Brain, Activity, Users, TrendingUp, Zap, Shield } from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface PNode {
  id: string; label: string; sub: string
  x: number; y: number; color: string; r: number; icon: string
  group: 'data' | 'process' | 'graph' | 'model' | 'rl' | 'agent' | 'ensemble' | 'fl' | 'output'
}

// ─── Pipeline nodes ───────────────────────────────────────────────────────────

const NODES: Record<string, PNode> = {
  nifty50:  { id:'nifty50',  label:'NIFTY 50 Data',  sub:'Yahoo Finance · 2015-2025',   x:165, y:80,   color:'#3B82F6', r:42, group:'data',     icon:'📊' },
  news:     { id:'news',     label:'Google News',     sub:'Live RSS · 10s timeout',      x:655, y:80,   color:'#8B5CF6', r:42, group:'data',     icon:'📰' },
  features: { id:'features', label:'Feature Eng.',    sub:'21 Indicators',               x:288, y:232,  color:'#06B6D4', r:38, group:'process',  icon:'⚙️' },
  sentiment:{ id:'sentiment',label:'FinBERT',          sub:'Score ∈ [-1,+1]',             x:632, y:232,  color:'#EC4899', r:38, group:'process',  icon:'💬' },
  graph:    { id:'graph',    label:'Graph Builder',   sub:'Sector+Supply+Corr.',         x:450, y:375,  color:'#F59E0B', r:42, group:'graph',    icon:'🕸️' },
  tgat:     { id:'tgat',     label:'T-GAT Model',    sub:'(n_stocks, 64) embeds',       x:450, y:500,  color:'#14B8A6', r:42, group:'model',    icon:'🧠' },
  rlenv:    { id:'rlenv',    label:'RL Environment', sub:'Gymnasium · Sharpe',          x:450, y:625,  color:'#6366F1', r:42, group:'rl',       icon:'🎮' },
  ppo:      { id:'ppo',      label:'PPO',            sub:'On-policy',                   x:148, y:742,  color:'#C15F3C', r:30, group:'agent',    icon:'🤖' },
  sac:      { id:'sac',      label:'SAC',            sub:'Entropy',                     x:276, y:742,  color:'#6366F1', r:30, group:'agent',    icon:'🤖' },
  td3:      { id:'td3',      label:'TD3',            sub:'Delayed',                     x:450, y:742,  color:'#0D9488', r:30, group:'agent',    icon:'🤖' },
  a2c:      { id:'a2c',      label:'A2C',            sub:'Fast',                        x:624, y:742,  color:'#F59E0B', r:30, group:'agent',    icon:'🤖' },
  ddpg:     { id:'ddpg',     label:'DDPG',           sub:'Determ.',                     x:752, y:742,  color:'#EC4899', r:30, group:'agent',    icon:'🤖' },
  ensemble: { id:'ensemble', label:'★ Ensemble',     sub:'Best-of-5 Average',           x:450, y:862,  color:'#16A34A', r:46, group:'ensemble', icon:'⭐' },
  fl:       { id:'fl',       label:'Federated',      sub:'4 Sectors · DP-SGD',          x:222, y:978,  color:'#2563EB', r:40, group:'fl',       icon:'🔒' },
  portfolio:{ id:'portfolio',label:'Portfolio',       sub:'Sharpe · Weights · Risk',     x:678, y:978,  color:'#16A34A', r:46, group:'output',   icon:'💼' },
}

const EDGES: [string, string][] = [
  ['nifty50', 'features'], ['news', 'sentiment'],
  ['features', 'graph'],   ['sentiment', 'graph'],
  ['graph', 'tgat'],       ['tgat', 'rlenv'],
  ['rlenv', 'ppo'],        ['rlenv', 'sac'],   ['rlenv', 'td3'],
  ['rlenv', 'a2c'],        ['rlenv', 'ddpg'],
  ['ppo', 'ensemble'],     ['sac', 'ensemble'],  ['td3', 'ensemble'],
  ['a2c', 'ensemble'],     ['ddpg', 'ensemble'],
  ['ensemble', 'fl'],      ['ensemble', 'portfolio'],
  ['fl', 'portfolio'],
]

const PLAY_SEQUENCE = [
  ['nifty50', 'news'],
  ['features', 'sentiment'],
  ['graph'],
  ['tgat'],
  ['rlenv'],
  ['ppo', 'sac', 'td3', 'a2c', 'ddpg'],
  ['ensemble'],
  ['fl'],
  ['portfolio'],
]

const STAGE_LABELS = [
  '① Data Ingestion — NIFTY 50 prices + financial news fetched',
  '② Data Processing — 21 technical indicators + FinBERT sentiment computed',
  '③ Graph Construction — 3-edge-type stock network built',
  '④ T-GAT Encoding — Temporal graph attention → 64-dim stock embeddings',
  '⑤ RL Environment — Portfolio Gym env initialized with all features',
  '⑥ Agent Training — 5 algorithms (PPO/SAC/TD3/A2C/DDPG) train in parallel',
  '⑦ Ensemble — 5 model predictions averaged → consensus portfolio weights',
  '⑧ Federated Learning — 4 sector clients + DP-SGD privacy aggregation',
  '⑨ Portfolio Output — Final NIFTY 50 allocation with full risk metrics',
]

// ─── Node details panel content ───────────────────────────────────────────────

const DETAILS: Record<string, { title: string; tech: string; input: string; output: string; points: string[] }> = {
  nifty50: {
    title: 'NIFTY 50 Price Data',
    tech: 'yfinance + curl_cffi',
    input: 'Ticker symbols, date range (2015-01-01 → 2025-01-01)',
    output: 'OHLCV DataFrame + all_close_prices.csv (45+ stocks)',
    points: [
      '5 retries with exponential backoff (3^n sec, max 60s)',
      'SSL bypass via curl_cffi for college/proxy networks',
      '1 sec rate-limit delay between downloads',
      '7 quality checks: NaN, duplicates, negative prices, extreme returns',
      'Forward-fill (ffill) for NSE holidays',
    ],
  },
  news: {
    title: 'Google News RSS Feed',
    tech: 'feedparser · 10s timeout',
    input: 'Company name query per ticker (e.g. "Reliance Industries stock NSE")',
    output: 'Up to 100 headlines: {title, published, ticker, sector}',
    points: [
      '20 key NIFTY 50 stocks + 2 market-wide queries',
      'Auto-refreshes every 3 minutes in Sentiment dashboard',
      'SQLite cache at data/sentiment.db avoids FinBERT recomputation',
      'Deduplication by headline string before scoring',
      '+N new badge tracks fresh headlines between refreshes',
    ],
  },
  features: {
    title: '21 Technical Indicators',
    tech: 'Pure numpy/pandas (no ta-lib dependency)',
    input: 'Raw OHLCV DataFrame per stock',
    output: 'numpy (n_stocks, n_timesteps, 21) float32',
    points: [
      'Trend: RSI, MACD, MACD Signal, MACD Histogram',
      'Bollinger: BB Upper, BB Mid, BB Lower (20-day, 2σ)',
      'Moving Avg: SMA20, SMA50, EMA12, EMA26',
      'Volatility: ATR, Vol20d, Vol60d',
      'Others: Stoch%K/D, Volume SMA, Volume Ratio, Return 1/5/20d',
      'Rolling z-score (252-day window), clipped to [-5, +5] — no look-ahead bias',
    ],
  },
  sentiment: {
    title: 'FinBERT Sentiment',
    tech: 'ProsusAI/finbert · FP16 on GPU · threading.Lock()',
    input: 'Financial headline strings (batch of up to 750)',
    output: 'score = P(positive) - P(negative) ∈ [-1, +1] per stock per day',
    points: [
      'Thread-safe singleton cache — prevents race condition on concurrent API calls',
      'Local model path (data/finbert_local/) for SSL/proxy environments',
      'Decay factor 0.95 — days without news retain 95% of previous score',
      'Batch size 16 for GPU inference efficiency (~200MB VRAM in FP16)',
      'Result feeds into RL observation space as sentiment dimension',
    ],
  },
  graph: {
    title: 'Multi-Relational Stock Graph',
    tech: 'PyTorch Geometric · Data(x, edge_index, edge_type)',
    input: 'Ticker registry + rolling close prices',
    output: 'list[PyG Data] — one graph per trading day',
    points: [
      'Edge type 0 — Sector: ~160 directed edges (same-sector pairs)',
      'Edge type 1 — Supply Chain: ~54 directed edges (e.g. TATASTEEL→MARUTI)',
      'Edge type 2 — Correlation: dynamic, |corr| > 0.6 over 60-day window',
      'All edges bidirectional, self-loops excluded',
      'Duplicate edges deduplicated (sector + supply chain overlap)',
      'NaN correlations → 0.0 (stocks with no price variance in window)',
    ],
  },
  tgat: {
    title: 'Temporal Graph Attention Network',
    tech: 'GATConv (3 types, 4 heads) + GRU · 56K params · 0.22MB',
    input: 'List of PyG Data objects (one per trading day)',
    output: '(n_stocks, 64) temporal stock embeddings',
    points: [
      'RelationalGATLayer × 2: separate GATConv per edge type',
      'Weighted aggregation of 3 edge-type attention outputs',
      'Residual connection + LayerNorm after each GAT layer',
      'GRU temporal encoder: stack T spatial embeddings → last hidden state',
      'FP16 on GPU: 0.11 MB VRAM. CPU fallback for testing.',
      'Output appended to RL observation space as 64×n_stocks dims',
    ],
  },
  rlenv: {
    title: 'Portfolio RL Environment',
    tech: 'Custom Gymnasium env · Sharpe-based reward',
    input: 'Feature tensor + price tensor + optional T-GAT embeddings + sentiment',
    output: 'obs (obs_dim,), reward (float), terminated (bool)',
    points: [
      'Observation = features(21n) + weights(n) + cash + norm_value + embeddings(64n) + sentiment(n)',
      'Action: continuous (n_stocks,) → softmax → portfolio weights',
      'Reward = rolling-20d Sharpe - drawdown penalty - turnover penalty',
      'Max position 20% per stock (clip + renormalize)',
      'Stop loss: -5% daily stock return → forced position exit',
      'Circuit breaker: -15% portfolio drawdown → episode termination',
      'Transaction cost: 0.1% + slippage 0.05% per turnover unit',
      'Shape validation added: raises ValueError on n_stocks/n_timesteps mismatch',
    ],
  },
  ppo: {
    title: 'PPO — Proximal Policy Optimization',
    tech: 'Stable-Baselines3 · MlpPolicy [128,64] · ~46K params',
    input: 'PortfolioEnv observations',
    output: 'Portfolio weight vector (n_stocks,)',
    points: [
      'On-policy: collects n_steps=2048 steps, then updates 10 epochs',
      'Clipped objective (clip_range=0.2) prevents catastrophic updates',
      'Most stable algorithm — primary baseline in backtesting',
      'PortfolioMetricsCallback logs Sharpe/return/drawdown during training',
      'Momentum-weighted portfolio strategy in simulation',
    ],
  },
  sac: {
    title: 'SAC — Soft Actor-Critic',
    tech: 'SB3 · replay buffer 100K · ent_coef=auto',
    input: 'PortfolioEnv observations',
    output: 'Stochastic portfolio weights with entropy bonus',
    points: [
      'Off-policy: learns from replay buffer (batch=256, buffer=100K)',
      'Auto-tuned entropy coefficient (ent_coef=auto)',
      'More sample-efficient than PPO — learns more from each transition',
      'Entropy bonus encourages exploration across different allocations',
      'PPO weights + Gaussian noise strategy in API simulation',
    ],
  },
  td3: {
    title: 'TD3 — Twin Delayed DDPG',
    tech: 'SB3 · policy_delay=2 · target_noise=0.2',
    input: 'PortfolioEnv observations',
    output: 'Deterministic portfolio weights (momentum-biased)',
    points: [
      'Twin critics reduce Q-value overestimation bias',
      'Delayed actor update (policy_delay=2) reduces policy variance',
      'Target policy smoothing (noise=0.2) prevents sharp policy peaks',
      'Aggressive momentum strategy — higher exponent than PPO',
      'Most stable off-policy algorithm in portfolio context',
    ],
  },
  a2c: {
    title: 'A2C — Advantage Actor-Critic',
    tech: 'SB3 · n_steps=5 · ent_coef=0.01',
    input: 'PortfolioEnv observations',
    output: 'Conservative portfolio weights (contrarian strategy)',
    points: [
      'On-policy with shorter rollouts (n_steps=5 vs PPO 2048)',
      'Advantage = Q(s,a) - V(s) reduces variance of policy gradient',
      'Faster per-update but noisier than PPO',
      'Contrarian/conservative — slight mean-reverting tendency',
      'Low entropy coefficient (0.01) → more deterministic than SAC',
    ],
  },
  ddpg: {
    title: 'DDPG — Deep Deterministic PG',
    tech: 'SB3 · deterministic · no entropy · buffer 100K',
    input: 'PortfolioEnv observations',
    output: 'Blend of momentum + equal-weight allocation',
    points: [
      'Deterministic policy gradient — no stochasticity in action selection',
      'No entropy regularization unlike SAC',
      'Off-policy with replay buffer (buffer=100K, batch=256)',
      'Sensitive to hyperparameters — needs careful tuning',
      'Blends PPO weights 50% + equal-weight 50% in simulation',
    ],
  },
  ensemble: {
    title: '★ Ensemble Meta-Policy',
    tech: 'Custom EnsembleAgent · equal-weighted average',
    input: 'Raw action vectors from all 5 trained models',
    output: 'Consensus portfolio weights (most robust allocation)',
    points: [
      'Averages raw action predictions: (PPO + SAC + TD3 + A2C + DDPG) / 5',
      'PortfolioEnv._action_to_weights() applies softmax afterwards',
      'Reduces single-model bias and overfitting to specific market regimes',
      'Optional weighted ensemble: models ranked by recent validation Sharpe',
      'Outperforms individual algorithms on out-of-sample NIFTY 50 data',
      'Shown as ★ in RL Agent dashboard — "Recommended" badge',
    ],
  },
  fl: {
    title: 'Federated Learning + DP-SGD',
    tech: 'Custom FedProx · ε=8.0 · δ=1e-5 · 4 sector clients',
    input: 'Per-client sector stock data (non-IID split)',
    output: 'Privacy-preserved globally aggregated model weights',
    points: [
      'Client 0: Banking + Finance (~10 stocks, RBI regulated)',
      'Client 1: IT + Telecom (~6 stocks, tech co-movement)',
      'Client 2: Pharma + FMCG (~8 stocks, defensive sectors)',
      'Client 3: Energy + Auto + Metals + Infra (~23 stocks, cyclical)',
      'FedProx: adds proximal term (mu=0.01) to prevent client drift',
      'DP-SGD: clip gradients (max_norm=1.0) + Gaussian noise',
      'Cumulative privacy budget tracked per round (ε composition)',
    ],
  },
  portfolio: {
    title: 'Final Portfolio Output',
    tech: 'Validated NIFTY 50 allocation · full risk metrics',
    input: 'Ensemble weights post-FL aggregation',
    output: 'Portfolio page: weights, Sharpe, Sortino, MaxDD, performance chart',
    points: [
      'Constraints enforced: max 20% per stock (position limit)',
      'Sharpe Ratio: (annualized return - 7% risk-free) / volatility',
      'Sortino: only penalizes downside deviation (not total volatility)',
      'Max Drawdown: worst peak-to-trough decline in cumulative returns',
      'Stress-tested: 4 crash scenarios (Normal, 2008, COVID, Flash Crash)',
      'Compared to NIFTY 50 equal-weight benchmark throughout',
    ],
  },
}

// ─── Helper: compute cubic bezier SVG path between two nodes ─────────────────

function edgePath(a: PNode, b: PNode): string {
  const x1 = a.x, y1 = a.y + a.r
  const x2 = b.x, y2 = b.y - b.r
  const dy = y2 - y1
  return `M ${x1} ${y1} C ${x1} ${y1 + dy * 0.45}, ${x2} ${y2 - dy * 0.45}, ${x2} ${y2}`
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function WorkflowViz() {
  const [activeNodes, setActiveNodes] = useState<Set<string>>(new Set())
  const [selected, setSelected] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [step, setStep] = useState(-1)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)


  // ── Auto-play logic ────────────────────────────────────────────────────────

  const startInterval = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = setInterval(() => {
      setStep(prev => {
        const next = prev + 1
        if (next >= PLAY_SEQUENCE.length) {
          setPlaying(false)
          return prev
        }
        const group = PLAY_SEQUENCE[next]
        setActiveNodes(cur => new Set([...cur, ...group]))
        setSelected(group[group.length - 1])
        return next
      })
    }, 1800)
  }, [])

  useEffect(() => {
    if (playing) {
      startInterval()
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, startInterval])

  const handlePlay = () => {
    if (step >= PLAY_SEQUENCE.length - 1) {
      // done — reset then replay
      setActiveNodes(new Set())
      setStep(-1)
      setSelected(null)
      setTimeout(() => setPlaying(true), 100)
    } else {
      setPlaying(p => !p)
    }
  }

  const handleReset = () => {
    setPlaying(false)
    if (intervalRef.current) clearInterval(intervalRef.current)
    setActiveNodes(new Set())
    setSelected(null)
    setStep(-1)
  }

  const handleNodeClick = (id: string) => {
    setSelected(id)
    setActiveNodes(prev => new Set([...prev, id]))
  }

  const isEdgeActive = (f: string, t: string) => activeNodes.has(f) && activeNodes.has(t)

  const detail = selected ? DETAILS[selected] : null
  const node = selected ? NODES[selected] : null
  const stageLabel = step >= 0 && step < STAGE_LABELS.length ? STAGE_LABELS[step] : null
  const isDone = step >= PLAY_SEQUENCE.length - 1 && step >= 0

  return (
    <div className="space-y-4">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
            <Workflow size={18} className="text-white" />
          </div>
          <div>
            <h1 className="font-display font-bold text-xl text-secondary">Pipeline Visualization</h1>
            <p className="text-xs text-text-muted mt-0.5">End-to-end FINQUANT-NEXUS v4 workflow · animated component flow</p>
          </div>
        </div>
        {/* Controls */}
        <div className="flex items-center gap-2">
          <motion.button onClick={handleReset}
            whileHover={{ scale: 1.04, y: -1 }} whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 320, damping: 22 }}
            className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-text-secondary bg-bg-card border border-border rounded-xl hover:bg-bg-card/80 transition-colors">
            <motion.span animate={playing ? { rotate: -360 } : { rotate: 0 }}
              transition={playing ? { repeat: Infinity, duration: 2, ease: 'linear' } : {}}>
              <RotateCcw size={14} />
            </motion.span>
            Reset
          </motion.button>
          <motion.button onClick={handlePlay}
            whileHover={{ scale: 1.04, y: -1 }} whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 320, damping: 22 }}
            className={`flex items-center gap-1.5 px-4 py-2 text-sm font-semibold rounded-xl transition-all shadow-sm ${
              isDone
                ? 'bg-primary text-white hover:bg-primary/90'
                : playing
                  ? 'bg-amber-500 text-white hover:bg-amber-400'
                  : 'bg-primary text-white hover:bg-primary/90'
            }`}>
            {playing ? <><Pause size={14} /> Pause</> : isDone ? <><Play size={14} /> Replay</> : <><Play size={14} /> Play</>}
          </motion.button>
        </div>
      </div>

      {/* ── Stage label banner ── */}
      <div className="h-9 flex items-center">
        <AnimatePresence mode="wait">
          {stageLabel && (
            <motion.div key={step}
              initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
              className="flex items-center gap-2 px-4 py-2 bg-primary-subtle border border-primary/20 rounded-xl text-sm font-medium text-primary">
              <ChevronRight size={14} />
              {stageLabel}
            </motion.div>
          )}
          {!stageLabel && !playing && step < 0 && (
            <motion.p key="hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="text-sm text-text-muted px-1">
              Click ▶ Play to animate the AI pipeline · or click any node to explore
            </motion.p>
          )}
          {isDone && (
            <motion.div key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="flex items-center gap-2 px-4 py-2 bg-[#F0FDF4] border border-[#16A34A]/30 rounded-xl text-sm font-semibold text-[#16A34A]">
              ✓ Pipeline complete — Portfolio output ready
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Two-column: SVG left | Info right ── */}
      <div className="grid grid-cols-[1fr_360px] gap-4 items-start">

        {/* ── LEFT: SVG Pipeline ── */}
        <div className="bg-white border border-border rounded-2xl overflow-hidden">
          <div className="p-3">
        <svg
          viewBox="0 0 900 1080"
          className="w-full"
          style={{ display: 'block' }}
        >
            <defs>
              {/* Glow filter for active nodes */}
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              {/* Subtle drop shadow for inactive nodes */}
              <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.12" />
              </filter>
              {/* Clip path for section labels */}
              <style>{`
                @keyframes flowDash {
                  from { stroke-dashoffset: 24; }
                  to   { stroke-dashoffset: 0; }
                }
                .edge-flow {
                  stroke-dasharray: 8 5;
                  animation: flowDash 1.2s linear infinite;
                }
                @keyframes pulse-ring {
                  0%   { r: 0; opacity: 0.6; }
                  100% { r: 18; opacity: 0; }
                }
              `}</style>
            </defs>

            {/* ── Section background bands ── */}
            {[
              { y:22,  h:140, label:'Data Sources',    color:'#DBEAFE' },
              { y:176, h:125, label:'Processing',       color:'#CCFBF1' },
              { y:314, h:255, label:'Graph + T-GAT',   color:'#FEF3C7' },
              { y:580, h:125, label:'RL Environment',  color:'#E0E7FF' },
              { y:700, h:118, label:'RL Agents (5)',    color:'#F3F4F6' },
              { y:812, h:118, label:'Ensemble',         color:'#DCFCE7' },
              { y:925, h:140, label:'FL + Portfolio',  color:'#DBEAFE' },
            ].map(b => (
              <g key={b.label}>
                <rect x="12" y={b.y} width="876" height={b.h} rx="10" fill={b.color} opacity="0.45" />
                <text x="26" y={b.y + 20} fontSize="11" fill="#9CA3AF" fontWeight="700" fontFamily="system-ui" letterSpacing="0.06em">
                  {b.label.toUpperCase()}
                </text>
              </g>
            ))}

            {/* ── Edges ── */}
            {EDGES.map(([fId, tId]) => {
              const f = NODES[fId], t = NODES[tId]
              const active = isEdgeActive(fId, tId)
              const path = edgePath(f, t)
              const edgeId = `ep-${fId}-${tId}`
              return (
                <g key={edgeId}>
                  {/* Base path */}
                  <path id={edgeId} d={path} fill="none"
                    stroke={active ? f.color : '#E5E7EB'}
                    strokeWidth={active ? 2.5 : 1.5}
                    opacity={active ? 0.85 : 0.5}
                    className={active ? 'edge-flow' : ''}
                  />
                  {/* Data packet moving along edge (only when active) */}
                  {active && (
                    <circle r="4" fill="white" stroke={f.color} strokeWidth="1.5" opacity="0.95">
                      <animateMotion dur="1.6s" repeatCount="indefinite" calcMode="linear">
                        <mpath href={`#${edgeId}`} />
                      </animateMotion>
                    </circle>
                  )}
                </g>
              )
            })}

            {/* ── Nodes ── */}
            {Object.values(NODES).map(n => {
              const isActive = activeNodes.has(n.id)
              const isSel = selected === n.id
              const isAgent = n.group === 'agent'
              return (
                <g key={n.id} style={{ cursor: 'pointer' }}
                  onClick={() => handleNodeClick(n.id)}>
                  {/* Pulse ring when first activated */}
                  {isActive && (
                    <circle cx={n.x} cy={n.y} r={n.r + 4}
                      fill="none" stroke={n.color} strokeWidth="2" opacity="0"
                      style={{
                        animation: 'none',
                        transformOrigin: `${n.x}px ${n.y}px`,
                      }}
                    />
                  )}
                  {/* Glow backdrop when active */}
                  {isActive && (
                    <circle cx={n.x} cy={n.y} r={n.r + 8}
                      fill={n.color} opacity="0.12"
                      filter="url(#glow)"
                    />
                  )}
                  {/* Selection ring */}
                  {isSel && (
                    <circle cx={n.x} cy={n.y} r={n.r + 5}
                      fill="none" stroke={n.color} strokeWidth="2.5" opacity="0.7"
                      strokeDasharray="4 3"
                    />
                  )}
                  {/* Main circle */}
                  <circle
                    cx={n.x} cy={n.y} r={n.r}
                    fill={isActive ? n.color : '#F9FAFB'}
                    stroke={isActive ? n.color : '#D1D5DB'}
                    strokeWidth={isActive ? 0 : 1.5}
                    filter={isActive ? 'url(#shadow)' : 'none'}
                    style={{ transition: 'fill 0.4s ease, stroke 0.4s ease' }}
                  />
                  {/* Icon emoji */}
                  <text
                    x={n.x} y={n.y + (isAgent ? 5 : 6)}
                    textAnchor="middle" dominantBaseline="middle"
                    fontSize={isAgent ? 17 : 22}
                    style={{ userSelect: 'none', pointerEvents: 'none' }}
                  >
                    {n.icon}
                  </text>
                  {/* Node label */}
                  <text
                    x={n.x} y={n.y + n.r + 14}
                    textAnchor="middle"
                    fontSize={isAgent ? 10 : 11.5}
                    fontWeight="700"
                    fontFamily="system-ui"
                    fill={isActive ? n.color : '#374151'}
                    style={{ transition: 'fill 0.4s ease' }}
                  >
                    {n.label}
                  </text>
                  {/* Sub-label */}
                  {!isAgent && (
                    <text
                      x={n.x} y={n.y + n.r + 28}
                      textAnchor="middle"
                      fontSize="9.5"
                      fontFamily="system-ui"
                      fill={isActive ? n.color : '#9CA3AF'}
                      opacity="0.85"
                    >
                      {n.sub}
                    </text>
                  )}
                </g>
              )
            })}

            {/* ── Progress stepper dots (right edge) ── */}
            {PLAY_SEQUENCE.map((_, i) => (
              <circle key={i}
                cx={876} cy={80 + i * 112}
                r="5"
                fill={step >= i ? '#16A34A' : '#E5E7EB'}
                style={{ transition: 'fill 0.3s ease' }}
              />
            ))}
            <line x1="876" y1="80" x2="876" y2={80 + 8 * 112}
              stroke="#E5E7EB" strokeWidth="1.5" strokeDasharray="3 3" />
            {step >= 0 && (
              <line x1="876" y1="80" x2="876" y2={80 + step * 112}
                stroke="#16A34A" strokeWidth="1.5" />
            )}
          </svg>
          </div>
        </div>

        {/* ── RIGHT: Info Panel ── */}
        <div className="space-y-3 sticky top-4">

          {/* Node Detail Card */}
          <AnimatePresence mode="wait">
            {detail && node ? (
              <motion.div key={selected}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 12 }}
                transition={{ duration: 0.22 }}
                className="bg-white border border-border rounded-2xl overflow-hidden"
              >
                {/* Color header */}
                <div className="px-4 py-4" style={{ backgroundColor: node.color + '18', borderBottom: `2px solid ${node.color}30` }}>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-9 h-9 rounded-xl flex items-center justify-center text-lg"
                      style={{ backgroundColor: node.color }}>
                      {node.icon}
                    </div>
                    <div>
                      <h2 className="font-display font-bold text-sm text-secondary leading-tight">{detail.title}</h2>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-md mt-1 inline-block"
                        style={{ backgroundColor: node.color + '22', color: node.color }}>
                        {detail.tech}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 space-y-3">
                  {/* Input / Output */}
                  <div className="space-y-2">
                    <div className="rounded-lg bg-bg-card p-2.5 border border-border-light">
                      <p className="text-[9px] font-bold uppercase tracking-widest text-text-muted mb-1">Input</p>
                      <p className="text-xs text-text-secondary leading-relaxed">{detail.input}</p>
                    </div>
                    <div className="rounded-lg p-2.5 border"
                      style={{ backgroundColor: node.color + '0d', borderColor: node.color + '30' }}>
                      <p className="text-[9px] font-bold uppercase tracking-widest mb-1" style={{ color: node.color }}>Output</p>
                      <p className="text-xs leading-relaxed font-medium" style={{ color: node.color }}>{detail.output}</p>
                    </div>
                  </div>

                  {/* Key points */}
                  <div>
                    <p className="text-[9px] font-bold uppercase tracking-widest text-text-muted mb-2">Key Details</p>
                    <ul className="space-y-1.5">
                      {detail.points.map((pt, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-text-secondary leading-relaxed">
                          <span className="mt-0.5 shrink-0 w-1.5 h-1.5 rounded-full"
                            style={{ backgroundColor: node.color }} />
                          {pt}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div key="empty"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="bg-white border border-border rounded-2xl p-6 text-center"
              >
                <div className="w-14 h-14 bg-bg-card rounded-2xl flex items-center justify-center mx-auto mb-3">
                  <Workflow size={24} className="text-text-muted" />
                </div>
                <p className="font-medium text-sm text-secondary mb-1">No component selected</p>
                <p className="text-xs text-text-muted leading-relaxed">
                  Press Play to animate the full pipeline, or click any node in the diagram to explore its details.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Pipeline Stages ── */}
          <div className="bg-white border border-border rounded-2xl p-4">
            <p className="text-xs font-bold uppercase tracking-widest text-text-muted mb-3">Pipeline Stages</p>
            <div className="space-y-1">
              {PLAY_SEQUENCE.map((group, i) => (
                <motion.button key={i}
                  whileHover={{ x: 2 }} whileTap={{ scale: 0.98 }}
                  transition={{ type: 'spring', stiffness: 320, damping: 24 }}
                  onClick={() => {
                    const newActive = new Set<string>()
                    for (let j = 0; j <= i; j++) PLAY_SEQUENCE[j].forEach(n => newActive.add(n))
                    setActiveNodes(newActive)
                    setStep(i)
                    setSelected(group[group.length - 1])
                    setPlaying(false)
                  }}
                  className={`relative w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-left text-xs transition-colors ${
                    step === i ? 'text-primary font-semibold'
                    : step > i ? 'text-[#16A34A] font-medium'
                    : 'text-text-secondary hover:bg-bg-card'
                  }`}>
                  {step === i && (
                    <motion.span layoutId="stage-active-bg"
                      className="absolute inset-0 bg-primary-subtle rounded-lg"
                      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span className={`relative z-10 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                    step > i ? 'bg-[#16A34A] text-white' :
                    step === i ? 'bg-primary text-white' :
                    'bg-border text-text-muted'
                  }`}>{step > i ? '✓' : i + 1}</span>
                  <span className="relative z-10 truncate">{STAGE_LABELS[i].slice(4)}</span>
                </motion.button>
              ))}
            </div>
          </div>

          {/* ── Stats grid ── */}
          <div className="bg-white border border-border rounded-2xl p-4">
            <p className="text-xs font-bold uppercase tracking-widest text-text-muted mb-3">System Stats</p>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: 'Components',   value: '15',           icon: <Cpu size={12} /> },
                { label: 'Connections',  value: '19',           icon: <Network size={12} /> },
                { label: 'RL Algos',     value: '5 + Ensemble', icon: <Brain size={12} /> },
                { label: 'FL Clients',   value: '4 Sectors',    icon: <Users size={12} /> },
                { label: 'Features',     value: '21 + 64 + 1',  icon: <Activity size={12} /> },
                { label: 'Tests',        value: '246 / 246',    icon: <Zap size={12} /> },
                { label: 'NIFTY Stocks', value: '45+',          icon: <TrendingUp size={12} /> },
                { label: 'Privacy',      value: 'ε=8 · δ=1e-5', icon: <Shield size={12} /> },
              ].map(s => (
                <div key={s.label} className="rounded-lg bg-bg-card p-2 border border-border-light">
                  <div className="flex items-center gap-1 text-text-muted mb-0.5">
                    {s.icon}
                    <span className="text-[9px] font-semibold uppercase tracking-wider">{s.label}</span>
                  </div>
                  <p className="text-xs font-bold text-secondary">{s.value}</p>
                </div>
              ))}
            </div>
          </div>

        </div>{/* end right col */}
      </div>{/* end grid */}
    </div>
  )
}
