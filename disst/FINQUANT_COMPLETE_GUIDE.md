# FINQUANT-NEXUS v4 — Component-wise Interview & Viva Guide
### Sirf wahi jo dissertation aur UI me hai — actual values dissertation se verified

---

## SYSTEM KA BIG PICTURE

```
44 NIFTY-50 Stocks (2015–2025)
        ↓
[1] Data Pipeline → 21 Technical Indicators per stock per day
        ↓                    ↓
[2] FinBERT Sentiment    [3] Graph (79+24+147 edges) → T-GAT → 32-dim embeddings
        ↓                    ↓
        └────────────────────┘
                 ↓
    [4] RL Environment — Observation = 2420 values
         PPO | SAC | TD3 | A2C | DDPG → Ensemble
                 ↓
    [5] Stress Testing (8 scenarios, 1000 paths each)
    [6] Federated Learning (4 clients, 50 rounds, ε=8.0)
                 ↓
    [7] FastAPI Backend (50+ endpoints)
                 ↓
    [8] React Dashboard (8 tabs)
```

**Ek line summary:** FINQUANT-NEXUS ek portfolio intelligence platform hai jo NIFTY-50 ke 44 stocks ke liye RL agents, T-GAT graph embeddings, FinBERT sentiment, aur federated learning ko ek pipeline me jodhta hai.

---
---

# COMPONENT 1 — DATA PIPELINE

## 1.1 Dataset — Exact Numbers

| Fact | Value |
|------|-------|
| Index | NIFTY 50 |
| Stocks used | **44** (6 exclude kiye — incomplete Yahoo Finance history) |
| Date range | Jan 2015 – Dec 2025 |
| Trading sessions per stock | ~2,761 |
| Feature matrix shape | **(2761, 44, 21)** |

**Train / Val / Test Split:**
```
Jan 2015 ──────── Dec 2021 ──── Dec 2023 ──── Dec 2025
|    TRAINING (~1757)  |  VAL (~502)  |  TEST (~502)  |
```
- Training: 2015–2021 (7 years — demonetisation, NBFC crisis, COVID crash, recovery — multiple regimes)
- Validation: 2022–2023 (hyperparameter tuning)
- Test: 2024–2025 (genuinely unseen, final evaluation)

**Kyun time-series split zaroori hai?**
Random split me future data training me aata — look-ahead bias. Real deployment me future pata nahi hota.

---

## 1.2 Preprocessing

- **Missing values:** Forward-fill (holiday gaps). Interpolation nahi — artificial price movements create karta.
- **Outliers:** 5-sigma threshold. Genuine corporate events (splits/bonus) rakhte hain. Data errors fix/forward-fill.
- **Adjusted Close:** yfinance `auto_adjust=True` — splits aur dividends automatically correct.
- **Volume zeros:** Forward-fill (trading halt days).

---

## 1.3 Feature Engineering — 21 Technical Indicators

**Normalization: Rolling Z-score (252-day window)**
```
Z(t) = (X(t) − rolling_mean(X, 252)) / rolling_std(X, 252)
```
Rolling normalization — global stats use karna future leak karega.

| No. | Indicator | Category | What it captures |
|-----|-----------|----------|-----------------|
| 1 | SMA 20 | Trend | Short-term price average |
| 2 | EMA 20 | Trend | Recent-weighted short trend |
| 3 | EMA 50 | Trend | Intermediate trend |
| 4 | MACD | Trend | EMA(12) − EMA(26) crossover |
| 5 | MACD Signal | Trend | EMA(9) of MACD |
| 6 | MACD Histogram | Trend | MACD − Signal divergence |
| 7 | RSI (14) | Momentum | Overbought/oversold |
| 8 | Stochastic %K | Momentum | Close position in range |
| 9 | Stochastic %D | Momentum | Smoothed stochastic |
| 10 | Williams %R | Momentum | Inverted stochastic |
| 11 | ROC (10) | Momentum | Rate of price change |
| 12 | CCI (20) | Momentum | Price deviation from average |
| 13 | Bollinger Upper | Volatility | SMA(20) + 2σ resistance |
| 14 | Bollinger Lower | Volatility | SMA(20) − 2σ support |
| 15 | Bollinger Bandwidth | Volatility | (Upper−Lower)/Mid — vol magnitude |
| 16 | ATR (14) | Volatility | Average True Range — intraday vol |
| 17 | Daily Return | Price | (Close_t / Close_{t-1}) − 1 |
| 18 | Momentum 10 | Price | Close_t / Close_{t-10} |
| 19 | OBV | Volume | Cumulative volume direction |
| 20 | Volume Ratio | Volume | Volume / SMA(Vol, 20) — anomaly |
| 21 | VWAP deviation | Volume | (Close − VWAP) / VWAP |

**21 kyun? Data-driven tha ya heuristic?** Industry standard indicators cover kiye — trend, momentum, volatility, volume. Systematic selection, manual tuning nahi.

**Multicollinearity:** SMA aur EMA correlated hain — acknowledged limitation. RL agent implicitly feature importance seekhta hai.

---
---

# COMPONENT 2 — SENTIMENT ANALYSIS (FinBERT)

## 2.1 Why FinBERT, General BERT Nahi?

General BERT "loss" ko "loss of life" samajhta hai. "Bear" = animal. Financial language domain-specific hai.

FinBERT (ProsusAI/finbert) — BERT fine-tuned on financial news + earnings call transcripts.

**Output per headline:**
```
P(positive) = 0.82, P(neutral) = 0.12, P(negative) = 0.06
Sentiment Score = P(positive) − P(negative) = +0.76
Range: [−1, +1]
```

Model locally stored: `data/finbert_local/` — inference me internet nahi chahiye.

---

## 2.2 News Sources

| Source | Coverage |
|--------|---------|
| Google News RSS | 80% of total signal (English, wide range) |
| Yahoo Finance News API | Stock-specific articles |
| Indian RSS (Moneycontrol, ET) | Domestic market stories |

**Fetching:** ThreadPoolExecutor — 44 stocks concurrently. 5-second timeout per source. Deduplication by headline text.

**Aggregation:** Multiple headlines → weighted average by recency → 1 score per stock per day.

**Daily score matrix shape: (2761, 44)**

---

## 2.3 Market Mood Classification

| Score | Classification |
|-------|--------------|
| > 0.15 | **Bullish** |
| 0 to 0.15 | **Neutral** |
| < 0 | **Bearish** |

**Observed on dissertation evaluation day:**
- Overall avg score: **0.2942** → Bullish
- Finance sector: **+0.7227** (strongest)
- Others: **+0.7062**
- FMCG: **+0.2095**
- Auto: **−0.3259** (only negative)
- Top mover: MARUTI SUZUKI **−2.21%** (aligned with Auto negative sentiment)

---

## 2.4 Caching

SQLite TTL = **3 minutes.** Inference 15–30 seconds on CPU → bar bar call wasteful.

---

## 2.5 Sentiment → RL me kaise jaata hai

44 sentiment values observation vector ke end me concatenate hote hain as one of 4 components. RL agent training me seekhta hai ki sentiment ko kitna weight dena hai.

---
---

# COMPONENT 3 — STOCK GRAPH (3 Edge Types)

## 3.1 Graph Structure

```
G = (V, E)
V = 44 stock nodes
E = 250 edges (3 types)
```

**Node features at each timestep: 22 values = 21 technical indicators + 1 sentiment score**

---

## 3.2 Edge Type 0 — Sector Edges (STATIC)

- Same NSE sector → undirected edge
- Example: HDFCBANK ↔ ICICIBANK ↔ AXISBANK ↔ KOTAKBANK (Banking sector)
- **Count: 79 edges**
- Never change — sector classification permanent hai

**Rationale:** RBI repo rate change → poora banking sector ek saath affect. Sector edges ye group behavior encode karte hain.

---

## 3.3 Edge Type 1 — Supply Chain Edges (STATIC)

- Known business dependency relationships — manually defined
- Example: TATASTEEL → MARUTI (steel supplier → car maker), ONGC → RELIANCE
- Bidirectional (information flows both ways)
- **Count: 24 edges**

**Rationale:** Upstream cost change downstream margins affect karta hai. Correlation capture nahi kar paati ye economic dependency.

---

## 3.4 Edge Type 2 — Correlation Edges (DYNAMIC)

- 60-day rolling Pearson correlation > **0.6 threshold** → undirected edge
- Daily update: correlation spike → more edges; low correlation → edges disappear
- **Count: ~147 on evaluation day** (range: 90–220 across test window)

**Threshold 0.6 kyun?**
- Too low (0.3) → overcrowded graph, noise dominates
- Too high (0.8) → sparse, T-GAT ko seekhne ko kuch nahi
- 0.6 = balance (acknowledged: tuned on validation set)

---

## 3.5 Graph Statistics (Dissertation Table 4.4)

| Metric | Value |
|--------|-------|
| Total nodes | 44 |
| Total edges | 250 |
| Sector edges | 79 |
| Supply chain edges | 24 |
| Correlation edges | 147 |
| Graph density | **0.264** |
| Average node degree | **11.4** |
| Highest degree stock | **HDFCBANK (degree 23, 22 neighbours)** |
| Strongest pair | **BAJFINANCE–BAJAJFINSV (corr = 0.89)** |

**HDFCBANK highest degree kyun?** Banking sector peers (sector edges) + supply chain links + return correlations — sab ek saath.

**BAJFINANCE–BAJAJFINSV 0.89 kyun important?** Ek hi Bajaj group — RL agent ko pata hai dono independent positions nahi hain, overweight prevent hoga.

---
---

# COMPONENT 4 — T-GAT (Temporal Graph Attention Network)

## 4.1 Intuition: GAT kya karta hai

**Simple GCN:** `h_i_new = mean(all neighbours)`  — sab equally weighted

**GAT:**
```
h'_i = Σ_j  α_ij × W × h_j
```
- `α_ij` = attention weight (learned) — kuch neighbours zyada important hain
- HDFCBANK aur SBIN banking peers hain, attention zyada milega vs unrelated pair

**T-GAT = GAT + Temporal (GRU):** sirf ek snapshot nahi, time pe evolving relationships capture karta hai.

---

## 4.2 Architecture — EXACT VALUES

**RelationalGATLayer:**
- **3 separate weight matrices** — ek har edge type ke liye (sector, supply chain, correlation)
- **8 attention heads per edge type** — different aspects of relationships
- Per edge type r, node i ke liye:
```
α_ij^r = softmax_j [ LeakyReLU( a_rᵀ [W_r·h_i ‖ W_r·h_j] ) ]
h'_i^r = σ( Σ_j α_ij^r × W_r × h_j )
```
- Final: `h'_i = h'_i^sector ‖ h'_i^supply ‖ h'_i^correlation`

**GRU Temporal Encoder:**
- 2-layer GRU, hidden size **128**
- Input: sequence of daily GAT outputs per stock
- Output projected to: **32-dimensional embedding per stock per day**

---

## 4.3 Training

- Pre-trained on 2015–2021 training data
- Task: binary cross-entropy on next-day return direction (up/down)
- Optimizer: Adam, lr = 0.001
- **Weights frozen during RL training** — embeddings static during RL phase

---

## 4.4 Output → RL me kaise jaata hai

32-dim embeddings × 44 stocks = **1408 values** → observation vector ka part

**Simple correlation matrix se T-GAT better kyun?**
- GAT 3 types of relationships alag handle karta hai
- GRU temporal evolution capture karta hai
- Attention weights interpretable hain — kaunsa neighbour zyada influence karta hai
- Multi-hop information: A→B→C chain capture possible

---
---

# COMPONENT 5 — RL ENVIRONMENT

## 5.1 Observation Space — 2420 values (EXACT)

```
Component              | Calculation        | Values
-----------------------|-------------------|--------
Technical indicators   | 44 stocks × 21    | 924
T-GAT embeddings       | 44 stocks × 32    | 1,408
Sentiment scores       | 44 stocks × 1     | 44
Portfolio weights      | 44 stocks         | 44
-----------------------|-------------------|--------
TOTAL                  |                   | 2,420
```

All values normalized to [−1, +1] range before agent.

---

## 5.2 Action Space

```python
action_space = Box(low=-1.0, high=1.0, shape=(44,))  # 44 stocks
```

**Action → Portfolio Weights (Softmax):**
```
w_i = exp(a_i) / Σ_j exp(a_j)
```
All weights positive, sum = 1.0. Short selling not permitted.

---

## 5.3 Constraints — DISSERTATION VALUES

| Constraint | Value | Meaning |
|-----------|-------|---------|
| Max position per stock | **12%** | Excess redistributed proportionally |
| Stop-loss | **−3% single-day** | Weight reduced 50% next step |
| Max drawdown circuit breaker | **−12%** | Episode terminate early |
| Transaction cost | **0.1%** per trade | STT + brokerage |
| Slippage | **0.05%** | Market impact |
| Episode length | **252 days** (1 year) |  |
| Start point | **Random** from training period | Diverse market conditions |

---

## 5.4 Reward Function — DISSERTATION FORMULA

```
R(t) = Sharpe_rolling_30  −  λ₁ × drawdown_penalty(t)  −  λ₂ × turnover_penalty(t)

λ₁ = 0.5  (drawdown weight — tuned on validation)
λ₂ = 0.3  (turnover weight — tuned on validation)
```

- **Rolling 30-day Sharpe** — risk-adjusted return, not raw return
- **Drawdown penalty:** activated when portfolio > 8% below peak
- **Turnover penalty:** proportional to Σ|w_new − w_old| — excessive rebalancing discourage

**Kyun Sharpe reward, raw return nahi?**
Raw return → agent concentrated positions lega, high volatility. Sharpe → risk per unit of return penalty → safer portfolio.

---
---

# COMPONENT 6 — RL AGENTS (5 + Ensemble)

## 6.1 On-Policy vs Off-Policy

```
On-Policy  → Current policy ke data se hi seekhna (PPO, A2C)
Off-Policy → Replay buffer ka purana data bhi use karna (SAC, TD3, DDPG)
```

Off-policy zyada data-efficient, lekin tune karna mushkil.

---

## 6.2 Training Protocol (Same for all)

| Parameter | Value |
|-----------|-------|
| Training steps | **500,000** |
| Discount factor (γ) | **0.99** |
| Optimizer | Adam |
| Learning rate | **3×10⁻⁴** |
| Batch size | 64 (on-policy: PPO, A2C) / 256 (off-policy: SAC, TD3, DDPG) |
| Library | Stable-Baselines3 v2.8.0 |

---

## 6.3 PPO — Primary Agent

**Proximal Policy Optimization**
- On-policy, stable training
- Key feature: **clip_epsilon = 0.2**

```
Clipping: ratio = π_new(a|s) / π_old(a|s) → clamp to [0.8, 1.2]
```
Policy ek step me zyada change nahi ho sakti → stability.

- **entropy coefficient = 0.01** → mild exploration bonus

---

## 6.4 SAC — Entropy Regularized

**Soft Actor-Critic**
- Off-policy, replay buffer size = **1 million**
- Extra term: Maximum Entropy RL

```
Objective = Expected Return + α × Entropy(π)
```

- `ent_coef = 'auto'` → alpha automatically tune hota hai
- **Financial markets me useful kyun:** Markets noisy → pure exploitation dangerous. Entropy → explore diverse strategies, overconfident allocation avoid.
- **Result:** Lowest volatility (12.58%), lowest return (14.31%) — conservative par consistent

---

## 6.5 TD3 — Twin Delayed DDPG

**3 tricks DDPG ki Q-overestimation fix karne ke liye:**
1. **Twin critics:** 2 Q-networks → minimum lao → overestimation reduce
2. **Delayed policy update:** `policy_delay = 2` → critic 2 baar update ho, policy 1 baar
3. **Target policy smoothing:** Noise add to target → robust Q estimates

- Off-policy, replay buffer = 1 million

---

## 6.6 A2C — Advantage Actor-Critic

- On-policy, shorter rollouts (n_steps = 5 vs PPO's 2048)
- Faster per update, noisier than PPO
- `entropy coef = 0.01`

**Advantage function:**
```
A(s,a) = Q(s,a) − V(s)
```
"Is action ne expected se kitna better/worse kiya?"

---

## 6.7 DDPG — Deterministic Policy

- Off-policy, **deterministic** policy (no entropy term)
- One specific action for each observation (no sampling)
- Most aggressive — concentrated positions
- **Result:** Highest return (21.27%) but WORST drawdown (−21.37%)

---

## 6.8 Ensemble — The Most Important Agent

```
a_ensemble = (a_PPO + a_SAC + a_TD3 + a_A2C + a_DDPG) / 5
```
Simple equal average of all 5 actions → softmax → final weights.

**No additional training.** Just inference-time averaging.

**Kyun better hai individual se?**
- DDPG → trending market best, drawdown worst
- SAC → stable/sideways best, upside limited
- Ensemble → average out → variance reduce → **more robust across regimes**

---

## 6.9 ACTUAL RESULTS — Test Period 2024–2025

| Algorithm | Sharpe | Sortino | Ann. Return | Volatility | Max Drawdown |
|-----------|--------|---------|-------------|------------|--------------|
| PPO | 0.7829 | 1.0721 | +15.22% | 12.76% | −17.06% |
| SAC | 0.7288 | 1.0089 | +14.31% | 12.58% | −16.42% |
| TD3 | 0.7480 | 1.0212 | +14.86% | 12.98% | −16.14% |
| A2C | 0.7520 | 1.0447 | +14.52% | 12.42% | −16.29% |
| DDPG | 0.8909 | 1.1279 | +21.27% | 17.84% | −21.37% |
| **Ensemble** | **0.8316** | **1.1086** | **+16.75%** | **13.76%** | **−17.80%** |
| NIFTY 50 (benchmark) | — | — | **+0.65%** | — | — |

**Key insight:** DDPG Sharpe highest, lekin Ensemble preferred — kyunki Max Drawdown DDPG ka worst hai (−21.37%). Investor downside risk care karta hai.

---

## 6.10 Live Portfolio (Apr 2025 – Mar 2026)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **0.2996** |
| Sortino Ratio | **0.4371** |
| Total Return | **+8.27%** (Rs. 10L → Rs. 10,82,745) |
| Annual Volatility | 12.86% |
| Max Drawdown | **−12.17%** |
| NIFTY 50 same period | +0.65% |
| Fixed Deposit (7%) | +4.96% |
| **Outperformance vs NIFTY** | **+7.62 pp** |
| **Outperformance vs FD** | **+3.31 pp** |

**Sharpe 0.2996 kyun lower than backtest 0.8316?**
Different windows — backtest 2024–2025 full year, live window Apr–Mar shorter. Also live window NIFTY essentially flat — different market regime.

---
---

# COMPONENT 7 — STRESS TESTING

## 7.1 What It Does

- Backtesting tells what happened in test period
- Stress testing asks: **"What if 2008 happens again?"**

**Setup:**
- **1,000 Monte Carlo paths** per scenario
- **252 trading days** per path
- Ensemble RL weights **fixed** across all paths → isolate scenario effect from agent rebalancing
- GAN-calibrated return generator (3-layer LSTM, hidden 128, latent 64, trained 500 epochs) → preserves fat tails, volatility clustering (better than plain Gaussian)

---

## 7.2 Three Risk Metrics

**VaR 95%:**
```
5th percentile of return distribution
95% confidence se portfolio isse zyada nahi giregi
```

**CVaR 95% (Expected Shortfall):**
```
Average loss in worst 5% of paths
VaR se zyada conservative — tail risk capture karta hai
```

**Survival Rate:**
```
% paths where portfolio drawdown stays within −12% circuit breaker
```

---

## 7.3 All 8 Scenarios — ACTUAL NUMBERS

| Scenario | Mean Return | VaR 95% | CVaR 95% | Survival |
|----------|------------|---------|----------|---------|
| Normal | +15.74% | −15.89% | −21.32% | 34.4% |
| 2008 Financial Crisis | −25.62% | −49.31% | −53.65% | **1.2%** |
| COVID-19 Crash | −12.35% | −29.37% | −32.32% | 21.1% |
| Flash Crash | −9.27% | −19.19% | −21.87% | **76.4%** |
| Dot-Com 2000 | −22.62% | −47.86% | −52.78% | **0.9%** |
| India Bear 2015 | −13.24% | −38.00% | −43.28% | 4.0% |
| Rate Hike 2022 | −7.45% | −33.51% | −39.30% | 7.0% |
| Geo-Political Shock | −14.33% | −32.99% | −37.55% | 12.2% |

---

## 7.4 Scenarios Interpret Kaise Karein

**Flash Crash best survival (76.4%) kyun?**
Historical event brief tha. Stop-loss constraints ne per-period loss limit kiya, forced selling at bottom nahi hua.

**2008 (1.2%) aur Dot-Com (0.9%) survival near-zero kyun?**
12–18 month prolonged decline 40–60%. Koi bhi equity strategy survive nahi kar sakti. System ki failure nahi — honest characterization hai.

**Rate Hike 2022 interesting case:**
Mean return mild (−7.45%), lekin VaR-CVaR gap wide (33.51% vs 39.30%). Rate-driven environments me sector impact bahut vary karta hai — debt-sensitive sectors vs growth sectors alag behave karte hain.

---
---

# COMPONENT 8 — FEDERATED LEARNING

## 8.1 Motivation

Real institutional investors apna sector allocation data share nahi karte — proprietary + regulatory constraints.

FL → **sirf model weight updates share karo, raw data kabhi server pe nahi jaata.**

---

## 8.2 Client Setup — 4 Sector Clients

| Client | Stocks | Portfolio Weight |
|--------|--------|-----------------|
| Banking & Finance | 10 stocks (HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK, INDUSINDBK, BAJFINANCE, BAJAJFINSV, others) | ~23% |
| IT & Telecom | 6 stocks (TCS, INFOSYS, WIPRO, HCLTECH, TECHM, BHARTIARTL) | ~14% |
| Pharma & FMCG | 8 stocks (SUNPHARMA, DRREDDY, DIVISLAB, CIPLA, HINDUNILVR, ITC, NESTLEIND, BRITANNIA) | ~18% |
| Energy, Auto & Others | 20 stocks | ~45% |

---

## 8.3 FedAvg vs FedProx

**FedAvg (baseline):**
```
global_weights = Σ (client_i_size / total_size) × client_i_weights
```
Problem: Non-IID data → clients drift far from global model. Banking client → banking-biased. FedAvg average oscillates.

**FedProx (used here):**
```
Client loss = Local_loss + (μ/2) × ||w − w_global||²
              ← proximal term (client-side) →
```
- `μ = 0.01`
- Client ko global model ke paas rehne ki penalty → drift prevent
- **Convergence:** FedProx faster, smoother. FedAvg first 15–20 rounds oscillate karta hai.

**Key:** Proximal term **client-side** apply hota hai, server pe aggregation FedAvg jaisi hi hai.

---

## 8.4 DP-SGD — Line by Line

**Step 1: Gradient Clipping**
```
g_clipped = g × min(1, C / ||g||₂)     C = max_norm = 1.0
```
Koi single sample gradient zyada influence nahi de sakta.

**Step 2: Gaussian Noise Add karo**
```
g_noisy = g_clipped + N(0, σ² × I)     σ = noise_multiplier = 1.1
```

**Step 3: Privacy Guarantee**
```
ε = 8.0,  δ = 0.00001,  across 50 rounds
```

**ε = 8.0 ka matlab:**
Adversary model output dekh ke training data identify karne ki probability bounded hai. Practical range 1–10 me hai. ε=1 → zyada private, but model quality degrade.

**δ = 10⁻⁵ ka matlab:**
"With 99.999% probability, DP guarantee holds."

**ε tracker reaches exactly 8.0 at round 50** — designed hai aisa.

---

## 8.5 FL Results — ACTUAL NUMBERS

| Metric | Value |
|--------|-------|
| Rounds | 50 |
| Local epochs per round | 5 |
| Global Sharpe after 50 rounds | **0.729** |
| Privacy budget | ε = 8.0, δ = 0.00001 |

**Per-client Sharpe improvement (vs isolated training):**

| Client | Change |
|--------|--------|
| Banking & Finance | **+0.298** |
| IT & Telecom | **+0.339** (highest) |
| Pharma & FMCG | **+0.134** |
| Energy, Auto & Others | **−0.138** |

**Energy decline kyun?** Energy stocks commodity prices aur government policy se driven hain — global model ka average behavior energy ke distinct characteristics se align nahi hota. Personalized FL natural solution hoga (out of scope).

---
---

# COMPONENT 9 — FUTURE PREDICTION TAB

## 9.1 Black Bootstrap

- **1,000 forward paths** over 1-year horizon
- Block resampling (30-day segments) from historical returns calibrated by GAN
- **Better than Gaussian MC:** Fat tails, volatility clustering, momentum preserve hota hai — Indian equities normal distribution follow nahi karte

## 9.2 Results

| Metric | Value |
|--------|-------|
| Median Return | **+9.3%** |
| Best Case | +31.6% |
| Worst Case | −11.1% |
| Probability of Profit | **75.9%** |

**Per-algorithm forward return:**

| Algorithm | Expected Forward Return | P(profit) |
|-----------|------------------------|----------|
| PPO | **10.7%** (highest) | — |
| SAC | moderate | — |
| TD3 | **0.6%** (near-zero) | **83.7%** (highest) |
| DDPG | 4.0% (dropped from 21.27% backtest) | — |
| Ensemble | ~9.3% (median) | 75.9% |

**DDPG forward 4.0% vs backtest 21.27% kyun?**
GAN bootstrap spans full 2015–2025 distribution. 2024–2025 test window DDPG ke concentrated strategy ke liye specifically favourable tha — broader sample me ye consistently nahi hota.

**TD3 83.7% P(profit) but low return:** Conservative allocation → almost always profit, lekin barely.

---
---

# COMPONENT 10 — FASTAPI BACKEND

## 10.1 Why FastAPI

- Async handling → multiple requests simultaneously
- Pydantic → auto request/response validation
- Swagger UI auto-generate at `/docs`
- Performance: ASGI (near-NodeJS speed)

## 10.2 Key Endpoint → Tab Mapping

| Endpoint | Dashboard Tab |
|----------|--------------|
| `GET /api/portfolio-summary` | Portfolio tab |
| `GET /api/rl-summary` | RL Agent tab |
| `GET /api/stress-test` | Stress Testing tab |
| `GET /api/fl-summary` | Federated tab |
| `GET /api/news-sentiment` | Sentiment tab |
| `GET /api/gnn-summary` | Graph Visualization tab |
| `GET /api/future-prediction` | Future Prediction tab |
| 50+ total | — |

## 10.3 Caching

- **Price data:** In-memory after first load (thread-safe)
- **Sentiment:** SQLite TTL = **3 minutes**

---
---

# COMPONENT 11 — REACT DASHBOARD (8 Tabs)

## Tab 1: Portfolio

**Kya dikhata hai:**
- 5 metric cards: Sharpe (0.2996), Sortino (0.4371), Total Return (8.34%), Annual Volatility (12.86%), Max Drawdown (−12.17%)
- Holdings table: 44 stocks, weight, sector (no stock > 12%)
- Growth chart: Portfolio vs NIFTY-50 vs Fixed Deposit (Apr 2025 – Mar 2026)

**Color thresholds:**
```
Sharpe > 1.0 → Green | 0.5–1.0 → Yellow | < 0.5 → Red
```

---

## Tab 2: RL Agent

**Kya dikhata hai:**
- 6 buttons: PPO | SAC | TD3 | A2C | DDPG | ★ Ensemble
- Per-algorithm metrics table (Sharpe, Sortino, Return, Vol, MaxDD)
- Training reward curves (line chart)
- Cumulative returns (all agents on one chart)
- Sector allocation bar chart per agent

---

## Tab 3: Stress Testing

**Kya dikhata hai:**
- 8 scenario cards: Normal, 2008, COVID, Flash Crash, Dot-Com, India Bear 2015, Rate Hike 2022, Geo-Political
- Per scenario: Mean Return, VaR 95%, CVaR 95%, Survival Rate
- Monte Carlo fan chart: 1000 paths, median highlighted, 10th–90th percentile bands

---

## Tab 4: Federated Learning

**Kya dikhata hai:**
- 4 client cards (sector + stock count + weight %)
- FedProx vs FedAvg convergence curves (50 rounds)
- Privacy ε tracker (reaches 8.0 at round 50)
- Per-client Sharpe improvement (Banking +0.298, IT +0.339, Pharma +0.134, Energy −0.138)

---

## Tab 5: Sentiment

**Kya dikhata hai:**
- LIVE badge (auto-refresh every 3 min)
- Per-stock sentiment score bar
- Market mood indicator (Bullish/Neutral/Bearish)
- Recent headlines with sentiment label (+score)
- Sector-level scores
- +N new badge when fresh headlines arrive

---

## Tab 6: Graph Visualization

**Kya dikhata hai:**
- Force-directed network: 44 nodes + 250 edges
- Node size = portfolio weight
- Node color = sector
- 3 edge-type toggles: Sector | Supply Chain | Correlation
- Click node → detail panel (sector, weight, degree, neighbours)
- High-correlation periods → graph visibly densifies

---

## Tab 7: Pipeline (Workflow)

**Kya dikhata hai:**
- Animated 15-stage data flow diagram
- Stage statuses: Running / Idle

**15 stages:**
1. Data Ingestion → 2. Data Cleaning → 3. Feature Engineering → 4. Indicator Computation → 5. FinBERT Sentiment → 6. News Caching → 7. Graph Construction → 8. T-GAT Embedding → 9. Observation Assembly → 10. RL Environment → 11. Agent Inference → 12. Portfolio Weights → 13. Risk Constraint Check → 14. API Response → 15. Dashboard Rendering

---

## Tab 8: Future Prediction

**Kya dikhata hai:**
- Black Bootstrap: 1000 paths, 1-year forward
- Median +9.3%, P(profit) 75.9%
- Best/Worst case
- Per-algorithm forward expected return comparison

---
---

# METRICS — EXACT FORMULAS (Actual Code)

```python
# Sharpe Ratio (rf=5%, periods=248 India calendar days)
Sharpe = sqrt(248) × mean(excess_return) / std(excess_return)
excess_return = daily_return − (0.05 / 248)

# Sortino Ratio
Sortino = sqrt(248) × mean(excess_return) / std(negative_excess_only)

# Max Drawdown (returns negative value)
drawdown = (portfolio_value − running_peak) / running_peak
max_drawdown = min(all drawdown values)

# Calmar Ratio
Calmar = (Annualized_Return − rf) / |Max Drawdown|

# Annualized Return (geometric)
Ann_Return = (1+r₁)(1+r₂)...(1+rₙ) ^ (248/n) − 1

# Annualized Volatility
Ann_Vol = daily_std × sqrt(248)

# Portfolio Turnover
Turnover = mean(Σ|w_new − w_old|) per day
```

**rf = 5% (conservative — below 10-yr G-Sec yield 6.8–7.1%, aligned with SBI FD rate)**
**periods = 248 (India calendar, NOT 252)**

**Sharpe vs Sortino:**
- Sharpe: upside + downside volatility dono penalize
- Sortino: sirf downside → upside volatility ko reward nahi karta
- Isliye Sortino > Sharpe hamesha: 0.2996 vs 0.4371 (portfolio), 0.8316 vs 1.1086 (Ensemble)

---
---

# RAPID FIRE — ONE LINE ANSWERS

| Term | Answer |
|------|--------|
| Sharpe Ratio | (Return − rf) / Std × sqrt(248) — risk-adjusted return |
| Sortino Ratio | Sharpe but only downside std — upside vol penalize nahi |
| Calmar Ratio | Annual return / Max Drawdown |
| VaR 95% | 5th percentile loss — 95% scenarios isse zyada nahi girenge |
| CVaR 95% | Average loss in worst 5% — tail risk measure |
| Max Drawdown | Peak se trough tak max % fall |
| FedAvg | Client weights ka dataset-size-weighted average |
| FedProx | FedAvg + proximal term client-side → drift prevent |
| Client Drift | Non-IID data se client globally diverge karna |
| DP-SGD | Gradient clip + Gaussian noise → (ε,δ)-DP guarantee |
| ε = 8.0 | Moderate privacy budget — adversary inference bounded |
| δ = 10⁻⁵ | 99.999% probability DP guarantee holds |
| FinBERT | BERT fine-tuned on financial text for sentiment |
| T-GAT | Temporal GAT — 3-edge-type attention + GRU temporal encoder |
| Non-IID | Clients ke data distributions alag hain (here: sectors) |
| Look-ahead bias | Future data se train karna → inflated backtest |
| Survivorship bias | Only surviving stocks include karna — failed ignore |
| Policy delay (TD3) | Critic 2× update, policy 1× → stability |
| Entropy (SAC) | Policy randomness — maximize karna exploration encourage |
| Replay Buffer | Off-policy agents ka experience memory |
| Episode | 252 trading days (1 year) |
| GAN stress | 3-layer LSTM, hidden 128, latent 64 — realistic paths generate |
| Black Bootstrap | 30-day block resampling — fat tails preserve |
| Circuit breaker | Portfolio −12% → episode terminate |
| Stop-loss | Stock −3% single day → 50% weight cut next step |

---
---

# IMPORTANT TRAPS — INTERVIEW ME YE GALAT MAT BOLNA

| Wrong (old question bank) | CORRECT (dissertation verified) |
|--------------------------|----------------------------------|
| 50 stocks | **44 stocks** (6 excluded) |
| 64-dim T-GAT embeddings | **32-dim** (GRU projected) |
| 8 attention heads (total) | **8 heads per edge type** (3 types) |
| Observation 4352 | **2420** (44×21 + 44×32 + 44 + 44) |
| Sharpe 1.87 | **Ensemble Sharpe = 0.8316** (test) |
| Return 28.4% | **Ensemble return = 16.75%** (backtest) / **8.27%** (live) |
| Max position 20% | **12%** (dissertation Ch3) |
| Stop-loss −5% | **−3%** single-day |
| Circuit breaker −15% | **−12%** |
| 4 stress scenarios | **8 scenarios** |
| Monte Carlo 10,000 paths | **1,000 paths** per scenario |
| Correlation threshold 0.4 | **0.6** |
| Reward rolling 20 days | **Rolling 30 days** |
| λ₁=0.4, λ₂=0.02 | **λ₁=0.5, λ₂=0.3** |
| Drawdown penalty at any drawdown | **Activated at −8%** drawdown |
| FL clients by sector count | Sizes: 10, 6, 8, 20 stocks |
| DP noise multiplier not stated | **σ = 1.1, max_norm = 1.0** |
| Pipeline stages unknown | **15 stages** (named) |

---

*Values source: Dissertation chapters 3, 4, 5 — verified May 2026*
