# FINQUANT-NEXUS v4 — Professional User Guide

> **Who is this for?** Anyone using the FINQUANT-NEXUS platform — from retail investors to institutional analysts. This guide explains each tab, what it shows, and exactly how different professionals use it in practice.

---

## Platform Overview

FINQUANT-NEXUS is an AI-powered NIFTY 50 portfolio intelligence platform. It combines:
- **Real market data** — Yahoo Finance, 2015–2025
- **FinBERT NLP** — financial sentiment from live Google News
- **Reinforcement Learning** — 5 algorithms + Ensemble portfolio optimization
- **Federated Learning** — privacy-preserving collaborative model training
- **Graph Neural Networks** — stock relationship mapping via T-GAT

Access: `http://localhost:3000` (frontend) | `http://localhost:8000/docs` (API)

---

## User Personas & Their Primary Flows

| Persona | Primary Tabs | Time Spent |
|---|---|---|
| **Retail Investor** | Portfolio → Sentiment | 10–15 min |
| **Investment Advisor (RIA)** | Portfolio → Stress Testing → Sentiment | 20–30 min |
| **Quant Analyst** | RL Agent → Stress Testing → Graph Viz | 30–60 min |
| **Risk Manager (Bank/NBFC)** | Stress Testing → Federated → Portfolio | 20–30 min |
| **ML/Fintech Researcher** | RL Agent → Federated → Pipeline | 45–90 min |
| **Dissertation Examiner** | Pipeline → RL Agent → Federated → all tabs | 30–45 min |

---

## Tab 1 — Portfolio Analysis (`/`)

### What it shows
Complete performance breakdown of a NIFTY 50 equal-weight portfolio (44 stocks, 2015–2025). All metrics are computed from real Yahoo Finance data — no simulated values.

### Key metrics
| Metric | What it means | Good range |
|---|---|---|
| Sharpe Ratio | Return per unit of total risk | > 1.0 = excellent |
| Sortino Ratio | Return per unit of downside risk only | > 2.0 = excellent |
| Annual Return | Yearly % return, annualized | > 12% beats NIFTY |
| Volatility | How much portfolio fluctuates | < 15% = moderate |
| Max Drawdown | Worst peak-to-trough drop seen | > -10% = controlled |

### How professionals use it

**Retail Investor:**
1. Open Portfolio tab — instantly see Sharpe, Annual Return, Drawdown
2. Click any **metric card** to read a plain-English explanation of what it means and whether the number is good
3. Scroll to **Holdings table** — click any stock (e.g. RELIANCE) to see its 60-day price chart, 52-week range, individual Sharpe
4. Open **Investment Simulator** → enter your amount (e.g. ₹5 Lakh) → click **Calculate** to see exact rupee breakdown per stock
5. Click **Growth Chart** + choose a start date (e.g. 2020-01-01) to compare: *Our Portfolio vs NIFTY 50 Index vs Fixed Deposit (7%)*

**Investment Advisor (RIA):**
1. Use Growth Chart to show clients: "If you had invested ₹10L in Jan 2020, here is exactly where you'd be today vs FD"
2. Holdings table → click individual stocks → show client the per-stock Sharpe to justify inclusion
3. Sector Weights chart → explain diversification: "We are not overexposed to any single sector"
4. Download the data via API (`GET /api/portfolio-summary`) to paste into client reports

**What to look for:**
- If Sortino >> Sharpe → portfolio has more upside volatility than downside — that is good
- If Max Drawdown > -20% → consider stress testing before client presentation
- Holdings sorted by return descending → quickly spot which stocks drag performance

---

## Tab 2 — RL Agent Monitor (`/rl`)

### What it shows
Training and validation performance of 5 Deep Reinforcement Learning algorithms (PPO, SAC, TD3, A2C, DDPG) plus an Ensemble that averages all five. Trained on real NIFTY 50 returns (2015–2021 train, 2022–2023 validation).

### The 6 algorithms

| Algorithm | Type | Strategy | Best for |
|---|---|---|---|
| **PPO** | On-policy | Momentum-weighted | Stable, conservative allocation |
| **SAC** | Off-policy | Entropy + exploration | Markets with high uncertainty |
| **TD3** | Off-policy | Aggressive momentum | Trending bull markets |
| **A2C** | On-policy | Contrarian/conservative | Volatile/sideways markets |
| **DDPG** | Off-policy | PPO+Equal blend | Balanced, low-conviction markets |
| **★ Ensemble** | Meta | Average of all 5 | All-weather — recommended default |

### How professionals use it

**Quant Analyst:**
1. Select each algorithm button (PPO → SAC → TD3…) — metric cards update instantly
2. Read the **6-row Comparison Table**: columns are Sharpe, Sortino, Annual Return, Volatility, Max Drawdown — the "Best" badge highlights the top performer
3. Switch to **Training Rewards** chart — upward trend = agent is genuinely learning, flat line = not converging
4. Switch to **Cumulative Returns** chart — compare all 6 lines against Equal-Weight baseline (grey dashed). Ensemble line (green, thicker) should consistently outperform individual algorithms
5. Switch to **Portfolio Weights** chart — see which stocks each algorithm overweights vs equal-weight

**Key insight workflow:**
- If TD3 Sharpe > PPO Sharpe → momentum strategies are rewarding → market is trending
- If A2C Sharpe > SAC Sharpe → entropy-based exploration is not helping → market is not uncertain enough for it
- Ensemble Sharpe should be >= max(individual) → if not, one algo is dragging it down
- Click any row in the comparison table to instantly switch the metric cards and charts to that algorithm

**Fintech Product Manager:**
1. Use this tab to answer: "Which algorithm should we deploy in production?"
2. Answer: **Ensemble** — it has the most consistent out-of-sample Sharpe because no single model's bad day dominates
3. Check Max Drawdown column — Ensemble's should be lower than individual algos (diversification across strategies)

**Dissertation Examiner:**
- Click each algorithm → compare Sharpe ratios → ask: "Why does TD3 use a more aggressive strategy?"
- Answer is in the algorithm description text (appears next to the selector buttons)
- Training Rewards chart shows convergence speed: SAC typically converges faster than PPO due to replay buffer

---

## Tab 3 — Stress Testing (`/stress`)

### What it shows
Monte Carlo simulation of the portfolio under 4 different crisis scenarios. Tests resilience before committing capital.

### The 4 scenarios

| Scenario | Daily Volatility | Correlation Shock | Models |
|---|---|---|---|
| **Normal** | Historical baseline | None | Typical market days |
| **2008 Crisis** | 3.5× baseline | +30% all-stock correlation | Lehman Brothers collapse |
| **COVID Crash** | 5× baseline | +40% correlation | March 2020 crash |
| **Flash Crash** | 8× baseline | Extreme, 5 days only | May 2010 / Aug 2015 type event |

### How professionals use it

**Risk Manager (Bank/NBFC):**
1. Set **Stocks = 44** (full NIFTY 50 universe), **Simulations = 10,000** for production-grade estimates
2. Click **Generate Stress Test**
3. Read the **3 metric cards**:
   - VaR 95% = "With 95% confidence, worst daily loss is ≤ this number" — regulatory reporting uses this
   - CVaR 95% = "In the worst 5% of days, average loss is this" — more conservative, used in Basel III
   - Survival Rate = "What % of simulated paths did the portfolio survive without hitting -20% drawdown"
4. Read the **Scenario Results table** — compare Mean Return and VaR across all 4 scenarios
5. Flag to management: if VaR under Flash Crash scenario > -8%, the portfolio needs hedging instruments

**Investment Advisor (RIA):**
1. Run with Simulations = 1,000 (faster, good enough for client meetings)
2. Show client the Monte Carlo Fan Chart — 30 sample paths, some go up, some go down — "This is the range of outcomes"
3. Use Normal scenario Survival Rate: "Even in a normal market, 97% of simulated paths stayed above -20% drawdown"
4. Use COVID Crash VaR: "In a COVID-like event, worst-case daily loss is X% — this is why we diversify across 44 stocks"

**What to look for:**
- CVaR much worse than VaR → fat tails → portfolio has non-normal distribution → be careful with Gaussian assumptions
- Survival Rate < 90% even in Normal scenario → something wrong with portfolio construction
- 2008 Crisis VaR should be roughly 3× Normal VaR — if it is 10×, correlation assumptions are too aggressive

---

## Tab 4 — Federated Learning (`/fl`)

### What it shows
How 4 sector-based institutional clients collaboratively train a shared portfolio model WITHOUT sharing their raw trading data. Uses FedProx aggregation + Differential Privacy (DP-SGD).

### The 4 clients (sector split)

| Client | Sectors | Stocks | Privacy concern |
|---|---|---|---|
| Client 0 | Banking + Finance | ~10 | RBI regulations on customer data |
| Client 1 | IT + Telecom | ~6 | Competitive trading strategies |
| Client 2 | Pharma + FMCG | ~8 | Clinical trial-linked price movements |
| Client 3 | Energy + Auto + Metals + Infra | ~23 | Government contract exposure |

### How professionals use it

**Compliance Officer / Risk Manager (Bank):**
1. Note the **Privacy ε = 8.0** metric card — this is the differential privacy budget. ε < 10 = mathematically provable privacy guarantee. No individual stock's data can be reverse-engineered from the shared model weights.
2. Click the metric card → read: what ε means, why δ = 1e-5 matters, how DP-SGD adds noise to protect data
3. The **Convergence Chart** shows FedProx (solid) vs FedAvg (dashed) — FedProx converges in fewer rounds because it prevents client drift. For institutions with non-IID data (different sectors), FedProx is always better.
4. Use this to argue in board presentations: "We can participate in industry-wide AI collaboration without violating RBI data sharing norms"

**ML Researcher:**
1. Convergence chart → watch Client 3 (Energy+Auto+Metals+Infra — 23 stocks) — it has lowest starting loss because of more data
2. Client 1 (IT, only 6 stocks) starts with worst loss but benefits most from global model by round 50 — this is the **fairness benefit** of FL
3. **Fairness Comparison** bar chart: With FL, all clients improve. Without FL, small clients (IT) perform much worse. This demonstrates FL's equity across participants.
4. Check: if FedAvg and FedProx lines converge to same value at round 50 → data is IID enough that proximal term isn't critical. If they diverge → Non-IID data (which is realistic for different sectors).

**What to look for:**
- If global Sharpe > any individual client's Sharpe → FL is adding value through cross-sector knowledge transfer
- If convergence stalls after round 20 → learning rate too high or proximal term mu too small
- Fairness chart: clients with fewer stocks should show the most improvement with FL (more to gain from global knowledge)

---

## Tab 5 — Sentiment Monitor (`/sentiment`)

### What it shows
Real-time financial news sentiment for NIFTY 50 stocks, analyzed by FinBERT — a BERT model fine-tuned on 10,000+ financial texts. Auto-refreshes every 3 minutes.

### Live indicators
- **LIVE badge** with pulsing green dot — confirms data is live, not cached
- **"Updated Xs ago"** timer — tells you how fresh the current data is
- **+N new** badge — appears on News tab when fresh headlines arrive since last refresh
- **Sentiment Trend chart** (top of page) — builds over your session, saved in browser localStorage

### How professionals use it

**Active Trader / Retail Investor:**
1. Open the tab — it automatically fetches live news and runs FinBERT (takes 15–30 seconds, skeleton loader shows)
2. Watch the **Market Mood** metric card — "Bullish / Neutral / Bearish" at a glance
3. Check **Avg Score** — anything above +0.10 = meaningfully positive news flow for the market today
4. Click the **News tab** → scroll the Live Feed strip at top (top 8 stocks, color-coded green/red/grey)
5. Click any **headline card** to expand it → see the full FinBERT breakdown:
   - Score badge (▲ for positive, ▼ for negative, — for neutral)
   - Confidence level (High/Medium/Low based on how strongly FinBERT is sure)
   - Stacked probability bar (green = positive %, grey = neutral %, red = negative %)
   - Three probability boxes with exact percentages
   - Net Sentiment Score (positive_prob − negative_prob)
6. **No external links** — all analysis is shown inline. You do not leave the platform.

**Investment Advisor (RIA):**
1. Click **Portfolio Impact tab** → table shows all 44 stocks with:
   - Sentiment Score per stock
   - Base Weight (equal weight = ~2.3% each)
   - Adjusted Weight (sentiment-modified)
   - Weight Change (+ = overweight due to positive news, - = underweight)
2. Top Mover metric card shows the stock with the largest sentiment-driven weight shift — use this to flag to clients: "Reliance has strong positive news today — sentiment model is overweighting it"
3. Click **Sectors tab** → horizontal bar chart of sector sentiment scores → identify which sector has best/worst news momentum today
4. Sector detail cards show: how many headlines, % positive, % negative per sector

**Portfolio Manager:**
1. Watch the **Sentiment Trend chart** over the trading day — if the trend line moves from +0.05 to -0.10, market news sentiment is deteriorating → consider risk reduction
2. The trend is saved in localStorage → comes back even if you refresh the page
3. Refresh button (manual) → forces an immediate FinBERT re-run → useful before important news events

**What to look for:**
- Avg Score > +0.15 on earnings day = strong bullish signal → consider momentum positions
- Sector with negative score but positive individual stock scores = mixed signals → reduce position sizing confidence
- Market Mood = Bearish + Survival Rate (from Stress Testing) < 92% → risk-off day → reduce exposure

---

## Tab 6 — Graph Visualization (`/graph`)

### What it shows
Interactive force-directed network graph of all NIFTY 50 stocks. Nodes = stocks, edges = relationships. 3 edge types: Sector (same industry), Supply Chain (business dependencies), Correlation (60-day price co-movement > 0.6).

### How to interact
- **Click any node** → right panel shows stock details, connections, neighbor list
- **Hover** → highlights all connected stocks, dims unconnected ones
- **Toggle edges** → show/hide Sector / Supply Chain / Correlation edges independently
- **Node size** = degree (number of connections). Bigger node = more connected stock.

### How professionals use it

**Quant Analyst / Portfolio Manager:**
1. Turn on only **Correlation edges** (turn off Sector and Supply Chain)
2. Now you see pure price co-movement clusters — stocks that move together
3. Key insight: if two stocks from different sectors have a strong correlation edge, they are not providing true diversification
4. If Banking cluster and IT cluster are heavily cross-connected → your portfolio has hidden concentration risk
5. Click HDFCBANK → see all its correlated neighbors → if ICICIBANK, KOTAKBANK, SBIN are all neighbors → these 4 are highly correlated → equal-weighting them gives less diversification than it appears

**Risk Manager:**
1. Turn on **Supply Chain edges** only
2. Find TATASTEEL — it has supply chain edges to TATAMOTORS, MARUTI (they buy steel)
3. If TATASTEEL is in trouble, these downstream companies are also at risk → correlation doesn't capture this, but supply chain does
4. This is exactly what GNN (T-GAT) uses as input — relationship-aware embeddings that pure price models miss

**Researcher / Examiner:**
1. The clustering visible in this graph is what T-GAT learns to encode into 64-dimensional embeddings
2. Sector edges → T-GAT learns that Banking stocks should have similar embeddings
3. Correlation edges → T-GAT learns cross-sector price relationships
4. Supply chain edges → T-GAT captures fundamental business links (not just price correlation)
5. Turn all 3 on → count the total edges → this is the graph density metric shown in the bottom-right stats panel

---

## Tab 7 — Pipeline Visualization (`/workflow`)

### What it shows
Animated, interactive end-to-end diagram of the entire FINQUANT-NEXUS AI pipeline — from raw data ingestion to final portfolio output. 15 components, 19 connections.

### How to use it
- Click **▶ Play** → pipeline animates stage by stage (9 stages, 1.8 seconds each)
- Click **Pause** to stop at any stage
- Click **any node** in the diagram to see its technical details in the right panel (Input, Output, Key Details)
- Click any **stage in the list** (right panel, bottom) to jump directly to that stage
- Progress dots on the right edge of the SVG track which stages are complete

### The 9 stages

| Stage | What activates | Description |
|---|---|---|
| ① | NIFTY 50 Data + Google News | Raw data ingestion |
| ② | Feature Eng. + FinBERT | 21 indicators computed, headlines scored |
| ③ | Graph Builder | 3-edge-type stock network constructed |
| ④ | T-GAT Model | Temporal graph attention → 64-dim embeddings |
| ⑤ | RL Environment | Gymnasium env initialized with all features |
| ⑥ | 5 RL Agents | PPO, SAC, TD3, A2C, DDPG train in parallel |
| ⑦ | ★ Ensemble | 5 model predictions averaged → consensus weights |
| ⑧ | Federated Learning | 4 sector clients + DP-SGD privacy aggregation |
| ⑨ | Portfolio Output | Final allocation with full risk metrics |

### How professionals use it

**Dissertation Examiner / Evaluator:**
1. Play the full animation — shows the evaluator you understand the end-to-end data flow
2. Click **FinBERT node** → right panel shows: Input (headline strings), Output (score ∈ [-1,+1]), Key details (thread-safe singleton, FP16, SQLite cache)
3. Click **T-GAT node** → shows: GATConv × 3 edge types, GRU temporal encoder, output shape (n_stocks, 64)
4. Click **Ensemble node** → explains: average of 5 raw action vectors, then softmax by RL env
5. Use this tab to narrate your dissertation chapter structure: each node = a section

**Client / Stakeholder Demo:**
1. Play the pipeline — visually compelling, shows complexity without code
2. The **System Stats** panel (bottom right) summarizes: 15 components, 19 connections, 5+1 RL algos, 4 FL clients, 246 passing tests, ε=8 privacy
3. Use this in investor/demo presentations as the opening slide equivalent

---

## Common Workflows

### "I want to pick the best algorithm for today's market"
1. Sentiment tab → check Market Mood (Bullish/Neutral/Bearish)
2. RL Agent tab → Comparison table → if Bullish: look at TD3 (momentum-aggressive), if Bearish: look at A2C (contrarian)
3. Portfolio tab → click the highest-Sharpe stock in Holdings → check its sentiment score in Sentiment tab
4. If consistent signal → use Ensemble (always safest, reduces single-algo bias)

### "I want to run a risk report for a client"
1. Portfolio tab → screenshot the 5 metric cards and sector weights chart
2. Stress Testing tab → set stocks=44, simulations=5000 → run → record VaR, CVaR, Survival Rate for all 4 scenarios
3. Sentiment tab → note the Market Mood and Avg Score for today
4. Combine: "Portfolio Sharpe X, Normal VaR Y%, under 2008 scenario VaR Z%, current sentiment mood: Bullish"

### "I want to understand hidden portfolio risks"
1. Graph Viz tab → turn on correlation edges only → identify tightly clustered groups
2. Click the most-connected node → see all its correlated neighbors
3. If 8+ stocks are all correlated → they are not diversifying each other despite being separate holdings
4. Cross-reference with Portfolio tab Holdings → these correlated stocks should not all be max-weight

### "I want to explain federated learning to a compliance team"
1. Federated tab → Privacy ε metric card → click it → read the explanation
2. Convergence chart → show that the global model improves for all clients over 50 rounds
3. Fairness chart → show that Client 1 (IT, only 6 stocks) improves most — even small participants benefit
4. Key message: "No raw data leaves any institution. Only encrypted model gradients are shared. ε=8.0 gives mathematical proof that individual records cannot be reconstructed."

---

## API Usage (for Developers/Integration)

All data in the dashboard comes from a FastAPI backend. Every endpoint is available directly:

```
GET  /api/portfolio-summary     → Portfolio metrics + holdings
GET  /api/stock/{ticker}        → Individual stock details
GET  /api/rl-summary            → All 6 algorithm performance data
GET  /api/fl-summary            → Federated learning results
GET  /api/gnn-summary           → Graph nodes, edges, statistics
GET  /api/news-sentiment        → Live FinBERT analysis (takes 15–30s)
POST /api/sentiment             → Analyze custom text with FinBERT
POST /api/stress-test           → Run Monte Carlo with custom parameters
POST /api/portfolio-growth      → Historical growth from any start date
```

Full interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Page shows "Failed to load" | Backend not running | Run `uvicorn src.api.main:app --port 8000` |
| Sentiment takes > 60s | FinBERT loading for first time | Wait — model caches after first load |
| Graph has no edges | Backend GNN data missing | Check `data/all_close_prices.csv` exists |
| RL data shows all zeros | Cache issue | Restart uvicorn, hard-refresh browser |
| Stress test "Failed" | Backend timeout on large simulation | Use n_simulations ≤ 10,000 |

---

*FINQUANT-NEXUS v4 — Built for NIFTY 50 Portfolio Optimization Research*
*Backend: FastAPI + Python | Frontend: React + TypeScript + Vite | Models: SB3 + HuggingFace FinBERT + PyG T-GAT*
