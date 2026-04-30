# FINQUANT-NEXUS v4 — Master Dissertation Planner
## Complete Chapter Guide | Context Preservation File

> **Purpose**: This file is your single source of truth. Open it at the start of every new session.
> Paste the relevant section into Claude along with the Writing Prompt at the bottom.

---

## Student & Project Identity

| Field | Value |
|-------|-------|
| Name | Praveen Pal Rawal |
| Enrollment No | 240031105151008 |
| Degree | MTech in Data Science and Machine Learning (Sem 4 — Final) |
| University | Rashtriya Raksha University (RRU), Lavad, Dehgam, Gandhinagar-382305, Gujarat |
| Supervisor | Dr. Mayur Makwana |
| Organization | Rashtriya Raksha University, Gandhinagar |
| Dissertation Title | FINQUANT-NEXUS v4: An AI-Powered Portfolio Optimization System for NIFTY 50 Using Deep Reinforcement Learning, Graph Neural Networks, Sentiment Analysis, and Federated Learning |

---

## Formatting Rules (from RRU .docx Template)

| Property | Value |
|----------|-------|
| Font | Times New Roman |
| Size | 12-point |
| Line Spacing | 1.5 |
| Left Margin | 38mm (1.5 inch) |
| Other Margins | 25.4mm (1 inch) each |
| Paper | A4, 80–90 gsm, single side |
| Max Pages | 150 (body only) |
| Abstract Limit | 250 words max |
| Hardbound | Navy Blue, Golden font |
| Citations | Superscript numbers [1], [2] |

---

## Page Estimate & Progress Tracker

| # | Section | Est. Pages | Status |
|---|---------|------------|--------|
| - | Front Matter (Title, Declaration, Certificate, Ack, TOC, List of Figs/Tables, Abstract) | 12–14 | Not Started |
| 1 | Chapter 1 — Introduction | 8–10 | Not Started |
| 2 | Chapter 2 — Literature Review | 15–20 | Not Started |
| 3 | Chapter 3 — System Design & Methodology | 28–35 | Not Started |
| 4 | Chapter 4 — Implementation & Results | 25–30 | Not Started |
| 5 | Chapter 5 — Analysis & Discussion | 15–18 | Not Started |
| 6 | Chapter 6 — Conclusions | 5–7 | Not Started |
| - | References (50–60 citations) | 4–5 | Not Started |
| - | Appendices | 8–10 | Not Started |
| - | Plagiarism Certificate | 1 | Not Started |
| | **TOTAL** | **~120–140** | |

---

## Project Technical Summary (Paste this in every session)

### What the Project Does
FINQUANT-NEXUS v4 is an AI-based portfolio optimization system built for NIFTY 50 Indian stocks. It combines multiple machine learning techniques into one unified platform with a real-time web dashboard. The system takes raw stock price data, news headlines, and market relationships as input and outputs intelligent portfolio weight allocations.

### What is INCLUDED in Dissertation (UI-driven scope)
Only the features visible in the dashboard frontend are covered:
1. Portfolio Analytics (Sharpe, Sortino, Return, Drawdown, Holdings, Growth Chart)
2. Reinforcement Learning Agent (6 algorithms + ensemble)
3. Stress Testing (Monte Carlo, VaR, CVaR, multiple crisis scenarios, 1000 paths)
4. Federated Learning (FedProx, DP-SGD, 4 sector clients, 50 rounds)
5. Sentiment Analysis (FinBERT, live news, daily scores)
6. Graph Visualization (T-GAT, 3 edge types, force-directed network)
7. Pipeline / Workflow (end-to-end animated view)
8. **Future Prediction** (Forward simulation, Black Bootstrap, Monte Carlo fan chart, algorithm comparison)

### What is EXCLUDED (not in dissertation scope)
- QAOA / Quantum Optimization
- NAS / DARTS Architecture Search
- TimeGAN (synthetic data generation — backend only, not shown in UI)

### Data Details
- Stocks: 44 NIFTY 50 constituents (Yahoo Finance via yfinance)
- Period: 2015–2025
- Train: 2015–2021 | Validation: 2022–2023 | Test: 2024–2025
- Benchmark: NIFTY 50 Index
- Risk-free rate: 5% (India)

### Feature Engineering
- 21 technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic, SMA/EMA, OBV, Volume Ratio
- Normalization: Z-score, rolling 252-day window
- No data leakage (strict temporal ordering)

### Sentiment Module
- Model: ProsusAI/finbert (locally cached)
- News sources: Google News, Yahoo Finance, Indian RSS feeds
- Output: Daily sentiment score per stock, range −1 to +1
- Updates every 3 minutes in live dashboard

### Graph Construction
- 3 edge types:
  - Sector edges (stocks in same industry)
  - Supply chain edges (business dependencies between companies)
  - Correlation edges (60-day rolling correlation > 0.6)
- Model: T-GAT (Temporal Graph Attention Network)
  - RelationalGATLayer per edge type
  - GRU temporal encoder
  - Output: 64-dimensional stock embedding per stock

### RL Framework
- Environment: Gymnasium-compatible PortfolioEnv
  - State: 21 features + current weights + T-GAT embeddings + sentiment scores
  - Action: Continuous portfolio weights (softmax normalized, sum = 1)
  - Reward: Sharpe ratio − drawdown penalty − turnover penalty
  - Constraints: max 12% per stock, stop-loss −3%, max drawdown −12%
- Algorithms:
  - PPO (Proximal Policy Optimization) — stable, momentum-weighted
  - SAC (Soft Actor-Critic) — entropy-maximizing, uncertainty-aware
  - TD3 (Twin Delayed DDPG) — aggressive trending
  - A2C (Advantage Actor-Critic) — contrarian
  - DDPG (Deep Deterministic Policy Gradient) — balanced
  - Ensemble — meta-average of all 5

### Stress Testing
- Scenarios: Normal, Crash 1999, Crash 2000, Flash Crash, Dot Com Crash, India Bear 2015, India Bear 2018
- 1000 Monte Carlo simulation paths per scenario
- Metrics: VaR 95%, CVaR 95%, Mean Return, Survival Rate
- Overall: VaR(95%) = -15.89%, CVaR = -21.32%, Survival = 34.4%

### Future Prediction (BONUS TAB — included in dissertation)
- Method: Black Bootstrap (GAN-calibrated, 30-day blocks)
- Horizon: 1 Year | Scenarios: 1000 Monte Carlo paths
- Median Return: +9.3% | Best Case: +31.6% | Worst Case: -11.1%
- Probability of Profit: 75.9%
- Shows algorithm-wise forward simulation comparison
- Forward allocation (Ensemble): ICICIBANK 3.8%, HINDUNILVR 3.7%, GRASIM 3.0%, SBIN 2.9%

### Benchmark Results (from Growth Chart)
- Period: April 2025 – March 2026 (247 trading days)
- Our Portfolio: +8.27% (₹10,82,745 from ₹10,00,000)
- NIFTY 50 Index: +0.65% (₹10,06,550)
- Fixed Deposit 7%: +4.96% (₹10,49,587)
- Portfolio outperforms NIFTY 50 by 7.62 percentage points

### Federated Learning
- 4 sector clients: Banking | IT/Telecom | Pharma/FMCG | Energy/Auto/Metals
- Aggregation: FedProx (proximal term μ = 0.01)
- Privacy: DP-SGD with Gaussian noise (ε = 8.0)
- Rounds: 50 communication rounds
- No raw data shared — only model weight updates

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI |
| ML/DL | PyTorch 2.1, Stable-Baselines3 |
| NLP | HuggingFace Transformers, FinBERT |
| GNN | PyTorch Geometric, GAT |
| Data | Pandas, NumPy, yfinance |
| Federated | Flower (flwr) |
| Frontend | React 19, TypeScript, Vite |
| Charts | Recharts |
| Styling | Tailwind CSS, Framer Motion |
| API | FastAPI, Pydantic |
| Testing | Pytest (246 tests, 14 files) |
| Hardware | RTX 3050 4GB VRAM |

---

## CHAPTER-WISE DETAILED OUTLINE

---

### FRONT MATTER (~12–14 pages)

**Pages needed**: Title Page, Cover Page, Declaration, Certificate, Acknowledgements, TOC, List of Figures, List of Tables, Abstract

#### Abstract (write this last, max 250 words)
Structure:
- **Objective**: What the system aims to do
- **Work Done**: Which techniques were implemented and how
- **Results**: Key performance metrics
- **Conclusions**: What the system achieves and its significance

Key points to mention in abstract:
- NIFTY 50, 2015–2025
- RL ensemble (PPO, SAC, TD3, A2C, DDPG)
- T-GAT for stock relationships
- FinBERT for news sentiment
- FedProx + DP-SGD for privacy
- Interactive React dashboard
- Outperforms NIFTY 50 benchmark (mention actual metric when results are ready)

---

### CHAPTER 1 — Introduction (~8–10 pages)

**Goal**: Set the context, explain why this project matters, state objectives clearly.

#### 1.1 Background and Motivation (2–3 pages)
- Indian stock market overview: NIFTY 50 as benchmark, market size, retail investor growth
- Traditional portfolio theory: Markowitz MPT limitations (assumes normal distribution, static correlation)
- Rise of algorithmic trading and AI in finance globally
- Why Indian markets specifically need AI-based tools (high volatility, sentiment-driven moves, supply chain relationships)
- Transition: single model approaches not sufficient → need multi-modal integration
- Motivation: build one unified system that captures price patterns + relationships + news + privacy

**Figures needed**: None mandatory. Optional: NIFTY 50 growth chart.

#### 1.2 Problem Statement (0.5–1 page)
- Existing systems handle one aspect (only RL, or only sentiment)
- No existing system for Indian markets combines RL + GNN + NLP + FL together
- Privacy barrier in collaborative financial ML
- Lack of real-time interactive analytics dashboard for retail/institutional investors

#### 1.3 Objectives (0.5 page)
List clearly:
1. Collect and process NIFTY 50 historical data with 21 technical indicators
2. Build FinBERT-based sentiment pipeline for Indian financial news
3. Construct multi-relational stock graph and train T-GAT
4. Develop Gymnasium RL environment and train 6 RL algorithms + ensemble
5. Implement stress testing via Monte Carlo simulation
6. Design privacy-preserving federated learning system (FedProx + DP-SGD)
7. Build REST API and React dashboard with 7 analytical views

#### 1.4 Scope of Work (0.5 page)
- NIFTY 50 universe (44 stocks), 2015–2025
- Simulation environment (not live brokerage trading)
- Local deployment (Python backend + React frontend)
- Excluded: Quantum optimization, NAS, live order execution

#### 1.5 Organization of the Dissertation (0.5 page)
- Brief one-paragraph description of what each chapter covers
- Helps examiner understand structure

---

### CHAPTER 2 — Literature Review (~15–20 pages)

**Goal**: Show awareness of existing research. Explain what others did and where the gap is.

#### 2.1 Portfolio Optimization: Classical Approaches (2–3 pages)
- Markowitz Mean-Variance Optimization (1952) [REF]
- Capital Asset Pricing Model (CAPM) [REF]
- Black-Litterman model [REF]
- Limitations: assumption of normal returns, static covariance matrix, no market regime awareness
- Transition to data-driven methods

#### 2.2 Deep Reinforcement Learning in Finance (3–4 pages)
- Early RL in trading: Q-Learning approaches [REF]
- Policy gradient methods for portfolio management [REF]
- PPO application in finance [REF]
- SAC for high-dimensional action spaces [REF]
- TD3 and DDPG for continuous control [REF]
- FinRL library and Stable-Baselines3 ecosystem [REF]
- Ensemble approaches in RL [REF]
- Gap: most work on US markets (S&P 500), very little on NIFTY 50

#### 2.3 Graph Neural Networks for Stock Markets (2–3 pages)
- GCN fundamentals: Kipf & Welling [REF]
- Graph Attention Networks (GAT): Velickovic et al. [REF]
- Stock relationship modeling via graphs [REF]
- Temporal GNN approaches for time-series [REF]
- Multi-relational graph approaches [REF]
- Gap: most GNN work uses single edge type (only correlation), not multi-relational

#### 2.4 Financial Sentiment Analysis (2–3 pages)
- Traditional sentiment: lexicon-based (Loughran-McDonald dictionary) [REF]
- BERT for NLP [REF]
- FinBERT: fine-tuned BERT for financial texts (Araci, 2019) [REF]
- Sentiment alpha studies in stock markets [REF]
- Indian financial news sentiment — limited prior work
- Gap: most systems use English global news, not Indian financial news sources

#### 2.5 Federated Learning in Financial Applications (2–3 pages)
- FedAvg: McMahan et al. [REF]
- FedProx for heterogeneous data [REF]
- Differential Privacy: DP-SGD (Abadi et al.) [REF]
- FL in banking and finance [REF]
- Privacy-utility tradeoff in financial ML [REF]
- Gap: FL rarely combined with RL for portfolio optimization

#### 2.6 Monte Carlo Methods in Risk Management (1–2 pages)
- Value at Risk (VaR) background [REF]
- Conditional VaR / Expected Shortfall [REF]
- Monte Carlo simulation for portfolio stress testing [REF]
- Historical crisis scenario analysis [REF]

#### 2.7 Research Gap and Contribution (1 page)
- Summary table: what prior works do vs what this work does
- Clear statement of novelty: Indian market + multi-modal (RL + GNN + NLP + FL) + unified dashboard

**Table needed**: Comparison table of related works (Author | Year | Method | Dataset | Limitation)

---

### CHAPTER 3 — System Design & Methodology (~28–35 pages)

**Goal**: Explain HOW the system is designed. This is the most important chapter.

#### 3.1 Overall System Architecture (2–3 pages)
- End-to-end pipeline description
- Data flow: Raw Data → Feature Eng → Sentiment → Graph → T-GAT → RL → API → Dashboard
- Component interaction diagram

**Figure needed**: Full system architecture diagram (most important figure in dissertation)

#### 3.2 Dataset Description (2 pages)
- NIFTY 50 stock list: 44 tickers, which sectors
- Data source: Yahoo Finance via yfinance library
- Date range: 2015–2025
- Train/Val/Test split rationale: 2015–2021 / 2022–2023 / 2024–2025
- NIFTY 50 Index as benchmark
- Data fields: Open, High, Low, Close, Volume, Adjusted Close

**Table needed**: Dataset statistics (stocks per sector, date range, total rows)

#### 3.3 Data Preprocessing (1.5 pages)
- Forward-fill NaN values (market holidays)
- Outlier detection and handling
- Adjusted close vs raw close: why adjusted close used
- Temporal ordering: no future data leakage

#### 3.4 Feature Engineering — Technical Indicators (2–3 pages)
- Why technical indicators? Capture momentum, trend, volume patterns
- List all 21 indicators with formulas for key ones:
  - RSI (Relative Strength Index): formula
  - MACD (Moving Average Convergence Divergence): formula
  - Bollinger Bands: formula
  - ATR (Average True Range)
  - Stochastic Oscillator
  - SMA/EMA
  - OBV (On-Balance Volume)
  - Volume Ratio
- Normalization: Z-score with rolling 252-day window
- Why rolling normalization instead of global? Avoids look-ahead bias

**Table needed**: All 21 indicators — Name | Type | Formula | Purpose

#### 3.5 Sentiment Analysis Module (3–4 pages)
- Financial NLP background: why general sentiment models fail for finance
- FinBERT: what it is, how it was trained (ProsusAI fine-tuning on financial corpus)
- News sources used: Google News, Yahoo Finance, Indian RSS feeds
- News fetching pipeline: thread-safe, timeout handling, deduplication
- Tokenization → inference → score extraction
- Score aggregation: daily sentiment per stock (weighted average)
- Output: sentiment score matrix (stocks × days), range −1 to +1
- Market mood classification: Bullish (>0.2) / Neutral / Bearish (<−0.2)
- Caching: SQLite cache with 3-minute TTL

**Figure needed**: Sentiment pipeline flow diagram

#### 3.6 Stock Relationship Graph Construction (2–3 pages)
- Why model stocks as a graph? Stocks don't move in isolation
- 3 edge types and their meaning:
  - Sector edges: stocks in same GICS sector (Banking, IT, Pharma, etc.)
  - Supply chain edges: manually defined business dependencies (e.g., TATASTEEL supplies MARUTI)
  - Correlation edges: 60-day rolling Pearson correlation > 0.6 threshold — dynamic
- Graph properties: 44 nodes, variable edges over time
- PyTorch Geometric graph construction
- Why multi-relational? Different edge types capture different market dynamics

**Figure needed**: Graph visualization screenshot from dashboard (3 edge types visible)
**Table needed**: Edge statistics (count per type)

#### 3.7 Temporal Graph Attention Network (T-GAT) (3–4 pages)
- Why GNN for stock embeddings? Captures structural relationships
- Standard GAT: how attention mechanism works — formula
- RelationalGATLayer: separate attention per edge type
- GRU temporal encoder: captures sequential price dynamics
- How sentiment + features are fused as node features
- Training: loss function, optimizer
- Output: 64-dimensional embedding per stock
- How embeddings fed into RL observation space

**Figure needed**: T-GAT architecture block diagram

#### 3.8 Reinforcement Learning Environment (3 pages)
- Why RL for portfolio? Sequential decision making under uncertainty
- Gymnasium-compatible PortfolioEnv
- **Observation space**: (21 features × 44 stocks) + current weights + 64-dim embeddings + sentiment
- **Action space**: continuous weight vector for 44 stocks (softmax ensures sum = 1)
- **Reward function**: Sharpe ratio − λ₁ × drawdown − λ₂ × turnover
  - Why Sharpe? Risk-adjusted returns more meaningful than raw returns
  - Why turnover penalty? Prevents excessive trading (realistic constraint)
- **Constraints enforced**:
  - Max position: 12% per stock
  - Stop-loss: −3% individual
  - Max drawdown: −12% portfolio
- Episode length: 252 trading days (1 year)

**Figure needed**: RL environment state-action-reward cycle diagram

#### 3.9 Reinforcement Learning Agents (3–4 pages)
- Overview: each algorithm trained separately, then ensemble
- **PPO**: clipped surrogate objective, why stable for finance
- **SAC**: entropy regularization, exploration encouragement
- **TD3**: twin critic networks, delayed policy updates, noise injection
- **A2C**: synchronous advantage estimation
- **DDPG**: deterministic policy, experience replay
- **Ensemble**: simple average of 5 action outputs, why it outperforms individuals
- Training: 500,000 steps on 2015–2021 data
- Validation: 2022–2023 (hyperparameter tuning)
- Test: 2024–2025 (final evaluation)

**Table needed**: Hyperparameter table for each algorithm

#### 3.10 Stress Testing Framework (2 pages)
- Purpose: evaluate portfolio resilience under extreme market conditions
- 4 crisis scenarios:
  - Normal Market: historical volatility
  - 2008 Financial Crisis: 3.5× historical volatility multiplier
  - COVID-19 Crash (March 2020): 5× volatility
  - Flash Crash scenario: 8× volatility, sudden drawdown
- Monte Carlo simulation: 10,000 paths per scenario
- Metrics computed:
  - VaR 95%: maximum loss not exceeded with 95% confidence
  - VaR 99%: stricter threshold
  - CVaR (Expected Shortfall): average loss beyond VaR
  - Survival Rate: % paths with drawdown < 12%

**Figure needed**: Monte Carlo fan chart (one scenario example)

#### 3.11 Federated Learning System (3 pages)
- Motivation: sector-based collaboration without sharing raw price/portfolio data
- 4 sector clients and their stock groups:
  - Banking: HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK, etc.
  - IT/Telecom: TCS, INFOSYS, WIPRO, HCLTECH, BHARTIARTL
  - Pharma/FMCG: SUNPHARMA, DRREDDY, DIVISLAB, HINDUNILVR, ITC
  - Energy/Auto/Metals: RELIANCE, ONGC, MARUTI, TATAMOTORS, TATASTEEL
- FedAvg vs FedProx: why FedProx is better for heterogeneous sector data (proximal term prevents client model from drifting too far)
- FedProx proximal term: formula with μ = 0.01
- DP-SGD: Gaussian noise added to gradients before sharing
  - Privacy budget ε = 8.0
  - What ε means: lower = more private, higher = more utility
- 50 communication rounds: each round — local train → send weights → aggregate → broadcast
- What is shared: only model weight updates, never raw stock data

**Figure needed**: FL system diagram (4 clients → server → aggregate → broadcast)

#### 3.12 REST API Design (1.5 pages)
- FastAPI framework
- Key endpoints used by dashboard:
  - GET /api/health
  - GET /api/portfolio-summary
  - GET /api/rl-summary
  - GET /api/news-sentiment
  - GET /api/stress-test
  - GET /api/fl-summary
  - GET /api/gnn-summary
- Caching strategy: CSV data in-memory, news TTL 180s
- CORS configuration for React on localhost:3000

#### 3.13 Dashboard Design (2 pages)
- React 19 + TypeScript + Vite
- 7 pages and their purpose:
  - Portfolio: performance metrics, holdings, growth chart
  - RL Agent: algorithm selector, comparison table, charts
  - Stress Testing: scenario cards, VaR table, Monte Carlo chart
  - Federated: convergence curves, privacy tracker, fairness chart
  - Sentiment: live feed, score chart, market mood badge
  - Graph Viz: force-directed network, edge toggles
  - Pipeline: animated workflow diagram
- Recharts for financial charts, Tailwind CSS for styling, Framer Motion for animations

---

### CHAPTER 4 — Implementation & Results (~25–30 pages)

**Goal**: Show what was actually built. Evidence chapter. Screenshots + tables + actual numbers.

#### 4.1 Development Environment (1 page)
**Table: Hardware & Software Setup**
| Component | Details |
|-----------|---------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3050 4GB VRAM |
| CPU | [your CPU] |
| RAM | [your RAM] |
| Python | 3.11 |
| PyTorch | 2.1 |
| CUDA | [version] |

#### 4.2 Data Collection & Processing Results (2 pages)
- How many stocks downloaded, date range confirmed
- Missing data stats: how many NaNs filled
- Feature matrix shape: (rows × 21 features × 44 stocks)
- Sample visualization: price chart of 3-4 stocks (2015–2025)

**Figure needed**: Sample stock price chart (RELIANCE, HDFCBANK, INFY)
**Table needed**: Dataset summary statistics

#### 4.3 Portfolio Analytics Dashboard (3 pages)
- Screenshot of Portfolio tab
- Explain each metric shown:
  - Sharpe Ratio: risk-adjusted return measure
  - Sortino Ratio: downside risk adjusted
  - Total Return %: cumulative
  - Max Drawdown: worst peak-to-trough
- Holdings table: top 10 stock allocations
- Growth curve: portfolio vs NIFTY 50 benchmark

**Figure needed**: Portfolio tab screenshot (full page)

#### 4.4 RL Training Results (4 pages)
- Training curves: reward over 500K steps for each algorithm
- Screenshot of RL Agent dashboard tab
- Comparison table of all 6 algorithms:

**Table needed**:
| Algorithm | Sharpe | Sortino | Total Return | Max Drawdown | Best For |
|-----------|--------|---------|--------------|--------------|----------|
| PPO | | | | | |
| SAC | | | | | |
| TD3 | | | | | |
| A2C | | | | | |
| DDPG | | | | | |
| Ensemble | | | | NIFTY 50 benchmark |

- Which algorithm performed best and why
- Ensemble advantage explanation

**Figures needed**: Training reward chart, Returns comparison chart

#### 4.5 Sentiment Analysis Results (3 pages)
- Screenshot of Sentiment tab
- Sample sentiment scores for key stocks over time
- Distribution chart: positive/negative/neutral proportion
- Example: correlation between sentiment spikes and price moves (qualitative)
- LIVE badge behavior explanation
- News feed screenshot

**Figure needed**: Sentiment tab screenshot, sentiment score chart

#### 4.6 Graph Visualization Results (3 pages)
- Screenshot of Graph tab
- Node count: 44 stocks
- Edge counts by type (sector, supply chain, correlation)
- Example: toggle between edge types
- How T-GAT embeddings quality was verified (loss curve during training)

**Figure needed**: Graph tab screenshot (all 3 edge types visible)
**Table needed**: Graph statistics (nodes, edges by type)

#### 4.7 Stress Testing Results (3 pages)
- Screenshot of Stress Testing tab
- Monte Carlo fan chart
- Risk metrics table per scenario:

**Table needed**:
| Scenario | VaR 95% | VaR 99% | CVaR | Survival Rate |
|----------|---------|---------|------|---------------|
| Normal | | | | |
| 2008 Crisis | | | | |
| COVID Crash | | | | |
| Flash Crash | | | | |

- Interpretation: which scenario is worst, how portfolio holds up

**Figure needed**: Stress Testing tab screenshot

#### 4.8 Federated Learning Results (3 pages)
- Screenshot of Federated tab
- FedProx vs FedAvg convergence curve (rounds vs accuracy/loss)
- Privacy epsilon tracker chart
- Per-client performance fairness chart

**Table needed**:
| Method | Rounds to Converge | Final Accuracy | Privacy ε |
|--------|-------------------|----------------|-----------|
| FedAvg | | | N/A |
| FedProx | | | 8.0 |

**Figure needed**: FL tab screenshot, convergence curve

#### 4.9 Pipeline Workflow Visualization (1 page)
- Screenshot of Workflow/Pipeline tab
- Brief explanation of each stage shown in animation

**Figure needed**: Pipeline tab screenshot

#### 4.10 Testing & Validation (2 pages)
- 246 test cases across 14 test files
- Test coverage per module
- Pytest results summary

**Table needed**: Test file | Module | Tests | Pass | Fail

---

### CHAPTER 5 — Analysis & Discussion (~15–18 pages)

**Goal**: Go deeper. Analyze WHY results are what they are. Compare, interpret, reflect.

#### 5.1 Portfolio Performance vs Benchmark (2–3 pages)
- Compare Ensemble RL vs Buy-and-Hold NIFTY 50
- Year-wise breakdown: 2022, 2023, 2024, 2025
- When did RL outperform? When did it underperform?
- Why Sharpe > 1.0 matters for institutional investors

#### 5.2 RL Algorithm Comparative Analysis (3 pages)
- PPO vs SAC stability comparison
- Why Ensemble consistently beats individual models (variance reduction argument)
- Which algorithm suits which market regime:
  - Trending market → TD3/DDPG better
  - Volatile market → SAC better (entropy exploration)
  - Stable market → PPO better
- Reward function design impact

#### 5.3 Sentiment Impact on Portfolio Decisions (2 pages)
- Does sentiment score correlate with RL weight changes?
- Example: negative sentiment on a stock → RL reduced allocation
- Latency issue: 15–30 seconds per FinBERT batch (practical limitation)
- Indian news coverage bias: English news vs Hindi news gap

#### 5.4 T-GAT Graph Embedding Quality (2 pages)
- Did T-GAT embeddings improve RL performance vs baseline (no embeddings)?
- Which edge type contributed most?
- Sector edges vs correlation edges effectiveness

#### 5.5 Stress Testing Interpretation (2 pages)
- Portfolio behavior under 2008 scenario vs COVID scenario
- CVaR interpretation: practical meaning for a fund manager
- Which algorithm survives stress best?

#### 5.6 Federated Learning Analysis (2 pages)
- FedProx vs FedAvg: convergence speed comparison
- Privacy-utility tradeoff: at ε = 8.0, how much accuracy lost vs ε = ∞ (no noise)?
- Client fairness: did any one sector dominate the global model?

#### 5.7 Limitations (1–2 pages)
- Simulation only: real brokerage integration not done
- FinBERT covers English news; Hindi/regional news missed
- Hardware constraint: RTX 3050 4GB required gradient accumulation
- QAOA/NAS not included (beyond current scope)
- Federated setup is simulated locally (not across real network nodes)
- DP privacy budget ε = 8.0 is on higher side (more utility, less strict privacy)

---

### CHAPTER 6 — Conclusions & Future Work (~5–7 pages)

**Goal**: Wrap up cleanly. What was achieved, what is next.

#### 6.1 Summary of Work Done (1.5 pages)
- Recap the full pipeline in 3–4 paragraphs
- What each component contributed
- System runs end-to-end: data → dashboard

#### 6.2 Key Contributions (1 page)
List clearly:
1. First integrated RL + GNN + NLP + FL system for NIFTY 50 Indian stocks
2. Multi-relational T-GAT graph (3 edge types) for stock relationship modeling
3. Privacy-preserving portfolio optimization via FedProx + DP-SGD
4. Comprehensive interactive dashboard (7 views) for real-time analytics
5. Ensemble RL consistently outperforms NIFTY 50 benchmark

#### 6.3 Conclusions (1 page)
- What the results prove about AI-based portfolio management
- Practical implications for Indian retail and institutional investors
- Academic contribution to the field

#### 6.4 Future Work (1.5–2 pages)
- Live trading integration: Zerodha Kite API or Upstox API
- Expand to BSE 200 universe
- FinBERT optimization: quantization to reduce latency
- Mobile dashboard: React Native app
- Real federated network: deploy across actual sector institutions
- Advanced risk models: regime detection (HMM) before RL

---

### REFERENCES (~4–5 pages)

**Target**: 50–60 references
**Format**: Superscript numbers [1] in text, full citation in References section

**Key papers to include (find actual DOIs/details):**
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. ICML.
- Fujimoto, S. et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods (TD3). ICML.
- Velickovic, P. et al. (2018). Graph Attention Networks. ICLR.
- Kipf, T. & Welling, M. (2017). Semi-Supervised Classification with GCN. ICLR.
- Araci, D. (2019). FinBERT. arXiv.
- Devlin, J. et al. (2019). BERT. NAACL.
- McMahan, B. et al. (2017). Communication-Efficient Learning (FedAvg). AISTATS.
- Li, T. et al. (2020). FedProx. MLSys.
- Abadi, M. et al. (2016). Deep Learning with Differential Privacy (DP-SGD). CCS.
- FinRL Library paper
- yfinance documentation
- Stable-Baselines3 paper
- PyTorch Geometric paper
- Flower (flwr) federated learning framework paper

---

### APPENDICES (~8–10 pages)

#### Appendix A — Full System Architecture Diagram
Full high-resolution diagram of the complete system

#### Appendix B — Configuration File (configs/base.yaml)
Complete hyperparameter configuration listing

#### Appendix C — REST API Endpoint Reference
All 50+ FastAPI endpoints with method, path, description

#### Appendix D — Test Results Summary
All 14 test files, 246 test cases, pass/fail counts

#### Appendix E — List of Abbreviations
| Abbreviation | Full Form |
|-------------|-----------|
| RL | Reinforcement Learning |
| GNN | Graph Neural Network |
| GAT | Graph Attention Network |
| T-GAT | Temporal Graph Attention Network |
| NLP | Natural Language Processing |
| FL | Federated Learning |
| FedProx | Federated Proximal Optimization |
| DP-SGD | Differentially Private Stochastic Gradient Descent |
| FinBERT | Financial BERT |
| NIFTY | National Fifty (NSE Index) |
| PPO | Proximal Policy Optimization |
| SAC | Soft Actor-Critic |
| TD3 | Twin Delayed DDPG |
| A2C | Advantage Actor-Critic |
| DDPG | Deep Deterministic Policy Gradient |
| VaR | Value at Risk |
| CVaR | Conditional Value at Risk |
| MPT | Modern Portfolio Theory |
| API | Application Programming Interface |
| REST | Representational State Transfer |
| CORS | Cross-Origin Resource Sharing |
| TTL | Time to Live |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| ATR | Average True Range |
| OBV | On-Balance Volume |
| EMA | Exponential Moving Average |
| SMA | Simple Moving Average |

---

## WRITING PROMPT — Copy This at Start of Every Chapter Session

```
You are an academic dissertation writing assistant.

Your task is to write dissertation content for M.Tech level students in India.

STRICT WRITING RULES:

1. Language Style
- Use simple Indian academic English.
- Keep sentences clear, natural, and human-written.
- Avoid robotic, overly polished, or Western research-paper tone.
- Avoid difficult vocabulary unless absolutely necessary.
- Write in a way that an M.Tech student can naturally explain during viva.

2. Originality Requirements
- Content must be fully plagiarism-free.
- Content must not sound AI-generated.
- Avoid repetitive sentence structures.
- Use natural variations in sentence length.
- Avoid template-style transitions.
- Do not copy from internet sources, papers, blogs, or books.

3. Humanization Rules
- Write like a genuine student researcher.
- Maintain slight natural imperfections in writing flow.
- Avoid exaggerated professionalism.
- Avoid marketing language and dramatic claims.
- Avoid "This revolutionary system...", "cutting-edge...", "state-of-the-art..." unless technically required.
- Keep explanations practical and realistic.

4. Academic Requirements
- Maintain proper academic structure.
- Use formal but readable tone.
- Ensure technical correctness.
- Keep explanations conceptually clear.
- Add logical flow between paragraphs.
- Avoid unnecessary filler content.
- Avoid unsupported claims.
- Use proper research-oriented explanation style.

5. Technical Writing Rules
- Explain concepts in simple terms first, then technical terms.
- Use examples where useful.
- Define abbreviations when first introduced.
- Keep paragraphs medium-sized and readable.
- Avoid very long paragraphs.
- Avoid excessive bullet points unless specifically requested.

6. Dissertation Formatting Style
- Write content suitable for M.Tech dissertation at Indian university.
- Maintain chapter continuity.
- Ensure smooth transitions between sections.
- Do not use conversational language.
- Do not use emojis.

7. AI Detection Reduction Rules
- Avoid predictable AI phrases.
- Avoid repeated transition words like: Furthermore, Moreover, In conclusion, Additionally.
- Use natural academic flow instead.
- Mix sentence patterns naturally.
- Occasionally use shorter explanatory sentences.

8. Content Quality Rules
- Content should be technically accurate, readable, practical, concise, academically acceptable.
- Do not over-explain basic concepts.
- Do not make fake citations — use [REF] placeholder where citation needed.
- If references are needed, leave [REF] placeholders only.

9. Output Rules
- Write in clean markdown format.
- Use proper headings and subheadings.
- Maintain numbering format properly.
- Keep content dissertation-ready.

PROJECT DETAILS:
[Project: FINQUANT-NEXUS v4]
[Student: Praveen Pal Rawal | 240031105151008]
[Degree: MTech Data Science and Machine Learning, Semester 4]
[University: Rashtriya Raksha University, Gandhinagar]
[Supervisor: Dr. Mayur Makwana]
[Scope: NIFTY 50 portfolio optimization using RL, GNN, NLP, Federated Learning]
[Excluded: QAOA, Quantum, NAS/DARTS]
[Dashboard: 7 pages — Portfolio, RL Agent, Stress Testing, Federated Learning, Sentiment, Graph Viz, Pipeline]
[Data: 44 NIFTY 50 stocks, Yahoo Finance, 2015-2025, Train/Val/Test: 2015-21 / 2022-23 / 2024-25]
[RL Algorithms: PPO, SAC, TD3, A2C, DDPG, Ensemble]
[GNN: T-GAT with 3 edge types — Sector, Supply Chain, Correlation]
[Sentiment: FinBERT (ProsusAI), local cache, 3 news sources]
[FL: FedProx + DP-SGD, 4 sector clients, 50 rounds, epsilon=8.0]
[Stress Testing: 4 scenarios, 10000 Monte Carlo paths, VaR/CVaR/Survival Rate]
[Tech Stack: Python 3.11, PyTorch 2.1, SB3, HuggingFace, PyG, Flower, FastAPI, React 19]
[Tests: 246 tests, 14 files, Pytest]

CHAPTER / SECTION TO WRITE:
[PASTE CHAPTER NAME AND SECTION HERE — e.g., "Chapter 1, Section 1.1 — Background and Motivation"]

SPECIAL INSTRUCTIONS:
[Add any specific instructions — e.g., "Write approx 3 pages", "Include formula for RSI", "Make it suitable for viva explanation"]
```

---

## Session Checklist — Do This Every Session

1. Open this file first
2. Update the **Status** column in Page Estimate table when a chapter is done
3. Copy the WRITING PROMPT above
4. Fill in `[CHAPTER / SECTION TO WRITE]`
5. Paste relevant section from the Chapter Outline above as context
6. Add any SPECIAL INSTRUCTIONS
7. Save the generated content in a separate file: `chapter1.md`, `chapter2.md`, etc.

---

*Last updated: 2026-04-29*
*Total estimated pages: 120–140 | RRU limit: 150 pages*
