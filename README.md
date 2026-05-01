<div align="center">

# FINQUANT-NEXUS v4

### An AI-Powered Self-Optimizing Portfolio Intelligence Platform for NIFTY 50

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-245%2F246%20Passing-22c55e?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-8B5CF6?style=for-the-badge)](LICENSE)

*MTech Dissertation — Rashtriya Raksha University, 2026*
*School of Information Technology, Artificial Intelligence and Cyber Security*

---

| Metric | Value |
|--------|-------|
| Portfolio Return (live) | **+8.27%** vs NIFTY +0.65% |
| Ensemble Sharpe Ratio | **0.8316** (backtest 2024–25) |
| Ensemble Annual Return | **+16.75%** (backtest 2024–25) |
| Max Drawdown | **−17.80%** (Ensemble) |
| Outperformance vs NIFTY 50 | **+7.62 pp** |
| Outperformance vs Fixed Deposit | **+3.31 pp** |

</div>

---

## What is FINQUANT-NEXUS?

FINQUANT-NEXUS is a research-grade portfolio management platform that combines four machine learning paradigms into a single end-to-end system:

- **Deep Reinforcement Learning** — five algorithms (PPO, SAC, TD3, A2C, DDPG) and a meta-level Ensemble agent allocate capital across 44 NIFTY 50 stocks
- **Graph Neural Networks** — a Temporal Graph Attention Network (T-GAT) models inter-stock relationships across sector, supply chain, and correlation dimensions
- **FinBERT Sentiment Analysis** — live financial news from three sources is processed by a domain-fine-tuned BERT model to extract stock-level sentiment scores
- **Federated Learning with Differential Privacy** — four sector clients train collaboratively under FedProx aggregation and DP-SGD (ε = 8.0) without sharing raw portfolio data

All results are surfaced through an eight-tab React dashboard backed by a FastAPI REST API with 50+ endpoints.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  Yahoo Finance (yfinance) — 44 NIFTY 50 stocks, 2015–2025       │
│  2,761 trading sessions · OHLCV · Adjusted Close                │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│  FEATURE ENGINEERING │    │        SENTIMENT MODULE             │
│  21 Technical        │    │  Google News RSS + Yahoo Finance    │
│  Indicators          │    │  + Indian RSS (Moneycontrol, ET)    │
│  Rolling Z-score     │    │  FinBERT → score ∈ [−1, +1]        │
│  252-day window      │    │  SQLite cache · TTL 3 min           │
│  Shape: (2761,44,21) │    │  Shape: (2761, 44)                  │
└──────────┬───────────┘    └────────────────────┬────────────────┘
           │                                     │
           ▼                                     │
┌──────────────────────┐                         │
│   GRAPH CONSTRUCTION │                         │
│  Sector edges:  79   │                         │
│  Supply chain:  24   │                         │
│  Correlation:  147   │                         │
│  (60-day, |r|>0.6)  │                         │
│  Total: 250 edges    │                         │
│  Density: 0.264      │                         │
└──────────┬───────────┘                         │
           │                                     │
           ▼                                     │
┌──────────────────────┐                         │
│       T-GAT          │                         │
│  Relational GAT      │                         │
│  8 heads/edge type   │                         │
│  GRU hidden: 128     │                         │
│  Output: 32-dim      │                         │
│  embeddings/stock    │                         │
└──────────┬───────────┘                         │
           │                                     │
           └──────────────┬──────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RL ENVIRONMENT (Gymnasium)                     │
│                                                                  │
│  Observation: 2,420 values                                       │
│  ├─ Technical indicators  44 × 21 = 924                          │
│  ├─ T-GAT embeddings      44 × 32 = 1,408                        │
│  ├─ Sentiment scores      44 × 1  = 44                           │
│  └─ Portfolio weights     44 × 1  = 44                           │
│                                                                  │
│  Action: softmax(44 values) → portfolio weights                  │
│  Reward: Sharpe_rolling30 − 0.5×drawdown − 0.3×turnover         │
│                                                                  │
│  Constraints: max 12%/stock · stop-loss −3% · circuit −12%      │
│  Episode: 252 days · Random start · 500,000 training steps       │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────┐
         ▼               ▼                   ▼
      ┌──────┐       ┌──────┐           ┌──────────┐
      │  PPO │       │  SAC │   TD3     │  DDPG    │
      │ A2C  │       └──────┘   A2C     └──────────┘
      └──────┘                      ↘
                              ┌─────────────┐
                              │  ENSEMBLE   │
                              │ (avg of 5)  │
                              └──────┬──────┘
                                     │
               ┌─────────────────────┴──────────────────┐
               ▼                                         ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│    STRESS TESTING        │          │   FEDERATED LEARNING      │
│  8 scenarios             │          │  4 sector clients         │
│  1,000 MC paths each     │          │  FedProx · μ=0.01        │
│  GAN-calibrated returns  │          │  50 rounds · 5 epochs     │
│  VaR · CVaR · Survival   │          │  DP-SGD ε=8.0, σ=1.1     │
│  Best: Flash Crash 76.4% │          │  Global Sharpe: 0.729     │
└──────────────────────────┘          └──────────────────────────┘
                         │                       │
                         └──────────┬────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │     FASTAPI BACKEND           │
                    │     50+ REST endpoints        │
                    │     Swagger UI at /docs       │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │   REACT 19 DASHBOARD          │
                    │   8 interactive tabs          │
                    │   TypeScript · Tailwind       │
                    │   Recharts · Framer Motion    │
                    └───────────────────────────────┘
```

---

## Results

### RL Algorithm Comparison — Test Period 2024–2025

| Algorithm | Sharpe | Sortino | Ann. Return | Volatility | Max Drawdown |
|-----------|:------:|:-------:|:-----------:|:----------:|:------------:|
| PPO | 0.7829 | 1.0721 | +15.22% | 12.76% | −17.06% |
| SAC | 0.7288 | 1.0089 | +14.31% | 12.58% | −16.42% |
| TD3 | 0.7480 | 1.0212 | +14.86% | 12.98% | −16.14% |
| A2C | 0.7520 | 1.0447 | +14.52% | 12.42% | −16.29% |
| DDPG | 0.8909 | 1.1279 | +21.27% | 17.84% | −21.37% |
| **Ensemble** | **0.8316** | **1.1086** | **+16.75%** | **13.76%** | **−17.80%** |
| NIFTY 50 (benchmark) | — | — | +0.65% | — | — |

### Live Portfolio — April 2025 to March 2026

| | Our Portfolio | NIFTY 50 | Fixed Deposit (7%) |
|-|:---:|:---:|:---:|
| Starting Capital | ₹10,00,000 | ₹10,00,000 | ₹10,00,000 |
| Final Value | **₹10,82,745** | ₹10,06,550 | ₹10,49,587 |
| Total Return | **+8.27%** | +0.65% | +4.96% |
| Sharpe Ratio | 0.2996 | — | — |
| Max Drawdown | −12.17% | — | — |

### Stress Testing — All 8 Scenarios

| Scenario | Mean Return | VaR 95% | CVaR 95% | Survival Rate |
|----------|:-----------:|:-------:|:--------:|:------------:|
| Normal | +15.74% | −15.89% | −21.32% | 34.4% |
| 2008 Financial Crisis | −25.62% | −49.31% | −53.65% | 1.2% |
| COVID-19 Crash | −12.35% | −29.37% | −32.32% | 21.1% |
| Flash Crash | −9.27% | −19.19% | −21.87% | **76.4%** |
| Dot-Com 2000 | −22.62% | −47.86% | −52.78% | 0.9% |
| India Bear 2015 | −13.24% | −38.00% | −43.28% | 4.0% |
| Rate Hike 2022 | −7.45% | −33.51% | −39.30% | 7.0% |
| Geo-Political Shock | −14.33% | −32.99% | −37.55% | 12.2% |

> 1,000 Monte Carlo paths per scenario · GAN-calibrated return generator · Fixed Ensemble weights

### Federated Learning

| Client | Stocks | Sharpe Change vs Isolated |
|--------|:------:|:------------------------:|
| Banking & Finance | 10 | **+0.298** |
| IT & Telecom | 6 | **+0.339** |
| Pharma & FMCG | 8 | +0.134 |
| Energy, Auto & Others | 20 | −0.138 |
| **Global Model** | **44** | **Sharpe = 0.729** |

---

## Dashboard — 8 Tabs

| Tab | What It Shows |
|-----|--------------|
| **Portfolio** | Sharpe, Sortino, Return, Volatility, Max Drawdown · Holdings table · Growth chart vs NIFTY 50 and Fixed Deposit |
| **RL Agent** | 6-algorithm comparison table · Training reward curves · Cumulative returns · Sector allocation |
| **Stress Testing** | 8 scenario cards (VaR, CVaR, Survival Rate) · Monte Carlo fan chart (1,000 paths) |
| **Federated Learning** | FedProx vs FedAvg convergence (50 rounds) · Privacy ε tracker · Per-client Sharpe improvement |
| **Sentiment** | Live FinBERT scores · Market mood indicator · News feed · Auto-refresh every 3 min |
| **Graph Visualization** | Force-directed stock network · 250 edges · 3 edge-type toggles · Click node for details |
| **Pipeline** | Animated 15-stage end-to-end data flow diagram with status indicators |
| **Future Prediction** | Black Bootstrap 1,000 forward paths · Median +9.3% · P(profit) 75.9% |

---

## Tech Stack

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.x | T-GAT, GAN, RL policy networks |
| Stable-Baselines3 | 2.8.0 | PPO, SAC, TD3, A2C, DDPG implementations |
| PyTorch Geometric | 2.7.0 | Graph construction, T-GAT layers |
| HuggingFace Transformers | 5.5.0 | FinBERT inference |
| Flower (flwr) | 1.29.0 | Federated learning framework |
| FastAPI | 0.135.2 | REST API, Swagger docs |
| Gymnasium | 0.29+ | RL environment interface |
| yfinance | 0.2.30+ | NIFTY 50 OHLCV data |
| SQLite | built-in | Sentiment cache (TTL 3 min) |

### Frontend
| Library | Version | Purpose |
|---------|---------|---------|
| React | 19 | UI component framework |
| TypeScript | 5.x | Type-safe API contracts |
| Vite | 5.x | Build tool, HMR dev server |
| Tailwind CSS | 3.x | Utility-first styling |
| Recharts | 2.x | Charts (line, area, bar, pie) |
| Framer Motion | 10.x | Page transitions, animations |

### Infrastructure
| Tool | Purpose |
|------|---------|
| Docker + docker-compose | Containerized deployment |
| pytest (246 tests, 245 pass) | Backend test suite |
| YAML config (`configs/base.yaml`) | All hyperparameters — no hardcoded values |

---

## Project Structure

```
fqn1/
├── configs/
│   └── base.yaml              ← Master hyperparameter config
├── src/
│   ├── api/
│   │   └── main.py            ← FastAPI: 50+ REST endpoints
│   ├── data/
│   │   ├── download.py        ← Yahoo Finance with retry/backoff
│   │   ├── features.py        ← 21 technical indicators
│   │   ├── live.py            ← Live price fetching
│   │   └── stocks.py          ← NIFTY 50 tickers, sectors, supply chain
│   ├── sentiment/
│   │   ├── finbert.py         ← ProsusAI/finbert inference + caching
│   │   ├── news_fetcher.py    ← Multi-source concurrent fetcher
│   │   └── indian_rss.py      ← Moneycontrol, Economic Times feeds
│   ├── graph/
│   │   └── builder.py         ← Sector + supply chain + correlation edges
│   ├── models/
│   │   └── tgat.py            ← Temporal Graph Attention Network (32-dim)
│   ├── rl/
│   │   ├── environment.py     ← Gymnasium env (obs=2420, action=44)
│   │   └── agent.py           ← PPO/SAC/TD3/A2C/DDPG + Ensemble
│   ├── gan/
│   │   └── stress.py          ← VaR, CVaR, Monte Carlo, 8 scenarios
│   ├── federated/
│   │   ├── server.py          ← FedAvg / FedProx aggregation
│   │   ├── client.py          ← Local training per sector client
│   │   └── privacy.py         ← DP-SGD: gradient clip + noise injection
│   └── utils/
│       ├── config.py          ← YAML config loader
│       ├── metrics.py         ← Sharpe, Sortino, Calmar, drawdown
│       └── logger.py          ← Logging setup
├── dashboard/
│   └── src/
│       ├── pages/             ← 8 tab components
│       │   ├── Portfolio.tsx
│       │   ├── RlAgent.tsx
│       │   ├── StressTesting.tsx
│       │   ├── Federated.tsx
│       │   ├── Sentiment.tsx
│       │   ├── GraphVisualization.tsx
│       │   ├── WorkflowViz.tsx
│       │   └── FuturePrediction.tsx
│       ├── components/        ← Charts, cards, layout
│       └── lib/               ← API client, formatters
├── tests/                     ← 246 tests (245 pass, 1 xfail)
├── data/
│   ├── all_close_prices.csv   ← Cached NIFTY 50 prices
│   ├── sentiment.db           ← SQLite sentiment cache
│   └── finbert_local/         ← Locally cached FinBERT weights
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites

| Software | Version |
|----------|---------|
| Python | 3.11.x |
| Node.js | 18+ |
| npm | 9+ |

### 1. Backend Setup

```bash
# Clone and enter project
cd fqn1

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

# Install PyTorch (CPU build — works without GPU)
pip install torch torchvision torchaudio

# Install all dependencies
pip install -r requirements.txt

# Run tests to verify everything works
python -m pytest tests/ -v --tb=short
# Expected: 245 passed, 1 xfailed
```

### 2. Start API Server

```bash
.\venv\Scripts\activate
python -m uvicorn src.api.main:app --reload --port 8001
```

Verify at:
- `http://localhost:8001/api/health` → `{"status":"ok","version":"4.0.0"}`
- `http://localhost:8001/docs` → Interactive Swagger UI

### 3. Start Dashboard

```bash
cd dashboard
npm install        # first time only
npm run dev
```

Open `http://localhost:5173` in your browser.

### 4. Docker (Alternative)

```bash
docker-compose up --build
```

- API: `http://localhost:8001`
- Dashboard: `http://localhost:3000`

---

## Key Design Decisions

**Why NIFTY 50?** Largest, most liquid Indian equities. Sufficient diversity across sectors while remaining tractable for a research platform.

**Why 44 stocks, not 50?** Six stocks excluded due to incomplete price history on Yahoo Finance for the full 2015–2025 period.

**Why Ensemble over best individual (DDPG)?** DDPG achieves highest Sharpe (0.8909) but worst drawdown (−21.37%). The Ensemble's averaging across five policies reduces sensitivity to any single algorithm's worst-case behaviour — directly analogous to portfolio diversification at the asset level, applied at the algorithm level.

**Why FinBERT over general BERT?** Domain specificity. "Bear", "loss", "correction", "short" have distinct financial meanings that general sentiment models handle incorrectly. FinBERT is fine-tuned on financial news and earnings call transcripts.

**Why FedProx over FedAvg?** Four sector clients have very different return distributions (non-IID). FedAvg lets clients drift toward sector-specific optima; FedProx's proximal term (μ=0.01) keeps local updates closer to the global model, producing faster and more stable convergence.

**Why ε=8.0 for differential privacy?** Practical balance. ε=1.0 requires much more gradient noise, visibly degrading model quality. ε=8.0 sits in the accepted range for real-world applications while still bounding adversarial inference.

---

## Limitations

- Backtested on Indian market data (2024–25 test window was a relatively flat market for NIFTY — results in a strong bull market may differ)
- Federated setup simulates four virtual clients on a single machine — real deployment requires distributed infrastructure
- FinBERT sentiment quality is uneven across sectors (English-language media coverage is denser for Finance/IT than for Energy/Metals)
- T-GAT and RL agents are trained on CPU; GPU-accelerated training would significantly reduce training time
- The system is a research platform, not a deployment-ready trading tool — no live order execution, authentication, or audit trail

---

## Citation

```bibtex
@mastersthesis{rawal2026finquant,
  title   = {FINQUANT-NEXUS: An AI-Powered Portfolio Optimization System for NIFTY 50},
  author  = {Praveen Pal Rawal},
  school  = {Rashtriya Raksha University},
  year    = {2026},
  type    = {MTech Dissertation},
  note    = {School of Information Technology, Artificial Intelligence and Cyber Security}
}
```

---

<div align="center">

**Supervised by Dr. Mayur Makwana**
*Assistant Professor, School of ITAICS, Rashtriya Raksha University, Gandhinagar*

</div>
