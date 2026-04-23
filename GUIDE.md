# FINQUANT-NEXUS v4 — Complete Running Guide

> Step-by-step guide to run backend, frontend, and tests.
> Last updated: 2026-04-21 (FinRL 0.3.7 fully installed — _FINRL_AVAILABLE=True, create_finrl_agent + run_finrl_baseline added)

.\venv\Scripts\activate
python -m uvicorn src.api.main:app --reload --port 8000

---

## Prerequisites

| Software | Version | Check Command |
|----------|---------|---------------|
| Python | 3.11.x | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | 2.x | `git --version` |
| CUDA (optional) | 12.1 | `nvidia-smi` |

---

## STEP 1: Python Virtual Environment Setup

```bash
cd c:\AAA\Personal\Clg\finquant\fqn1

# Create venv (once)
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies (includes finrl, alpaca-trade-api, exchange-calendars, stockstats)
pip install -r requirements.txt
```

**Verify:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import fastapi; print(f'FastAPI OK')"
python -c "import stable_baselines3; print(f'SB3 OK')"
python -c "from finrl.agents.stablebaselines3.models import DRLAgent; print('FinRL OK')"
```

---

## STEP 2: Run All Tests

```bash
cd c:\AAA\Personal\Clg\finquant\fqn1
.\venv\Scripts\activate
python -m pytest tests/ -v --tb=short
```

**Expected:** 246 collected, 245 passed, 1 xfail (CUDA device assertion)

**Individual phase tests:**
```bash
python -m pytest tests/test_phase0.py -v     # Phase 0: Config, seed, logger
python -m pytest tests/test_data.py -v       # Phase 1: Data pipeline
python -m pytest tests/test_features.py -v   # Phase 2: Feature engineering
python -m pytest tests/test_sentiment.py -v  # Phase 3: FinBERT sentiment
python -m pytest tests/test_graph.py -v      # Phase 4: Graph construction
python -m pytest tests/test_tgat.py -v       # Phase 5: T-GAT model
python -m pytest tests/test_env.py -v        # Phase 6: RL environment
python -m pytest tests/test_agent.py -v      # Phase 7: PPO/SAC/TD3/A2C/DDPG/Ensemble (30 tests)
python -m pytest tests/test_gan.py -v        # Phase 8-9: TimeGAN + Stress
python -m pytest tests/test_fl.py -v         # Phase 11: Federated Learning
python -m pytest tests/test_quantum.py -v    # Phase 12: QAOA Quantum
python -m pytest tests/test_api.py -v        # Phase 13: REST API
```

---

## STEP 3: Start Backend API (FastAPI)

**Terminal 1:**
```bash
cd c:\AAA\Personal\Clg\finquant\fqn1
.\venv\Scripts\activate
python -m uvicorn src.api.main:app --reload --port 8001
```

**Verify:**

| URL | Expected |
|-----|----------|
| http://localhost:8001/api/health | `{"status":"ok","version":"4.0.0"}` |
| http://localhost:8001/docs | Swagger UI |
| http://localhost:8001/api/rl-summary | JSON with `td3_sharpe`, `ensemble_sharpe` fields |

**Quick API tests:**
```bash
# Health
curl http://localhost:8001/api/health

# 6-algo RL summary (includes TD3, A2C, DDPG, Ensemble)
curl http://localhost:8001/api/rl-summary | python -m json.tool | head -30

# Live sentiment (takes ~15-20s — FinBERT runs)
curl http://localhost:8001/api/news-sentiment | python -m json.tool | head -20

# Stress test
curl -X POST http://localhost:8001/api/stress-test \
  -H "Content-Type: application/json" \
  -d "{\"n_stocks\": 5, \"n_simulations\": 500}"
```

---

## STEP 4: Start Frontend Dashboard (React)

**Terminal 2:**
```bash
cd c:\AAA\Personal\Clg\finquant\fqn1\dashboard
npm install    # first time only
npm run dev
```

**Open:** http://localhost:3000

---

## STEP 5: Dashboard Pages (6 Total)

| # | Page | URL | What to Check |
|---|------|-----|---------------|
| 1 | **Portfolio** | `/` (home) | Sharpe/Sortino/Return/Drawdown metrics, holdings table, performance chart |
| 2 | **RL Agent** | `/rl` | 6 algorithm buttons (PPO/SAC/TD3/A2C/DDPG/★ Ensemble), comparison table with 6 rows, 6-line reward + cumulative charts |
| 3 | **Stress Testing** | `/stress` | Click Generate → 4 scenario table (normal/2008/COVID/flash crash), Monte Carlo fan chart |
| 4 | **Federated** | `/fl` | 4 sector client cards, FedProx vs FedAvg convergence curves, privacy ε |
| 5 | **Sentiment** | `/sentiment` | LIVE badge + "Xs ago" timer, auto-refreshes every 3 min, trend chart builds over time, +N new badge on News tab |
| 6 | **Graph Viz** | `/graph` | Force-directed NIFTY 50 network, click node → details panel, edge type toggles |

### RL Agent Page — New Features
- **6 algorithm buttons** — PPO, SAC, TD3, A2C, DDPG, ★ Ensemble (green = recommended)
- **Comparison table** — 6 rows (one per algorithm), click row to select that algorithm
- **Reward chart** — 6 colored lines (Ensemble is thick green)
- **Cumulative returns** — 6 lines + Equal-Weight baseline (gray dashed)

### Sentiment Page — Real-Time Features
- **Auto-refresh**: News + FinBERT re-runs every 3 minutes automatically
- **LIVE badge**: Pulsing green dot + "Xs ago" timer in header
- **+N new badge**: Shows how many new headlines appeared since last refresh
- **Trend chart**: Session history of avg sentiment score (localStorage persistent — survives page reload)

---

## STEP 6: Production Build

```bash
cd c:\AAA\Personal\Clg\finquant\fqn1\dashboard
npm run build    # → dist/
npm run preview  # → http://localhost:4173
```

---

## STEP 7: Docker (Optional)

```bash
cd c:\AAA\Personal\Clg\finquant\fqn1
docker-compose up --build
curl http://localhost:8001/api/health
docker-compose down
```

---

## Quick Reference

### Start Everything
```bash
# Terminal 1 — Backend
cd c:\AAA\Personal\Clg\finquant\fqn1 && .\venv\Scripts\activate
python -m uvicorn src.api.main:app --reload --port 8001

# Terminal 2 — Frontend
cd c:\AAA\Personal\Clg\finquant\fqn1\dashboard
npm run dev
```

### Run Tests
```bash
cd c:\AAA\Personal\Clg\finquant\fqn1 && .\venv\Scripts\activate
python -m pytest tests/ -v
```

---

## Troubleshooting

### Backend Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | `cd fqn1` karo pehle |
| `Address already in use :8001` | `npx kill-port 8001` |
| `CUDA out of memory` | `configs/base.yaml` mein `device: cpu` karo |
| `finrl import error` | Run: `pip install alpaca-trade-api exchange-calendars stockstats "websockets>=13"` |
| `SSL/proxy error (yfinance)` | Mobile hotspot try karo |

### Frontend Issues

| Problem | Solution |
|---------|----------|
| Blank page at `/` | Backend port 8001 pe chal raha hai? Check karo |
| Charts show "Loading..." | API down hai — Terminal 1 check karo |
| Sentiment "Refreshing..." stuck | FinBERT 15-20s leta hai — normal hai |
| `npm run dev` fails | `npm install` pehle run karo |

### Test Issues

| Problem | Solution |
|---------|----------|
| `test_model_loads` fail | GPU available hai — CUDA pe load hota hai. Known xfail. |
| TD3/DDPG tests slow | Off-policy agents learning_starts wait karte hain — normal |
| FinBERT download fail | `data/finbert_local/` mein model hona chahiye |

---

## File Structure (Updated)

```
fqn1/
├── src/
│   ├── api/           → FastAPI (schemas.py, main.py) — all endpoints
│   ├── data/          → NIFTY 50 stocks, download, quality, features
│   ├── sentiment/     → FinBERT model, news_fetcher (with timeout)
│   ├── graph/         → 3 edge types, PyG Data objects
│   ├── models/        → T-GAT (multi-relational GAT + GRU)
│   ├── rl/            → PPO/SAC/TD3/A2C/DDPG + EnsembleAgent, Gym env
│   ├── gan/           → TimeGAN, stress testing, Monte Carlo
│   ├── federated/     → FedAvg/FedProx, DP-SGD, 4 sector clients
│   ├── quantum/       → QAOA, portfolio optimization
│   └── utils/         → Config, seed, logger, metrics
├── tests/             → 14 test files, 246 tests total
├── dashboard/         → React frontend (6 pages)
│   ├── src/pages/     → Portfolio, RlAgent, StressTesting,
│   │                    Federated, Sentiment, GraphVisualization
│   ├── src/components/→ Layout (Sidebar 6 items), UI, Charts
│   └── src/lib/       → API client (with RL 6-algo types), animations
├── configs/base.yaml  → All hyperparameters (incl. td3/a2c/ddpg/ensemble)
├── requirements.txt   → Dependencies (incl. finrl>=3.0, curl_cffi)
├── Dockerfile         → Container image (configs/ path fixed)
└── docker-compose.yml → API + PostgreSQL
```

---

## Summary Checklist

- [ ] `python -m pytest tests/ -v` → 245+ passed
- [ ] `uvicorn src.api.main:app --reload --port 8001` → running
- [ ] `http://localhost:8001/api/health` → `{"status":"ok"}`
- [ ] `http://localhost:8001/api/rl-summary` → has `td3_sharpe` + `ensemble_sharpe`
- [ ] `cd dashboard && npm run dev` → running on :3000
- [ ] Portfolio page loads at `/` (not `/portfolio`)
- [ ] RL Agent shows 6 algorithm buttons
- [ ] Sentiment shows LIVE badge + auto-refreshes
- [ ] Graph Viz nodes visible, click → details show
- [ ] `npm run build` → clean build, 0 TypeScript errors

**Sab green hai? FINQUANT-NEXUS v4 fully operational!**
