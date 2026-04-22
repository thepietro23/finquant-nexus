# FINQUANT-NEXUS v4 — React Dashboard

Interactive financial AI dashboard built with React + TypeScript + Vite.

## Tech Stack

- **React 19** + TypeScript + Vite
- **Tailwind CSS v4** — Light warm theme (terracotta `#C15F3C` primary)
- **Framer Motion** — Spring animations
- **Recharts** — Area, bar, line, pie charts
- **Lucide React** — Icons
- **TanStack React Query** — API state management
- **Zustand** — Global state

## Pages (6 Total)

| Page | Route | Description |
|------|-------|-------------|
| Portfolio | `/` | Core output — Sharpe, Sortino, holdings, performance |
| RL Agent | `/rl` | 5 algorithms (PPO/SAC/TD3/A2C/DDPG) + Ensemble comparison |
| Stress Testing | `/stress` | Monte Carlo + 4 crash scenarios (VaR, CVaR, Survival Rate) |
| Federated | `/fl` | FedAvg/FedProx convergence, sector clients, DP-SGD privacy |
| Sentiment | `/sentiment` | Live FinBERT analysis, auto-refresh every 3 min, trend chart |
| Graph Viz | `/graph` | Interactive NIFTY 50 stock network (sector + supply + correlation edges) |

## Running

```bash
npm install
npm run dev        # → http://localhost:3000
npm run build      # production build
npm run preview    # preview production build
```

Backend (FastAPI) must be running on port 8001:
```bash
cd ..
python -m uvicorn src.api.main:app --port 8001 --reload
```

## API Proxy

`vite.config.ts` proxies `/api/*` requests to `http://localhost:8001`.
