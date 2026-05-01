# Appendices
## Target: 8–10 pages | Status: [x] Done
## Reference: DISSERTATION_FORMATTING.md
> Note: Appendices continue the Arabic page numbering from the References section.

---

# APPENDICES

---

## Appendix A — System Architecture Diagram

The complete FINQUANT-NEXUS system architecture is shown in Figure A.1. The diagram illustrates all major components and the direction of data flow from input sources to the React dashboard output.

<span style="color:red;font-weight:bold;">[INSERT Figure A.1 here — file: imgs/fig_3_1_architecture.png — Caption: Figure A.1: Complete FINQUANT-NEXUS System Architecture showing data flow from Yahoo Finance and news sources through the ML pipeline to the React dashboard. High-resolution version rendered from DIAGRAMS_MERMAID.md using Mermaid.live.]</span>

The architecture diagram covers the following components in sequence.

- **Data Layer:** Yahoo Finance API (price data for 44 stocks), Google News API, Yahoo Finance News, Indian RSS feeds
- **Processing Layer:** Data cleaning, forward-fill, 21 technical indicator computation, 252-day rolling Z-score normalization
- **Sentiment Layer:** FinBERT tokenization, three-class sentiment classification, SQLite caching with 180-second TTL
- **Graph Layer:** Multi-relational graph construction (sector, supply chain, correlation edges), Temporal Graph Attention Network producing 32-dimensional stock embeddings
- **RL Layer:** Gymnasium PortfolioEnv, six RL agents (PPO, SAC, TD3, A2C, DDPG, Ensemble), risk constraint enforcement
- **Risk Layer:** Monte Carlo stress testing across eight scenarios, Black Bootstrap forward simulation
- **FL Layer:** FedProx server, four sector clients, DP-SGD privacy mechanism
- **Output Layer:** FastAPI REST backend, React 19 dashboard with eight interactive tabs

---

## Appendix B — Configuration File (configs/base.yaml)

The following listing shows the complete hyperparameter configuration used for all experiments in this dissertation. All parameter values referenced in Chapters 3 and 4 can be verified against this file. No values were hardcoded in the source code; all experiments used this configuration file as the single source of truth for hyperparameters.

```yaml
# === FINQUANT-NEXUS v4 Master Config ===
# All hyperparameters here. No hardcoded values in code.

seed: 42
device: 'cuda'             # configured target; falls back to CPU on the CPU-only PyTorch build used in this work
fp16: true

# --- Data ---
data:
  stocks: 'nifty50'
  start_date: '2015-01-01'
  end_date: '2025-12-31'
  train_end: '2021-12-31'
  val_end: '2023-12-31'
  risk_free_rate: 0.05        # 5% (Indian 10-year government bond yield)
  transaction_cost: 0.001     # 0.1% STT + brokerage
  slippage: 0.0005            # 0.05%
  min_trading_days: 1000      # Minimum days for a stock to be included
  max_nan_pct: 0.05           # Max 5% NaN allowed
  trading_days_per_year: 248

# --- Feature Engineering ---
features:
  normalize: 'zscore'
  rolling_window: 252         # 1 year rolling for normalization
  clip_range: 5.0             # Clip z-scores to [-5, +5]
  min_periods: 60             # Minimum periods for rolling stats
  indicators:
    - rsi
    - macd
    - macd_signal
    - macd_hist
    - bb_upper
    - bb_mid
    - bb_lower
    - sma_20
    - sma_50
    - ema_12
    - ema_26
    - atr
    - stoch_k
    - stoch_d
    - volume_sma
    - volume_ratio
    - return_1d
    - return_5d
    - return_20d
    - volatility_20d
    - volatility_60d

# --- Portfolio ---
portfolio:
  starting_capital: 1000000   # Rs. 10,00,000 initial capital
  eval_days: 248              # 1-year evaluation window (trading days)
  sparkline_days: 60          # days shown in price sparkline chart
  train_split: 0.70           # train/validation split ratio

# --- Sentiment ---
sentiment:
  model: 'ProsusAI/finbert'
  fine_tune_epochs: 3
  fine_tune_lr: 0.00002
  fine_tune_batch_size: 16
  max_length: 128
  decay_factor: 0.95          # Decay for days without news
  cache_db: 'data/sentiment.db'
  news_cache_ttl: 180         # seconds before re-fetching live news
  sentiment_threshold: 0.1   # score > threshold positive; < -threshold negative
  market_mood_threshold: 0.08 # threshold for Bullish/Bearish market classification
  sensitivity: 2.0            # how much sentiment shifts portfolio weights
  max_news_tickers: 20        # max stocks to fetch news for (speed limit)

# --- GNN ---
gnn:
  hidden_dim: 64
  output_dim: 32
  num_layers: 2
  num_heads: 4
  dropout: 0.1
  correlation_threshold: 0.6
  correlation_window: 60      # 60-day rolling correlation
  neighbor_sample: 15         # NeighborSampler max neighbors per hop

# --- RL ---
rl:
  algorithm: 'PPO'            # Primary: PPO. Comparison: SAC
  lr: 0.0003
  gamma: 0.99
  batch_size: 64
  n_steps: 2048
  n_epochs: 10
  clip_range: 0.2
  total_timesteps: 500000
  max_position: 0.12          # 12% max per stock
  stop_loss: -0.03            # -3% per stock
  max_drawdown: -0.12         # -12% circuit breaker
  episode_length: 252         # 1 year trading days
  reward:
    sharpe_weight: 1.0
    drawdown_penalty: 0.4
    turnover_penalty: 0.02

  sac:
    lr: 0.0003
    buffer_size: 100000
    batch_size: 256
    tau: 0.005
    ent_coef: 'auto'

  td3:
    lr: 0.0003
    buffer_size: 100000
    batch_size: 256
    tau: 0.005
    policy_delay: 2
    target_policy_noise: 0.2

  a2c:
    lr: 0.0007
    n_steps: 5
    ent_coef: 0.01

  ddpg:
    lr: 0.001
    buffer_size: 100000
    batch_size: 256
    tau: 0.005

  ensemble:
    rebalance_window: 63      # ~3 months
    top_k: 3                  # average top-3 models by recent Sharpe

# --- GAN ---
gan:
  seq_length: 128
  latent_dim: 64
  hidden_dim: 128
  num_layers: 3
  epochs: 500
  lr: 0.0005
  batch_size: 32
  grad_accumulation: 4        # Effective batch = 32 * 4 = 128

# --- Stress Testing ---
stress:
  n_scenarios: 1000
  var_confidence: [0.95, 0.99]
  monte_carlo_paths: 10000
  crash_types:
    - normal
    - crash_2008
    - crash_covid
    - flash_crash

# --- Federated Learning ---
fl:
  num_clients: 4
  rounds: 50
  local_epochs: 5
  strategy: 'FedProx'
  fedprox_mu: 0.01
  dp_epsilon: 8.0
  dp_delta: 0.00001
  dp_max_grad_norm: 1.0

# --- API ---
api:
  host: '0.0.0.0'
  port: 8000
  cors_origins:
    - 'http://localhost:3000'

# --- Logging ---
logging:
  level: 'INFO'
  log_dir: 'logs'
```

> Note: All values used in Chapter 3 and Chapter 4 are directly traceable to the sections above.

---

## Appendix C — REST API Endpoint Reference

Table C.1 lists the primary REST API endpoints exposed by the FINQUANT-NEXUS FastAPI backend. All endpoints return JSON. The full interactive documentation is available at http://localhost:8000/docs when the system is running.

**Table C.1: REST API Endpoint Reference**

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | GET | /api/health | System health check, uptime, component status |
| 2 | GET | /api/portfolio-summary | Portfolio metrics: Sharpe, Sortino, Return, Drawdown, weights |
| 3 | GET | /api/rl-summary | All six RL algorithm results and comparison metrics |
| 4 | GET | /api/rl-summary/{algorithm} | Single algorithm detailed results |
| 5 | GET | /api/news-sentiment | Live FinBERT sentiment scores for all 44 stocks |
| 6 | GET | /api/stress-test | Monte Carlo stress test results across all eight scenarios |
| 7 | GET | /api/stress-test/{scenario} | Single scenario risk metrics (VaR, CVaR, survival rate) |
| 8 | GET | /api/fl-summary | Federated learning results: convergence, privacy, client fairness |
| 9 | GET | /api/gnn-summary | Graph neural network statistics: nodes, edges, embeddings |
| 10 | GET | /api/portfolio-growth | Historical portfolio value versus benchmark time series |
| 11 | GET | /api/holdings | Current portfolio holdings with weight and sector per stock |
| 12 | GET | /api/sentiment-trend | Historical sentiment scores over time per stock |
| 13 | GET | /api/graph-data | Graph nodes, edges, and coordinates for frontend visualization |
| 14 | GET | /api/workflow-status | Pipeline stage statuses for all 15 system stages |
| 15 | GET | /api/future-prediction | Black Bootstrap forward simulation results (1000 scenarios) |
| 16 | GET | /api/sector-performance | Aggregated performance metrics by market sector |
| 17 | GET | /api/stock-detail/{ticker} | Individual stock price history, indicators, and sentiment |
| 18 | GET | /api/stocks | Full list of all 44 NIFTY 50 constituent stocks with sector labels |
| 19 | GET | /api/config | Active system configuration values from configs/base.yaml |
| 20 | POST | /api/sentiment | Score a single news headline on demand with FinBERT |
| 21 | POST | /api/stress-test | Run Monte Carlo stress test with custom scenario parameters |
| 22 | GET | /api/refresh-data | Trigger fresh OHLCV price download from Yahoo Finance |
| 23 | POST | /api/cache/refresh | Force-invalidate and rebuild the news sentiment SQLite cache |
| 24 | GET | /docs | Swagger UI interactive API documentation |

All endpoints return HTTP 200 with JSON on success. Standard HTTP error codes (400, 404, 500) are returned on failure with a JSON error message body. GET and POST endpoints share the same error handling convention. CORS is configured for http://localhost:3000 to allow the React dashboard to communicate with the backend running on port 8000. Endpoints marked POST accept a JSON request body, and parameter schemas are defined in the Swagger documentation at /docs.

---

## Appendix D — Test Results Summary

The pytest test suite for FINQUANT-NEXUS was run on the local development machine described in Chapter 4, Section 4.1. Table D.1 shows the results by test file.

**Table D.1: Test Suite Results by Module**

| # | Test File | Module Tested | Result |
|---|-----------|--------------|--------|
| 1 | test_phase0.py | Config validation, logging setup, random seed | PASS |
| 2 | test_data.py | Data download, cleaning, forward-fill | PASS |
| 3 | test_features.py | 21 technical indicators computation | PASS |
| 4 | test_sentiment.py | FinBERT inference, news fetching, SQLite caching | PASS |
| 5 | test_graph.py | Graph construction, three edge types, density | PASS |
| 6 | test_tgat.py | T-GAT model forward pass and training loop | PASS |
| 7 | test_env.py | RL environment, reward function, risk constraints | PASS |
| 8 | test_agent.py | Six RL algorithms: training and inference | PASS |
| 9 | test_gan.py | Monte Carlo stress testing, VaR computation | PASS |
| 10 | test_fl.py | Federated learning, FedProx, DP-SGD privacy | PASS |
| 11 | test_api.py | FastAPI endpoints, response schema validation | PASS |
| | **TOTAL** | | **245 passed, 0 failed (245 total)** |

> Two assertions in test_phase0.py initially carried outdated values (0.07 for risk-free rate, 0.20 for max position) from an earlier configuration. After updating these to 0.05 and 0.12 respectively to match the active configs/base.yaml, the full suite passed cleanly. All calculations in the dissertation use 0.05 for the risk-free rate and 0.12 for the maximum single-stock position.

**Pytest run command used:**
```
cd fqn1
pytest tests/ -v --tb=short
```

---

## Appendix E — List of Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| A2C | Advantage Actor-Critic |
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| ATR | Average True Range |
| CVaR | Conditional Value at Risk |
| DDPG | Deep Deterministic Policy Gradient |
| DP-SGD | Differentially Private Stochastic Gradient Descent |
| DRL | Deep Reinforcement Learning |
| EMA | Exponential Moving Average |
| FedAvg | Federated Averaging |
| FedProx | Federated Proximal Optimisation |
| FinBERT | Financial Bidirectional Encoder Representations from Transformers |
| FL | Federated Learning |
| FMCG | Fast-Moving Consumer Goods |
| GAT | Graph Attention Network |
| GNN | Graph Neural Network |
| GRU | Gated Recurrent Unit |
| MACD | Moving Average Convergence Divergence |
| ML | Machine Learning |
| MPT | Modern Portfolio Theory |
| NLP | Natural Language Processing |
| NSE | National Stock Exchange of India |
| NIFTY | National Fifty |
| OBV | On-Balance Volume |
| OHLCV | Open, High, Low, Close, Volume |
| PPO | Proximal Policy Optimisation |
| REST | Representational State Transfer |
| RL | Reinforcement Learning |
| RSI | Relative Strength Index |
| RRU | Rashtriya Raksha University |
| SAC | Soft Actor-Critic |
| SEBI | Securities and Exchange Board of India |
| SMA | Simple Moving Average |
| T-GAT | Temporal Graph Attention Network |
| TD3 | Twin Delayed Deep Deterministic Policy Gradient |
| VaR | Value at Risk |
| VWAP | Volume Weighted Average Price |

---

*Reference: DISSERTATION_FORMATTING.md*
*Last updated: 2026-04-30*
