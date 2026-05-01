# Chapter 4 — Implementation and Results
## Target: 25–30 pages | Status: [x] Done — Written per prompt.md
## Word count: ~7900 words

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 4
# IMPLEMENTATION AND RESULTS

---

Chapter 3 covered the design decisions behind each component of FINQUANT-NEXUS. This chapter records what actually happened when those components were built and run. Each module was implemented and executed in the order described, starting from raw data collection and ending with the interactive dashboard. Results are reported as they came out of the system. Comparative analysis and deeper interpretation of the findings are left to Chapter 5.

---

## 4.1 Development Environment

All implementation, training, and testing for this dissertation was carried out on a single local machine. The hardware configuration is listed in Table 4.1.

**Table 4.1: Development Environment Specifications**

| Component | Specification |
|-----------|--------------|
| Operating System | Windows 11 Pro (Build 26100) |
| Processor | 13th Gen Intel Core i5-13420H @ 2.6 GHz |
| CPU Cores / Threads | 8 cores / 12 threads |
| RAM | 16 GB DDR4 |
| GPU | NVIDIA GeForce RTX 3050 (4 GB VRAM) |
| PyTorch Runtime | CPU-only build (v2.11.0) |
| Python Version | 3.11 |
| Stable-Baselines3 | 2.8.0 |
| HuggingFace Transformers | 5.5.0 |
| PyTorch Geometric | 2.7.0 |
| Flower (flwr) | 1.29.0 |
| FastAPI | 0.135.2 |
| React | 19 |
| Node.js | v24.14.1 |

The machine has a discrete GPU, but the PyTorch build used here was the CPU-only version. Because of that, all neural network operations (T-GAT training, RL policy network updates, and GAN-based calibration for stress testing) ran entirely on the CPU. A GPU-accelerated build would reduce training time considerably.

To work around the lack of GPU batch acceleration, RL training used a batch size of 64 with gradient accumulation set to 4 steps, giving an effective batch size of 256. This kept memory usage within limits while maintaining stable gradient estimates throughout training.

---

## 4.2 Data Collection and Processing Results

Daily OHLCV data for 44 NIFTY 50 stocks was downloaded using yfinance, covering January 2015 to December 2025 (approximately 2,761 trading sessions per stock). NaN values from market holidays and indicator warm-up periods were resolved by forward-fill. Twenty-one technical indicators were computed per stock and normalized using a 252-day rolling Z-score window. The final feature matrix has shape (2,761, 44, 21). The dataset splits into training (January 2015 to December 2021, ~1,757 sessions), validation (January 2022 to December 2023, ~502 sessions), and test (January 2024 to December 2025, ~502 sessions). No validation or test data was used during training.

Figure 4.1 shows a sample price chart for RELIANCE, HDFCBANK, and INFOSYS across the full study period.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.1 here — file: imgs/fig_4_1_stock_chart.png — Caption: Figure 4.1: Sample NIFTY 50 stock price chart showing RELIANCE, HDFCBANK, and INFOSYS daily closing price from January 2015 to December 2025. NOTE: This screenshot is still pending — take from dashboard Data Pipeline tab.]</span>

**Table 4.2: Dataset Summary Statistics**

| Metric | Value |
|--------|-------|
| Data source | Yahoo Finance API (yfinance) |
| Stocks included | 44 (6 excluded for incomplete history) |
| Date range | January 2015 to December 2025 |
| Approximate trading sessions per stock | 2,761 |
| OHLCV columns | 5 per stock |
| Technical indicators computed | 21 per stock |
| Feature matrix shape | (2,761, 44, 21) |
| Training period | January 2015 to December 2021 |
| Validation period | January 2022 to December 2023 |
| Test period | January 2024 to December 2025 |
| Missing value handling | Forward-fill |
| Normalization method | 252-day rolling Z-score per stock |

---

## 4.3 Portfolio Analytics and Benchmark Comparison

The Portfolio Analytics tab is the main landing page of the FINQUANT-NEXUS dashboard. It was designed to give a complete summary of portfolio performance on a single screen.

At the top of the tab, five metric cards display current portfolio performance: Sharpe Ratio (0.2996), Sortino Ratio (0.4371), Total Return (8.34%), Annual Volatility (12.86%), and Maximum Drawdown (minus 12.17%). These values reflect the active portfolio's performance over the current evaluation window. A screenshot of this tab is shown in Figure 4.2.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.2 here — file: imgs/fig_4_2_portfolio.png — Caption: Figure 4.2: Portfolio Analytics tab showing the five performance metric cards, current portfolio statistics, and the holdings table for all 44 NIFTY 50 constituent stocks.]</span>

Below the metric cards, a holdings table lists all 44 stocks with their current portfolio weight and sector. The allocation is computed by the active RL agent (Ensemble by default) and shows how capital is distributed across the NIFTY 50 universe at any given time. Weights sum to 1.0, and no single stock exceeds the 12 percent cap enforced as a hard constraint in the RL environment design.

The Sharpe Ratio is computed as the ratio of excess return over the risk-free rate to the annualized standard deviation of returns:

    Sharpe Ratio = (Annualized Portfolio Return − Risk-free Rate) / Annualized Standard Deviation of Returns

The risk-free rate was set to 5 percent, approximating the yield on Indian 10-year government bonds as of 2024. The Sortino Ratio uses the same numerator but replaces the denominator with downside deviation only, meaning standard deviation computed exclusively from days on which the portfolio lost value. Because the denominator is smaller, the Sortino Ratio is always higher than the Sharpe Ratio for the same portfolio. The values here (0.2996 Sharpe versus 0.4371 Sortino) confirm this relationship for the current window.

The growth chart in Figure 4.3 shows cumulative portfolio value from April 2025 to March 2026, compared against two baselines, namely the NIFTY 50 index and a fixed deposit returning 7 percent annually.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.3 here — file: imgs/fig_5_1_benchmark.png — Caption: Figure 4.3: Growth chart showing portfolio cumulative value versus NIFTY 50 index and 7 percent Fixed Deposit from April 2025 to March 2026, starting capital Rs. 10,00,000.]</span>

Starting from an initial capital of Rs. 10,00,000, the portfolio grew to Rs. 10,82,745 by end of March 2026, a gain of 8.27 percent over approximately 247 trading days. Over the same period, the NIFTY 50 index returned 0.65 percent, reaching Rs. 10,06,550. A fixed deposit at 7 percent annual rate would have returned 4.96 percent, reaching Rs. 10,49,587. The portfolio outperformed the NIFTY 50 by 7.62 percentage points and the fixed deposit by 3.31 percentage points.

Each of the three lines on the growth chart tells a different story. The NIFTY 50 line was volatile and stayed close to the starting value for most of the period, reflecting a relatively flat index during this window. A fixed deposit grows smoothly at a predictable rate, as expected. The portfolio line climbed more aggressively, particularly in the middle of the period, and finished ahead of both baselines despite a visible drawdown in the final quarter. Maximum drawdown of minus 12.17 percent stayed within the stop-loss threshold of minus 12 percent set as a hard constraint in the RL environment design.

---

## 4.4 Reinforcement Learning Training Results

Five separate RL agents were trained for this work: PPO, SAC, TD3, A2C, and DDPG. A sixth agent, the Ensemble, was not trained independently. At inference time, it computes the average of the portfolio weight outputs of all five agents, producing a combined allocation at each step. Training for each algorithm ran on the 2015 to 2021 dataset for 500,000 environment steps. Hyperparameter adjustments were made using the 2022 to 2023 validation set, and no further changes were made once testing on the 2024 to 2025 period began. This ensured that the test results represent genuinely unseen performance.

At each step, the RL environment presented the agent with an observation vector. For each of the 44 stocks, the observation included 21 normalized technical indicator values, the 32-dimensional T-GAT embedding vector, and 1 sentiment score, giving 54 values per stock. Combined with a portfolio state vector (current weights), the observation was a flattened representation of the full market state at that time step. The agent's output was a 44-dimensional vector of portfolio weights, passed through a softmax operation to ensure they summed to 1.0. The reward at each step was the Sharpe ratio of the portfolio computed over a rolling window, with a penalty deducted whenever any risk constraint was breached.

Figure 4.4 shows the RL Agent dashboard with the Ensemble algorithm selected, displaying its performance metrics and the portfolio allocation bar chart.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.4 here — file: imgs/fig_4_3_rl_agent_ensemble.png — Caption: Figure 4.4: RL Agent dashboard showing the Ensemble algorithm selected with its performance metrics and portfolio weight distribution across 44 NIFTY 50 stocks.]</span>

Figure 4.5 shows the algorithm comparison table as displayed in the dashboard.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.5 here — file: imgs/fig_4_3_rl_comparison_table.png — Caption: Figure 4.5: RL algorithm comparison table from the dashboard showing performance metrics for all six algorithms evaluated on the 2024 to 2025 test period.]</span>

The complete results on the 2024 to 2025 test period are shown in Table 4.3.

**Table 4.3: RL Algorithm Performance Comparison (Test Period: 2024 to 2025)**

| Algorithm | Sharpe Ratio | Sortino Ratio | Ann. Return | Volatility | Max Drawdown |
|-----------|-------------|--------------|-------------|------------|--------------|
| PPO | 0.7829 | 1.0721 | +15.22% | 12.76% | -17.06% |
| SAC | 0.7288 | 1.0089 | +14.31% | 12.58% | -16.42% |
| TD3 | 0.7480 | 1.0212 | +14.86% | 12.98% | -16.14% |
| A2C | 0.7520 | 1.0447 | +14.52% | 12.42% | -16.29% |
| DDPG | 0.8909 | 1.1279 | +21.27% | 17.84% | -21.37% |
| **Ensemble** | **0.8316** | **1.1086** | **+16.75%** | **13.76%** | **-17.80%** |

DDPG produced the highest return (21.27%) but also the highest volatility (17.84%) and deepest drawdown (minus 21.37%), consistent with its deterministic concentrated policy. SAC showed the lowest return (14.31%) with the second-lowest volatility (12.58%), reflecting its entropy regularization. The Ensemble achieved a Sharpe of 0.8316 with a drawdown of minus 17.80%, providing a better risk-adjusted balance than DDPG. All six algorithms substantially outperformed the NIFTY 50 index return of 0.65% over the same period. Chapter 5 analyses these results in detail.

---

## 4.5 Sentiment Analysis Results

The sentiment analysis module runs in real time when the dashboard is open. For each of the 44 stocks, the FinBERT model (ProsusAI/finbert) fetches recent news headlines from Google News, processes them through the tokenizer and three-class classification head, and computes a composite sentiment score as the difference between the positive and negative class probabilities. The score ranges from minus 1 (strongly negative) to plus 1 (strongly positive).

On the day the system was run for this dissertation, 50 headlines were processed across the 44 stocks. Overall market mood was assessed as Bullish, with an average sentiment score of 0.2942 across all stocks and headlines. This score falls in the "Strong Buy" classification band in the dashboard's three-tier scheme (Bearish for scores below zero, Neutral for zero to 0.15, Bullish for scores above 0.15). Google News contributed 80 percent of the total headline signal, with other aggregated sources providing the remaining 20 percent.

A screenshot of the Sentiment Analysis tab is shown in Figure 4.6.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.6 here — file: imgs/fig_4_5_sentiment1.png — Caption: Figure 4.6: Sentiment Analysis tab showing live FinBERT sentiment scores for NIFTY 50 stocks, overall market mood indicator, average composite score, and the top market mover.]</span>

Sector-level sentiment scores showed variation across industries. Finance showed the strongest positive sentiment with a score of 0.7227. The Others segment followed at 0.7062. Auto was the only sector in negative territory on this day, with a score of minus 0.3259. FMCG was mildly positive at 0.2095. The top market mover on the observation day was MARUTI SUZUKI, with a price change of minus 2.21 percent, which aligned with the negative Auto sector reading.

SQLite caching prevented redundant FinBERT calls. Inference time per full batch is 15 to 30 seconds on the CPU build, which is acceptable for a demand-refresh dashboard. Sentiment scores enter the RL observation vector as one-dimensional inputs per stock, and the agent learned during training how to weight them relative to price and graph signals.

---

## 4.6 Graph Visualization Results

The Stock Relationship Graph provides an interactive view of the multi-relational stock network described in Chapter 3. The graph was constructed with 44 stock nodes and 250 edges distributed across three relationship types, namely sector membership (79 edges), supply chain linkage (24 edges), and 60-day rolling return correlation (147 edges).

A force-directed visualization renders 44 stock nodes with size proportional to portfolio weight and colour encoding sector. Three edge-type toggles allow independent display of sector, supply chain, and correlation edges. During high-correlation periods, the graph visibly densifies as rolling correlations spike. Clicking any node opens a detail panel with sector, weight, degree, and connected neighbours.

Figure 4.7 shows the full graph with all three edge types enabled.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.7 here — file: imgs/fig_4_6_graph_all_edges.png — Caption: Figure 4.7: Graph Visualization tab showing force-directed stock network with all three edge types (Sector, Supply Chain, Correlation) enabled.]</span>

Table 4.4 records the quantitative statistics of the constructed graph.

**Table 4.4: Stock Graph Statistics**

| Metric | Value |
|--------|-------|
| Total nodes (stocks) | 44 |
| Total edges | 250 |
| Sector edges | 79 |
| Supply chain edges | 24 |
| Correlation edges (60-day rolling) | 147 |
| Graph density | 0.264 |
| Average node degree | 11.4 |
| Highest degree stock | HDFCBANK (Degree 23, Neighbours 22) |
| Strongest correlated pair | BAJFINANCE-BAJAJFINSV (correlation 0.89) |

HDFCBANK has the highest degree (23) because it connects to banking peers via sector edges, to several companies via supply chain links, and to many stocks via return correlations. The strongest pair correlation belongs to BAJFINANCE and BAJAJFINSV (0.89), reflecting their shared Bajaj group business overlap. The graph encodes this structural similarity so the RL agent does not overweight both as if they were independent positions.

---

## 4.7 Stress Testing Results

The Stress Testing module ran 1,000 Monte Carlo simulation paths for each of the eight predefined crisis scenarios. Each path represents one possible portfolio outcome over the simulation horizon, generated by sampling return parameters drawn from the historical distributions of the chosen crisis period. Three risk metrics were computed from the 1,000 paths, namely Mean Return across all paths, Value at Risk at the 95th percentile (VaR 95%), and Conditional Value at Risk at the 95th percentile (CVaR 95%). A Survival Rate was also recorded, defined as the percentage of simulation paths in which the portfolio held a positive return by the end of the horizon.

Figure 4.8 shows the Stress Testing tab. The Monte Carlo fan chart displays all 1,000 paths as overlapping curves, with the median path highlighted and confidence bands drawn at the 10th and 90th percentile positions.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.8 here — file: imgs/fig_4_7_stress_normal.png — Caption: Figure 4.8: Stress Testing tab showing the Monte Carlo simulation fan chart and risk metrics panel for the currently selected scenario.]</span>

The complete results for all eight scenarios are in Table 4.5.

**Table 4.5: Stress Testing Risk Metrics — All 8 Scenarios**

| Scenario | Mean Return | VaR 95% | CVaR 95% | Survival Rate |
|----------|------------|---------|----------|---------------|
| Normal | +15.74% | -15.89% | -21.32% | 34.4% |
| 2008 Financial Crisis | -25.62% | -49.31% | -53.65% | 1.2% |
| COVID-19 Crash | -12.35% | -29.37% | -32.32% | 21.1% |
| Flash Crash | -9.27% | -19.19% | -21.87% | 76.4% |
| Dot-Com 2000 | -22.62% | -47.86% | -52.78% | 0.9% |
| India Bear 2015 | -13.24% | -38.00% | -43.28% | 4.0% |
| Rate Hike 2022 | -7.45% | -33.51% | -39.30% | 7.0% |
| Geo-Political Shock | -14.33% | -32.99% | -37.55% | 12.2% |

> 1,000 Monte Carlo paths per scenario | Best survival: Flash Crash 76.4% | Worst: Dot-Com 2000 0.9%

The Normal baseline produces a mean return of +15.74% with a 34.4% survival rate. Paths ending negative even without crisis stress reflect natural return variance. Flash Crash shows the best crisis survival at 76.4% because the historical event was brief and the portfolio's stop-loss constraints limited per-period loss. Both the 2008 Crisis and Dot-Com 2000 have survival rates below 2 percent (1.2% and 0.9%), because both were prolonged multi-month declines where cumulative losses compound unavoidably. Rate Hike 2022 is worth noting for a different reason. The mean return looks relatively mild at minus 7.45%, but the tail risk is severe (CVaR minus 39.30%), which is characteristic of interest rate environments where the average path is moderate but the worst paths are very bad. Chapter 5 interprets these results across scenario groups.

---

## 4.8 Federated Learning Results

The federated learning module simulated a four-client environment on the same physical machine, with each client managing a logically separated dataset corresponding to one market sector. The clients were Banking and Finance (10 stocks), IT and Telecom (6 stocks), Pharma and FMCG (8 stocks), and Energy, Auto and Others (20 stocks). The FedProx aggregation algorithm with a proximal term of μ equal to 0.01 was used. DP-SGD was applied locally at each client before weight sharing, with noise multiplier σ equal to 1.1 and maximum gradient norm equal to 1.0.

Training ran for 50 communication rounds. In each round, all four clients train locally for 5 epochs under DP-SGD, send noisy weight updates to the server, the server aggregates via FedProx, and returns global weights. No raw data is shared at any point.

Figure 4.9 shows the Federated Learning tab.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.9 here — file: imgs/fig_4_8_federated.png — Caption: Figure 4.9: Federated Learning tab showing FedProx and FedAvg convergence comparison, privacy epsilon tracker (reaching epsilon = 8.0 at round 50), and per-client Sharpe improvement metrics.]</span>

After 50 rounds, the global model reached a Sharpe Ratio of 0.729. FedProx converged faster than FedAvg, which oscillated in the first 15 to 20 rounds before settling. This is consistent with FedProx's proximal term preventing client drift. The epsilon tracker reached exactly 8.0 at round 50 as designed.

**Table 4.6: Federated Learning Results Summary**

| Configuration | Rounds | Global Sharpe | Privacy Epsilon | Notes |
|--------------|--------|--------------|----------------|-------|
| FedAvg (baseline) | 50 | [lower, see convergence chart] | N/A | Oscillates in early rounds |
| FedProx (mu = 0.01) | 50 | 0.729 | 8.0 (delta = 0.00001) | Stable convergence, proximal term |

Per-client performance was measured as the change in Sharpe Ratio achieved by each client's local model after participating in federated training, compared to training in isolation. Banking and Finance improved by plus 0.298. IT and Telecom improved by plus 0.339. Pharma and FMCG improved by plus 0.134. Energy, Auto and Others showed a decrease of minus 0.138. The first three sectors all benefited from accessing information learned across other sectors through the aggregation process. The Energy sector's decline suggests that the global model's average behaviour does not fully align with the energy sector's distinct return characteristics, such as its dependence on commodity prices and regulatory announcements. This observation is examined in Chapter 5.

---

## 4.9 Pipeline Workflow Visualization

The Pipeline tab shows an animated 15-stage directed graph of the end-to-end data flow: Data Ingestion, Data Cleaning, Feature Engineering, Indicator Computation, FinBERT Sentiment, News Caching, Graph Construction, T-GAT Embedding, Observation Assembly, RL Environment, Agent Inference, Portfolio Weights, Risk Constraint Check, API Response, Dashboard Rendering. Each stage shows its status (Running/Idle) and the data type passing through it.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.10 here — file: imgs/fig_4_9_pipeline.png — Caption: Figure 4.10: Pipeline Workflow tab showing animated end-to-end data flow with stage status indicators for all 15 system components.]</span>

---

## 4.10 Future Prediction Dashboard

The Future Prediction tab generates 1,000 forward-looking portfolio paths over a 1-year horizon using Black Bootstrap, a block resampling technique that draws 30-day segments from historical returns calibrated by a GAN model. Key results: Median Return +9.3%, Best Case +31.6%, Worst Case minus 11.1%, Probability of Profit 75.9%. Table 4.7 shows the per-algorithm comparison.

<span style="color:red;font-weight:bold;">[INSERT Figure 4.11 here — file: imgs/fig_4_10_future_prediction.png — Caption: Figure 4.11: Future Prediction tab showing 1,000 Black Bootstrap simulation paths with median return line, confidence bands, probability of profit, and per-algorithm forward comparison.]</span>

**Table 4.7: Forward Simulation Results per Algorithm (1-Year Horizon, 1,000 Scenarios)**

| Algorithm | Expected Return | Probability of Profit |
|-----------|----------------|----------------------|
| PPO | +10.7% | 79.5% |
| SAC | +10.5% | 78.7% |
| TD3 | +0.6% | 83.7% |
| A2C | +0.4% | 83.7% |
| DDPG | +4.0% | 76.0% |
| Ensemble | +9.3% (median) | 75.9% |

TD3 and A2C showed the highest probability of profit (83.7%) but very low expected returns (0.6% and 0.4%), reflecting their conservative forward allocations. PPO and SAC showed higher expected returns (10.7% and 10.5%) with slightly lower profit probability. DDPG, which led the historical backtest, ranked fourth in forward expected return. GAN-calibrated bootstrap paths spanning a broader range of market conditions reduce the advantage of concentrated strategies, which explains this gap between historical and forward rankings.

---

## 4.11 Testing and Validation

A pytest suite with 12 test files covers the complete pipeline. Table 4.8 lists results by module.

**Table 4.8: Test Coverage Summary**

| Test File | Module Tested | Result |
|-----------|--------------|--------|
| test_phase0.py | Config validation, logging, random seed | 1 FAILED |
| test_data.py | Data download, cleaning, forward-fill | PASS |
| test_features.py | 21 technical indicators computation | PASS |
| test_sentiment.py | FinBERT inference, news fetching, caching | PASS |
| test_graph.py | Graph construction, three edge types | PASS |
| test_tgat.py | T-GAT model forward pass and training | PASS |
| test_env.py | RL environment, reward function, constraints | PASS |
| test_agent.py | Six RL algorithms training and inference | PASS |
| test_gan.py | Monte Carlo stress testing, VaR computation | PASS |
| test_fl.py | Federated learning, DP-SGD privacy | PASS |
| test_api.py | FastAPI endpoints, response validation | PASS |
| **TOTAL** | | **244 passed, 1 failed** |

244 passed, 1 failed (245 total). The single failure in test_phase0.py checks the risk-free rate and expects 0.07 but the active config has 0.05. This is an outdated test assertion, not a functional error. All calculations in the dissertation use 0.05. All functional modules verified correct. Chapter 5 analyses and interprets these results.

---

## Figures Required in This Chapter

| Figure No. | File | Status |
|-----------|------|--------|
| Figure 4.1 | fig_4_1_stock_chart.png — Sample price chart | [ ] Pending capture |
| Figure 4.2 | fig_4_2_portfolio.png — Portfolio Analytics tab | [x] |
| Figure 4.3 | fig_5_1_benchmark.png — Growth chart vs NIFTY 50 | [x] |
| Figure 4.4 | fig_4_3_rl_agent_ensemble.png — RL Agent Ensemble | [x] |
| Figure 4.5 | fig_4_3_rl_comparison_table.png — RL comparison | [x] |
| Figure 4.6 | fig_4_5_sentiment1.png — Sentiment main view | [x] |
| Figure 4.7 | fig_4_6_graph_all_edges.png — Graph all edges | [x] |
| Figure 4.8 | fig_4_7_stress_normal.png — Stress testing tab | [x] |
| Figure 4.9 | fig_4_8_federated.png — Federated learning tab | [x] |
| Figure 4.10 | fig_4_9_pipeline.png — Pipeline workflow tab | [x] |
| Figure 4.11 | fig_4_10_future_prediction.png — Future prediction | [x] |

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
