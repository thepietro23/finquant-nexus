# Chapter 3 — System Design and Methodology
## Target: 28–35 pages | Status: [x] Done
## Word count: ~9600 words (~32 pages at 300 words/page)

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 3
# SYSTEM DESIGN AND METHODOLOGY

---

FINQUANT-NEXUS is built around thirteen distinct components. Each is described in its own section below, following the natural data flow of the system. We start from raw market data collection, move through feature engineering, sentiment extraction, and graph modelling, then enter reinforcement learning training and evaluation, stress testing, and federated learning, and end with the API and dashboard that make all results accessible.

---

## 3.1 Overall System Architecture

FINQUANT-NEXUS is a pipeline system where data moves through seven processing stages before reaching the user interface. Figure 3.1 shows the overall architecture.

Data acquisition is the first stage. Historical daily price and volume data for forty-four NIFTY 50 stocks is downloaded from Yahoo Finance using the yfinance Python library. [33] Financial news from three sources (Google News RSS, Yahoo Finance News API, and Indian RSS feeds from Moneycontrol and Economic Times) is fetched concurrently by a thread-safe news fetcher.

Processing comes next. On the price side, raw data is cleaned, adjusted for corporate actions, and passed through a feature engineering module that computes twenty-one technical indicators. On the news side, headlines are deduplicated, tokenized, and processed by a locally cached FinBERT model to produce a sentiment score per stock per day.

Graph construction and embedding is the third stage. The forty-four stocks are modelled as nodes in a multi-relational graph with three types of edges. This graph is processed by the Temporal Graph Attention Network (T-GAT), which produces a thirty-two dimensional embedding vector for each stock at each time step.

Stage four is the reinforcement learning environment. The RL observation space combines the twenty-one technical indicators, the thirty-two dimensional T-GAT embeddings, the FinBERT sentiment scores, and the current portfolio weights. Five RL algorithms (PPO, SAC, TD3, A2C, DDPG) are trained in a Gymnasium-compatible environment that simulates portfolio allocation with realistic constraints. Their outputs are averaged by a meta-level Ensemble agent.

Risk evaluation is the fifth stage. A Monte Carlo stress testing module takes the portfolio weights produced by the Ensemble agent and runs one thousand simulation paths under eight historical crisis scenarios, computing VaR, CVaR, and survival rates.

Federated learning is the sixth stage. The T-GAT model is also trained in a federated setting where four sector clients train locally under differential privacy constraints, and a FedProx server aggregates their updates across fifty communication rounds.

The seventh stage is output. A FastAPI backend exposes all results through more than fifty REST endpoints. A React dashboard with eight interactive pages presents portfolio analytics, RL agent comparisons, sentiment data, graph visualisations, stress test results, federated learning convergence, the data pipeline view, and a forward simulation.

The technology stack is Python 3.11 on the backend (PyTorch, Stable-Baselines3, PyTorch Geometric, HuggingFace Transformers, Flower, FastAPI) and React 19 with TypeScript and Vite on the frontend.

---

## 3.2 Dataset Description

For this work, forty-four of the fifty NIFTY 50 constituent stocks are used. The NIFTY 50 index covers the largest and most liquid companies listed on the National Stock Exchange of India, selected by free-float market capitalisation. [2] Six stocks were dropped because their complete price history was not available through the Yahoo Finance API for the full study period.

Data is sourced using the yfinance library. [33] For each of the forty-four stocks, daily OHLCV data (Open, High, Low, Close, Volume) and the Adjusted Close price are downloaded covering January 2015 to December 2025. The Adjusted Close is the primary price series used for return computation, because it accounts for stock splits and dividend distributions. Using the raw Close price would introduce artificial jumps on ex-dividend dates, distorting the technical indicators and model inputs.

The NIFTY 50 index itself is also downloaded as a separate series. It is used as the benchmark for portfolio performance comparison.

A three-way split is applied to the data. The training set covers January 2015 to December 2021, a span of seven years that includes multiple market regimes. These are the 2016 demonetisation shock, the 2018 NBFC crisis, the COVID crash of March 2020, and the subsequent recovery. The validation set covers January 2022 to December 2023 and was used for hyperparameter tuning and architecture decisions. The test set covers January 2024 to December 2025 and represents completely unseen data for final evaluation. No data from the validation or test sets was used at any stage of model training.

Table 3.1 shows the distribution of stocks across sectors. The sector classification follows the standard NSE sector grouping.

**Table 3.1**: Dataset, Sector-wise Stock Distribution

| Sector | Stocks | Key Stocks |
|--------|--------|-----------|
| Banking | 6 | HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK, INDUSINDBK |
| Finance | 4 | BAJFINANCE, BAJAJFINSV, HDFC Life, Muthoot |
| Information Technology | 5 | TCS, INFOSYS, WIPRO, HCLTECH, TECHM |
| Telecom | 1 | BHARTIARTL |
| Pharmaceuticals | 4 | SUNPHARMA, DRREDDY, DIVISLAB, CIPLA |
| FMCG | 4 | HINDUNILVR, ITC, NESTLEIND, BRITANNIA |
| Energy | 4 | RELIANCE, ONGC, NTPC, POWERGRID |
| Automotive | 3 | MARUTI, TATAMOTORS, EICHERMOT |
| Metals | 3 | TATASTEEL, JSWSTEEL, HINDALCO |
| Others (Infrastructure, Diversified) | 10 | LT, ULTRACEMCO, GRASIM, ADANIENT, etc. |
| **Total** | **44** | |

Date range: January 2015 to December 2025 (approximately 2700 trading days per stock).

---

## 3.3 Data Preprocessing

Raw financial data from Yahoo Finance has several imperfections that need to be addressed before the data can go into model training.

The most common issue is missing values. Indian markets observe scheduled holidays, and on those days no price data is recorded. Gaps are handled by forward-filling. Each missing value is replaced by the most recently observed valid price. Forward-fill is chosen over interpolation because financial prices are not smooth across market closures. Interpolating between a Friday close and a Tuesday open would imply a price movement on Monday (a holiday) that did not actually happen. That kind of artificial signal does not exist in the real market, so introducing it would corrupt the model inputs.

Extreme price observations are checked against a threshold of five standard deviations from the rolling mean. Values beyond this threshold are examined individually. Most correspond to genuine corporate events (bonus share issues, stock splits that the Adjusted Close did not fully correct) or data errors from the source. Data errors are corrected or forward-filled. Genuine corporate events are kept as-is.

Volume data follows the same forward-fill logic for days with zero recorded volume, which typically indicates trading halts.

After cleaning, the data is verified to be in strict chronological order with no remaining NaN values. This ordering is preserved throughout all subsequent processing steps. No shuffling of the time series is applied at any point. Shuffling would destroy the temporal structure that both the RL environment and the T-GAT model depend on, and it would also create data leakage by allowing information from future time steps to appear before past ones in the training sequence.

The final cleaned dataset has shape (trading days, 44, 6), representing all six price fields for all forty-four stocks across all trading days in the study period.

---

## 3.4 Feature Engineering — Technical Indicators

Twenty-one technical indicators are computed per stock from raw OHLCV data, spanning four categories: trend (SMA, EMA, MACD family), momentum (RSI, Stochastic, Williams %R, ROC, CCI), volatility (Bollinger Bands, ATR), and volume (OBV, Volume Ratio, VWAP deviation). Table 3.2 lists all twenty-one indicators with their parameters and purpose.

After computation, all indicators are normalised using a rolling Z-score with a 252-day window:

    Z(t) = (X(t) − rolling_mean(X, 252)) / rolling_std(X, 252)

Rolling normalisation is used rather than global normalisation to prevent data leakage. If global statistics were used, they would incorporate future data into the normalisation of early training points (information that would not be available at deployment time).

**Table 3.2**: All 21 Technical Indicators

| No. | Indicator | Category | Period / Parameter | What it Captures |
|-----|-----------|----------|-------------------|-----------------|
| 1 | SMA 20 | Trend | 20-day window | Short-term price average |
| 2 | EMA 20 | Trend | 20-day, k = 2/21 | Recent-weighted short trend |
| 3 | EMA 50 | Trend | 50-day, k = 2/51 | Intermediate trend |
| 4 | MACD | Trend | EMA(12) - EMA(26) | Trend momentum crossover |
| 5 | MACD Signal | Trend | EMA(9) of MACD | Smoothed MACD |
| 6 | MACD Histogram | Trend | MACD - Signal | Divergence from signal |
| 7 | RSI | Momentum | 14-period | Overbought / Oversold |
| 8 | Stochastic %K | Momentum | 14-period | Close position in range |
| 9 | Stochastic %D | Momentum | 3-period SMA of %K | Smoothed stochastic |
| 10 | Williams %R | Momentum | 14-period | Inverted stochastic |
| 11 | ROC | Momentum | 10-period | Rate of price change |
| 12 | CCI | Momentum | 20-period | Price deviation from average |
| 13 | Bollinger Upper | Volatility | SMA(20) + 2sigma | Resistance level |
| 14 | Bollinger Lower | Volatility | SMA(20) - 2sigma | Support level |
| 15 | Bollinger Bandwidth | Volatility | (Upper - Lower) / Mid | Volatility magnitude |
| 16 | ATR | Volatility | 14-period EMA of True Range | Intraday volatility |
| 17 | Daily Return | Price | (Close_t / Close_{t-1}) - 1 | Single-day price change |
| 18 | Momentum 10 | Price | Close_t / Close_{t-10} | 10-day price momentum |
| 19 | OBV | Volume | Cumulative volume direction | Volume pressure |
| 20 | Volume Ratio | Volume | Volume / SMA(Volume, 20) | Volume anomaly |
| 21 | VWAP deviation | Volume | (Close - VWAP) / VWAP | Price vs. volume-weighted average |

---

## 3.5 Sentiment Analysis Module

Financial news carries information that price history alone cannot capture. When a company announces quarterly results, when a sector regulator releases new guidelines, or when macroeconomic data comes out differently from what analysts expected, the market reaction often begins in news text before it appears in price movements. The sentiment module in FINQUANT-NEXUS extracts this forward-looking signal from live news and converts it into a numerical feature the RL agents can use.

General-purpose sentiment classifiers trained on movie reviews or social media text do not transfer well to financial language. Words like "bear", "short", "correction", and "liability" have domain-specific meanings in finance that a general classifier handles incorrectly. Loughran and McDonald showed this clearly in their 2011 paper on finance-specific lexicons. [24] So a domain-trained model is needed instead.

FinBERT, published by Araci in 2019 [26] and fine-tuned by ProsusAI on a large corpus of financial news and earnings call transcripts [35], is the model used here. It is based on BERT [25], which processes text bidirectionally, considering both the left and right context of each word. The ProsusAI/finbert model outputs three class probabilities: P(positive), P(neutral), and P(negative) for any input text.

The model is downloaded once and stored locally at data/finbert_local/. All subsequent inference runs from this local cache, so the sentiment module does not require an internet connection during operation.

**News Sources.** Three sources are queried for each stock.

The Google News RSS feed provides English-language headlines from a wide range of Indian and international sources. The Yahoo Finance news API provides stock-specific articles. Indian financial RSS feeds from Moneycontrol and Economic Times cover domestic market stories that may not appear in the first two sources.

**Fetching Pipeline.** News fetching is implemented with Python's ThreadPoolExecutor, allowing all forty-four stocks to be queried concurrently rather than one at a time. Each individual source request has a five-second timeout to prevent a slow or unresponsive feed from blocking the entire batch. After fetching, headlines from all three sources for the same stock are deduplicated by comparing headline text. The same article appearing in multiple feeds is counted only once.

**Inference and Score Computation.** Each unique headline is tokenised using the FinBERT tokenizer with a maximum token length of 512. The tokenised input is passed through the FinBERT model, which outputs the three class probability values. The sentiment score for that headline is:

    score = P(positive) − P(negative)

This gives a score in the range from minus one to plus one. A score near plus one indicates strongly positive sentiment. A score near minus one indicates strongly negative. A score near zero indicates neutral.

**Daily Aggregation.** Multiple headlines may be available for the same stock on the same day. These are aggregated into a single daily score using a weighted average, where each headline's weight is proportional to how recently it was published within the trading day. The result is one sentiment score per stock per trading day, giving a feature matrix of shape (trading days, 44).

Market mood is computed as the unweighted average of all forty-four stock scores for a given day. If this average exceeds 0.2, the overall market mood is classified as Bullish. If it falls below minus 0.2, it is Bearish. Values in between are Neutral.

**Caching.** Sentiment inference for forty-four stocks takes roughly fifteen to thirty seconds per full batch on the CPU build. To avoid calling the model repeatedly for every dashboard refresh, results are stored in a SQLite database with a time-to-live (TTL) of three minutes. If a dashboard request arrives within three minutes of the last computation, the cached result is returned immediately. If the cache has expired, fresh inference is triggered.

**Integration into RL Observation Space.** The daily sentiment scores for all forty-four stocks form a forty-four dimensional feature vector. This is concatenated with the technical indicators and T-GAT embeddings to form the full RL observation.

Figure 3.4 (see fqn1/disst/imgs/fig_3_4_sentiment_pipeline.png) shows the complete flow from news sources through inference to the RL observation space.

---

## 3.6 Stock Relationship Graph Construction

Individual stock models treat each company as an independent series. But companies within the same sector respond to the same regulatory changes, and companies linked by supply chains have economically meaningful dependencies. Modelling these relationships explicitly (rather than expecting the RL agent to infer them from correlated price histories alone) gives the system structural information that is stable and interpretable.

The stock universe is represented as a graph G = (V, E), where V is the set of forty-four stock nodes and E is the set of edges. Three types of edges are used, each capturing a different kind of relationship.

**Sector Edges.** Two stocks are connected by a sector edge if they belong to the same NSE sector classification. These edges are undirected, static, and do not change over the study period. The rationale is straightforward. Sector-level events affect all companies in that sector simultaneously. When the Reserve Bank of India changes the repo rate, it affects every Indian bank. When crude oil prices move sharply, all energy-sector stocks respond. Sector edges encode this group membership directly in the graph structure. The graph has 79 sector edges across the eleven sector categories represented in the dataset.

**Supply Chain Edges.** A second set of edges encodes known business dependency relationships between companies in different sectors. These were defined manually based on publicly available information about major supplier and customer relationships among NIFTY 50 companies. Examples include TATASTEEL to MARUTI (steel as input to automotive manufacturing) and ONGC to RELIANCE (crude oil supply). These edges are directed and static, reflecting that the dependency has a direction. A cost change upstream propagates to margins downstream, not the reverse. The graph has 24 supply chain edges.

**Correlation Edges.** The third edge type is dynamic. A 60-day rolling Pearson correlation is computed between the daily returns of every pair of stocks. If the correlation between two stocks exceeds a threshold of 0.6, an undirected edge is created between them for that time step. If the correlation later drops below 0.6, the edge is removed. This captures temporary co-movement clusters, such as the IT sector rallying together when the rupee weakens, or metals stocks moving together when global commodity prices shift. The number of correlation edges therefore changes day to day. Across the 2024 to 2025 test window the count varied roughly between 90 and 220 edges depending on the market regime, with an average close to 150. The figure of 147 edges quoted later (Table 3.4 and Chapter 4.6) refers to the snapshot taken from the dashboard at the time of evaluation.

PyTorch Geometric (PyG) [34] is used for graph construction and batching. PyG stores the graph in a compressed adjacency format with separate edge index tensors for each edge type, which allows the RelationalGATLayer in the T-GAT to apply different attention mechanisms to each edge type separately.

**Table 3.4**: Graph Edge Statistics

| Edge Type | Count | Static or Dynamic | Update Frequency |
|-----------|-------|-------------------|-----------------|
| Sector | 79 | Static | Never |
| Supply Chain | 24 | Static | Never |
| Correlation (60-day rolling, threshold 0.6) | 147 | Dynamic | Daily |
| **Total** | **250** | | |

Node features at each time step consist of the twenty-one technical indicators plus the one FinBERT sentiment score, giving a twenty-two dimensional feature vector per node.

Figure 3.5 (see fqn1/disst/imgs/fig_4_6_graph_all_edges.png) shows the force-directed visualisation of the stock graph with all three edge types enabled.

---

## 3.7 Temporal Graph Attention Network (T-GAT)

The T-GAT model has two components. A RelationalGATLayer that applies edge-type-specific attention, and a GRU Temporal Encoder that captures how stock representations evolve over time.

**RelationalGATLayer.** Separate weight matrices are used for each of the three edge types. For node i and neighbour j connected by an edge of type r (sector, supply_chain, or correlation):

    α_ij^r = softmax_j [ LeakyReLU( a_rᵀ [W_r*h_i ‖ W_r*h_j] ) ]
    h'_i^r = σ( Σ_j α_ij^r * W_r * h_j )

The final node representation concatenates contributions from all three edge types:

    h'_i = h'_i^sector ‖ h'_i^supply ‖ h'_i^correlation

Eight attention heads are used per edge type. Their outputs are averaged before concatenation.

**GRU Temporal Encoder.** A two-layer GRU with hidden size 128 processes the daily sequence of GAT outputs for each stock. The output is projected to a 32-dimensional embedding per stock per time step.

**Training.** The T-GAT is pre-trained on the 2015 to 2021 training set using binary cross-entropy loss on next-day return direction (Adam, lr=0.001). Once pre-training is done, the 32-dimensional embeddings are extracted and stored for use as RL observation features. T-GAT weights are frozen during RL training.

Figure 3.6 (see fqn1/disst/imgs/fig_3_6_tgat.png) shows the T-GAT architecture.

---

## 3.8 Reinforcement Learning Environment

Portfolio allocation is naturally a sequential decision problem. At each trading day, the agent observes the current state of the market and the portfolio, decides how to allocate capital across stocks, then observes the result of that allocation over the next day, and faces the same decision again. This loop, repeated over hundreds of trading days, is what reinforcement learning is designed for.

The environment in this work implements the Gymnasium interface [36] (the actively maintained successor to OpenAI Gym). Following this interface ensures compatibility with the Stable-Baselines3 [18] implementations of all five RL algorithms used in this work.

**Observation Space.** The observation that the agent receives at each time step is a concatenation of four components.

The technical indicator matrix contains twenty-one features for each of the forty-four stocks, giving 924 values. The T-GAT embedding matrix contains thirty-two dimensional embeddings for each of the forty-four stocks, giving 1408 values. The sentiment score vector contains one score per stock, giving 44 values. The current portfolio weight vector contains the current allocation across all forty-four stocks, giving 44 values.

The total observation size is 924 + 1408 + 44 + 44 = 2420. All values are normalised to the range minus one to plus one before being passed to the agent.

**Action Space.** The agent outputs a continuous vector of forty-four values, one for each stock. The raw output is passed through a softmax function to convert it into a valid probability distribution:

    w_i = exp(a_i) / Σ over j of exp(a_j)

This ensures all weights are non-negative and sum to exactly 1.0. Short selling is not permitted.

**Reward Function.** The reward signal is designed to encourage the agent to achieve a high Sharpe ratio while penalising excessive drawdown and unnecessary portfolio turnover:

    R(t) = Sharpe_rolling_30 − λ₁ * drawdown_penalty(t) − λ₂ * turnover_penalty(t)

The rolling Sharpe is computed over the most recent thirty trading days. The drawdown penalty is activated when the portfolio's peak-to-trough drawdown exceeds eight percent. The turnover penalty is proportional to the total absolute change in portfolio weights between consecutive steps, discouraging the agent from rebalancing excessively and incurring unnecessary transaction costs. The penalty weights are λ₁ = 0.5 and λ₂ = 0.3, tuned on the validation set.

Using the Sharpe ratio as the primary reward component rather than raw return is a deliberate choice. Raw return gives no penalty for taking on high risk. An agent optimising for raw return alone would likely take concentrated positions with high volatility. Using Sharpe penalises this behaviour and pushes the agent toward risk-adjusted performance.

**Constraints.** Three hard constraints are enforced at every step.

A maximum position size of 12% per stock prevents excessive concentration in any single company. If the softmax output allocates more than 12% to any stock, the excess is redistributed proportionally to the other stocks.

A stop-loss rule reduces a stock's weight by fifty percent in the next step if its single-day return falls below -3%. This models the risk management behaviour of a professional fund manager.

A maximum portfolio drawdown of -12% terminates the current episode early. This prevents the agent from training in episodes where the portfolio has already suffered a catastrophic loss, which would distort the reward signal.

**Episode Structure.** Each episode covers 252 trading days (one calendar year). The starting date is drawn randomly from the training period (2015 to 2021) at the beginning of each episode, exposing the agent to diverse market conditions during training. A transaction cost of 0.1% per trade and market slippage of 0.05% are deducted from portfolio returns at each rebalancing step.

Figure 3.7 (see fqn1/disst/imgs/fig_3_7_rl_env.png) shows the state-action-reward cycle of the environment.

---

## 3.9 Reinforcement Learning Agents

Five RL algorithms are trained in the environment described in Section 3.8. PPO [13] and A2C [16] are on-policy algorithms that update using data from the current policy. SAC [14], TD3 [15], and DDPG [12] are off-policy algorithms that learn from a replay buffer. The algorithmic details of each are covered in Chapter 2. A key design choice in this work is that all five share the same observation space, action space, and risk constraints, enabling a direct performance comparison on identical data.

**Ensemble Agent.** A sixth agent averages the weight outputs of all five:

    a_ensemble = (a_PPO + a_SAC + a_TD3 + a_A2C + a_DDPG) / 5

Softmax normalisation is applied to produce the final portfolio weights. No additional training is required. The averaging reduces sensitivity to any single algorithm's worst-case behaviour across market conditions.

**Training Protocol.** All five algorithms are trained for 500,000 steps on the 2015 to 2021 training data. Discount factor γ = 0.99. Adam optimizer, learning rate 3×10⁻⁴. Batch size 64 for on-policy methods, 256 for off-policy. All implementations use Stable-Baselines3 v2.8.0 with custom environment wrappers.

**Table 3.3**: RL Algorithm Hyperparameters

| Parameter | PPO | SAC | TD3 | A2C | DDPG |
|-----------|-----|-----|-----|-----|------|
| Type | On-policy | Off-policy | Off-policy | On-policy | Off-policy |
| Training steps | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 |
| Learning rate | 3e-4 | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| Discount factor (gamma) | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| Batch size | 64 | 256 | 256 | 64 | 256 |
| Clip epsilon | 0.2 | — | — | — | — |
| Entropy coefficient | 0.01 | Auto | — | 0.01 | — |
| Policy delay | — | — | 2 | — | — |
| Replay buffer size | — | 1e6 | 1e6 | — | 1e6 |
| Optimizer | Adam | Adam | Adam | Adam | Adam |

---

## 3.10 Stress Testing Framework

Backtesting on historical data tells how a portfolio would have performed during the specific sequence of events that actually occurred. Stress testing answers a different question. How does the portfolio behave under market conditions that are more extreme or more volatile than what was observed in the test period?

The stress testing module uses Monte Carlo simulation. For each of eight crisis scenarios, one thousand independent forward price paths are generated. Each path covers 252 trading days. The Ensemble RL agent's portfolio weights (computed on the most recent observation from the test set) are held fixed across all paths in a given scenario. What this does is isolate the effect of the market scenario from any agent rebalancing decisions.

A GAN-based return generator is used to calibrate the simulation parameters. The generator is a three-layer LSTM with hidden size 128 and latent dimension 64, trained for 500 epochs on the daily returns of the 44 stocks across the 2015 to 2021 training window. During training, the discriminator learns to distinguish real return sequences from generated ones, and the generator learns to produce sequences whose statistical fingerprint matches the historical data. The advantage over a plain Gaussian model is that the GAN preserves heavy tails, volatility clustering, and short-term momentum that Indian equity returns actually exhibit. The forward simulation in Section 4.10 (Black Bootstrap) uses the same calibrated generator to draw block-resampled paths over a one-year horizon.

The eight scenarios, each calibrated to a historical crisis period, are: Normal (baseline), 2008 Financial Crisis, COVID-19 Crash, Flash Crash, Dot-Com 2000, India Bear 2015, Rate Hike 2022, and Geo-Political Shock. Each applies volatility parameters from the corresponding historical period rather than long-run averages.

Three risk metrics are computed from the distribution of simulated returns across the one thousand paths for each scenario.

Value at Risk at the 95th percentile (VaR 95%) is the return that is not exceeded in 95% of the simulated paths. In other words, only 5% of paths have a worse outcome.

Conditional Value at Risk at the 95th percentile (CVaR 95%), also called Expected Shortfall, is the average return across the worst 5% of paths. This is more informative than VaR alone because it characterises the severity of tail losses, not just the threshold. [7]

The survival rate is the percentage of paths in which the portfolio's drawdown stays within the maximum drawdown constraint of -12%. This metric is directly interpretable. It is the probability that the portfolio survives the scenario without hitting its circuit breaker.

---

## 3.11 Federated Learning System

The motivation for federated learning here is practical. Sector-specific portfolio data is sensitive. A banking fund's allocation across HDFCBANK, ICICIBANK, and KOTAKBANK reflects proprietary investment decisions. An energy fund's positions in RELIANCE, ONGC, and NTPC reflect months of research and risk assessment. Neither fund would share this data with a central server, and regulatory constraints on sharing client portfolio information make centralised training infeasible in real institutional settings.

So federated learning is used instead. Each sector participant trains a local model on their own data and shares only the model weight updates. The central server never sees the raw data of any client.

**Client Architecture.** Four sector clients participate in the federated training. The four clients between them cover all 44 stocks (10 + 6 + 8 + 20 = 44). The percentages quoted below describe each client's share of the *total portfolio weight* under the Ensemble allocation, not the share of the stock count. The two figures differ because the Ensemble allocates capital unevenly across sectors.

Client 1 (Banking and Finance) holds ten stocks: HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK, INDUSINDBK, BAJFINANCE, BAJAJFINSV, and others in the finance category. This client carries roughly 23% of the portfolio by weight.

Client 2 (IT and Telecom) holds six stocks: TCS, INFOSYS, WIPRO, HCLTECH, TECHM, and BHARTIARTL. This client carries roughly 14% of the portfolio by weight.

Client 3 (Pharma and FMCG) holds eight stocks: SUNPHARMA, DRREDDY, DIVISLAB, CIPLA, HINDUNILVR, ITC, NESTLEIND, and BRITANNIA. This client carries roughly 18% of the portfolio by weight.

Client 4 (Energy, Auto, and Others) holds twenty stocks covering the energy, automotive, metals, infrastructure, and conglomerate categories. This client carries roughly 45% of the portfolio by weight.

**Local Training.** In each communication round, each client receives the current global model weights from the server. It trains locally for five epochs on its own sector data. Only the weight updates (the difference between the locally trained weights and the received global weights) are computed and sent back to the server. The raw training data never leaves the client.

**DP-SGD (Differential Privacy).** Before sending weight updates to the server, each client applies DP-SGD noise. [30] The mechanism clips each gradient to a maximum L2 norm of 1.0 to prevent any single data sample from having too large an influence on the update:

    g_clipped = g * min(1, C / ‖g‖₂)

Calibrated Gaussian noise is then added:

    g_noisy = g_clipped + N(0, σ² * I)

The noise standard deviation σ is set to achieve a privacy budget of ε = 8.0 with δ = 0.00001 across 50 communication rounds. The choice of ε = 8.0 is a deliberate balance. Stricter privacy (smaller ε) requires more noise, which degrades model quality noticeably.

**FedProx Server Aggregation.** Standard FedAvg aggregation computes a weighted average of client model weights proportional to each client's data size. [28] This works well when client data is similarly distributed, but the four sector clients in this system have very different return distributions, volatility patterns, and factor exposures. Under FedAvg, the client with the most data or the highest loss gradient tends to dominate the aggregated model.

FedProx [29] addresses this by adding a proximal term to each client's local optimisation objective:

    minimize F_k(w) + (μ / 2) * ‖w − w_global‖²

The proximal term penalises the local model for moving too far from the global model during local training. With μ = 0.01, client models are regularised toward the global model at each step, preventing any single sector from dominating the global update. The convergence chart in the dashboard (Figure 3.8, see fqn1/disst/imgs/fig_4_8_federated.png) shows that FedProx converges faster and more smoothly than FedAvg on this heterogeneous sector data.

**Communication Rounds.** The full federated training runs for 50 rounds. In each round, the server broadcasts the current global model weights to all four clients. Each client trains locally for five epochs under DP-SGD. Each client sends its noisy weight updates back to the server. The server applies FedProx aggregation to produce the updated global model. The global Sharpe ratio of the aggregated model is tracked across rounds. After 50 rounds, the system reaches a converged state (marked CONVERGED in the dashboard) with a Global Sharpe of 0.729.

The global FL model's sector allocation weights are used as one of three signals in the portfolio's smart optimisation mode, contributing a 20% signal weight alongside RL (40%) and sentiment (40%).

---

## 3.12 REST API Design

The FastAPI backend [37] connects all Python-based backend components (data pipeline, RL models, T-GAT, sentiment, FL system, stress testing) with the React frontend. FastAPI is chosen because it provides automatic request validation through Pydantic models, generates interactive API documentation at /docs automatically, and supports asynchronous request handling, which is important for long-running calls like live sentiment inference.

CORS middleware is configured to allow requests from the React development server at localhost:3000 to the FastAPI server at localhost:8000.

The API endpoints are organised into functional groups. The health group provides a liveness check at GET /api/health. The portfolio group provides the current portfolio summary (Sharpe, Sortino, return, drawdown, holdings table) at GET /api/portfolio-summary. The RL group provides performance metrics for all six algorithms at GET /api/rl-summary and training reward curves at GET /api/rl-training-data. The sentiment group at GET /api/news-sentiment triggers live FinBERT inference and returns scores for all forty-four stocks along with recent headlines. The stress testing group at GET /api/stress-test returns the Monte Carlo results for the currently selected scenario. The federated learning group at GET /api/fl-summary returns convergence curve data, privacy tracker, and client fairness metrics. The graph group at GET /api/gnn-summary returns node, edge, and embedding statistics.

In total, the API exposes more than fifty endpoints covering every data dimension of the system.

Caching is applied at two levels. Historical price data and pre-computed technical indicators are held in memory after the first load, avoiding repeated disk reads for every API call. News sentiment results are cached in SQLite with a three-minute TTL.

All responses follow a consistent JSON schema defined by Pydantic models. This makes the React frontend code predictable and allows the Swagger documentation to accurately describe every endpoint's response structure.

---

## 3.13 Dashboard Design

The FINQUANT-NEXUS dashboard is built with React 19, TypeScript, and Vite. State management uses Zustand for UI state and React Query for API data. Charts use Recharts. Layout uses Tailwind CSS. Pipeline animations use Framer Motion.

The eight pages and their primary content are:

- **Portfolio**: Sharpe/Sortino/Return/Drawdown metric cards, holdings table, growth chart vs NIFTY 50 and Fixed Deposit
- **RL Agent**: Per-algorithm metrics, six-algorithm comparison table, training reward curves, sector allocation chart
- **Stress Testing**: Eight crisis scenario cards (VaR, CVaR, survival rate), Monte Carlo fan chart for selected scenario
- **Federated Learning**: FedProx vs FedAvg convergence chart, epsilon tracker, per-client Sharpe improvement
- **Sentiment**: Live FinBERT scores per stock, market mood indicator, news feed with sentiment labels, sector-level chart
- **Graph Visualisation**: Force-directed stock network, edge-type toggles, node detail panel on click
- **Pipeline**: Animated 15-stage data flow diagram with status indicators
- **Future Prediction**: Black Bootstrap 1,000 forward paths, probability of profit, per-algorithm forward comparison

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
