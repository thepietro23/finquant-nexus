# Chapter 1 — Introduction
## Target: 8–10 pages | Status: [x] Done — Rewritten per prompt.md
## Word count: ~2900 words

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 1
# INTRODUCTION

---

## 1.1 Background of the Work

The National Stock Exchange of India is among the highest-volume equity exchanges in Asia, and its benchmark index, the NIFTY 50, tracks fifty large-cap companies spread across thirteen sectors of the Indian economy. [2] Together these companies account for more than sixty percent of total NSE trading value by market capitalisation. For any investor with Indian equity exposure, this index acts as the primary reference point, whether held directly or simply used as a comparison benchmark.

Managing a NIFTY 50 portfolio is not just about picking which stocks to hold. The harder question is how much capital should go into each stock at any given point, and how those weights should shift as conditions change. This is the portfolio allocation problem. Getting it right consistently across bull runs, sideways stretches, and sudden corrections is something even experienced fund managers struggle with. Treating it as a solved problem would therefore be a mistake.

The theoretical starting point for this problem comes from Markowitz's 1952 paper. [1] He showed that if expected returns and the covariance of a set of assets are known, mathematics can find the allocation that minimises risk for any given return level. Every such solution traces out the efficient frontier, which became the central idea in modern portfolio theory. The framework is still taught in every finance course today.

But the assumptions underneath it break down badly in real markets. Stock returns are assumed to follow a normal distribution. Indian equities do not behave that way. Sharp single-day drops and fast recoveries occur far more often than any normal distribution predicts, especially around election results, RBI rate decisions, or sector-level regulatory announcements. [4] Also, correlations between stocks are treated as fixed over time. They are not. During a sell-off, stocks that had been moving independently suddenly start moving together. That is precisely when diversification is supposed to protect the portfolio, and that is exactly when the model fails.

A second problem compounds this. For fifty stocks, computing a full covariance matrix means estimating 1225 pairwise correlations. Small estimation errors, which are unavoidable when using historical data, can shift the computed optimal weights dramatically. So the method is too sensitive to input noise to be reliable outside controlled settings.

Research has therefore moved toward learning-based methods. Reinforcement learning in particular has been studied for portfolio management. [9] An RL agent does not need a pre-specified return model. It interacts with historical data, makes allocation decisions, observes what happens to the portfolio, and updates its policy over time. Run over financial time series, this process can find allocation strategies that adapt to changing conditions rather than assuming stability. Also, the environment can be designed to enforce constraints like position limits and drawdown controls directly, which is difficult to handle cleanly in classical optimisation.

Price data and technical indicators still leave out two things that matter. The first is company relationships. Stocks in the same sector move together when sector news breaks. A change in banking regulation hits HDFCBANK, ICICIBANK, KOTAKBANK, and AXISBANK in similar ways at the same time. Supply chain links are more specific. If raw material costs rise for TATASTEEL, that flows into the margins of manufacturers like MARUTI SUZUKI. Treating each stock as an independent series captures none of this. Graph neural networks offer a way to represent these relationships formally. [19, 20] A graph where nodes are stocks and edges encode sector membership, supply chain links, or return correlations lets a network learn representations that carry relational context alongside individual price history.

The second missing piece is news sentiment. A large share of market-moving information arrives as text before prices reflect it. Earnings calls, central bank statements, regulatory changes, governance disclosures, all of these shift investor behaviour before any price chart shows a clear signal. Pre-trained financial language models like FinBERT [26] can read news headlines and return a sentiment score from negative to positive for each company on a given day. Feeding that into the portfolio model gives it access to a signal that price history simply cannot provide.

A further question is how multiple institutions could jointly train better models without sharing sensitive data. A fund focused on banking stocks and one focused on technology would both benefit from a shared general model. But neither can hand over holdings or transaction history. Federated learning [28] addresses this by letting each participant train locally and share only model weight updates. A central server aggregates those updates into a global model that performs better than what any single participant could produce alone, and never sees raw data from any client.

These four ideas, namely RL-based allocation, graph-based relational modelling, news sentiment signals, and federated learning, each have their own research literature. No open system has put all four together for the NIFTY 50 and shown the results through an interface that a non-specialist can actually use. FINQUANT-NEXUS is the attempt to build that system.

---

## 1.2 Motivation

One observation shaped this project more than anything else. After 2020, retail participation in Indian equity markets grew faster than at any earlier point. SEBI data shows active demat accounts went from around 40 million in 2019 to over 140 million by 2024. [3] Most of these are first-time investors with no formal finance background. They make decisions based on news tips, social media, or simple heuristics. None of those are reliable for managing a diversified equity portfolio across different market conditions over time.

At the same time, research tools available for studying portfolio optimisation on Indian markets remain fragmented. Methods like RL-based allocation, graph-based stock modelling, and federated learning all appear in academic papers. But there is no open platform that implements them together for NIFTY 50 stocks, trains them on real data, and shows outputs through a working interface.

That gap is what motivated this work. FINQUANT-NEXUS is not a trading product. It is a research and demonstration platform. The goal is to show what becomes possible when these components are integrated for the Indian market context, and to make the outputs readable not just by quantitative researchers but by anyone trying to understand what an AI-driven portfolio system actually does.

---

## 1.3 Problem Statement

Most existing approaches to equity portfolio management for Indian markets address one piece of the problem in isolation. RL systems for portfolio allocation typically treat each stock as an independent input, ignoring the sector and supply chain relationships between companies. Graph-based models can encode those relationships but do not learn dynamic portfolio allocation policies. Sentiment tools extract signals from news text but have no direct connection to portfolio weight decisions. Federated learning systems for finance exist in literature, but not in the context of portfolio optimisation with Indian sector-based clients.

The specific gap is this. No available system, open or commercial, integrates all four components for the NIFTY 50 stock universe. No system simultaneously models stocks as a multi-relational graph, trains portfolio weights through reinforcement learning on combined price and sentiment inputs, applies differential privacy to sector-based federated training, and presents all of this through an interactive dashboard with stress testing and forward simulation. This integration gap is what the present work addresses.

---

## 1.4 Objectives of the Work

The work was structured around seven objectives. Each one corresponds to a specific module in the FINQUANT-NEXUS system:

1. To collect and preprocess ten years of daily price and volume data for forty-four NIFTY 50 constituent stocks (January 2015 to December 2025) using the Yahoo Finance API, and compute twenty-one technical indicators as the primary input feature set.

2. To build a news sentiment pipeline using the ProsusAI FinBERT model that produces a daily sentiment score in the range of minus one to plus one for each stock, with SQLite-based caching to support live dashboard updates.

3. To construct a multi-relational stock graph with three edge types (sector membership, supply chain linkage, and 60-day rolling return correlation), and train a Temporal Graph Attention Network (T-GAT) to generate thirty-two dimensional stock embeddings that capture both structural relationships and temporal dynamics.

4. To design a Gymnasium-compatible reinforcement learning environment with a Sharpe-ratio-based reward function and hard constraints on maximum position size (12%), stop-loss per trade (minus 3%), and portfolio-wide maximum drawdown (minus 12%), and to train five RL algorithms (PPO, SAC, TD3, A2C, and DDPG) along with a meta-level Ensemble agent that averages the outputs of all five.

5. To implement a Monte Carlo stress testing framework that runs 1000 simulation paths under eight historical crisis scenarios, reporting Value at Risk at the 95th percentile, Conditional Value at Risk, and portfolio survival rate for each scenario.

6. To design and simulate a federated learning system using FedProx aggregation and DP-SGD differential privacy across four sector-based clients (Banking and Finance, IT and Telecom, Pharma and FMCG, Energy and Auto and Others), with privacy budget epsilon equal to 8.0 and delta equal to 0.00001 across 50 communication rounds.

7. To develop a FastAPI REST backend and a React-based web dashboard with eight interactive pages (Portfolio Analytics, RL Agent Control, Sentiment Analysis, Graph Visualisation, Stress Testing, Federated Learning, Data Pipeline, and Future Prediction) that present system outputs in real time.

These objectives cover the full pipeline from raw data collection to interactive result presentation. Each one maps directly to a module in the FINQUANT-NEXUS system.

---

## 1.5 Scope of the Work

In this work, the design, implementation, and evaluation of FINQUANT-NEXUS covers a locally deployed simulation system. The stock universe is forty-four NIFTY 50 constituent stocks for which a complete price history was available through the Yahoo Finance API. Six index constituents were excluded because of incomplete data. The dataset spans January 2015 to December 2025, divided into training (2015 to 2021), validation (2022 to 2023), and test (2024 to 2025) periods.

Everything runs on a single local machine. There is no brokerage connection and no real-money execution. All portfolio performance figures in this dissertation come from historical simulation. Transaction costs are modelled at 0.1% per trade and slippage at 0.05%. Market impact and bid-ask spread are not modelled.

For the federated learning module, four sector clients are simulated on the same physical machine with logically separated data. Real network distribution and actual inter-institution communication are not part of this work.

---

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
