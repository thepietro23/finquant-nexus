# Chapter 1 — Introduction
## Target: 8–10 pages | Status: [x] Done — Rewritten per prompt.md
## Word count: ~2900 words

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 1
# INTRODUCTION

---

## 1.1 Background of the Work

The National Stock Exchange of India manages one of the highest-volume equity markets in Asia, and its benchmark index, the NIFTY 50, tracks fifty large-cap companies spread across thirteen sectors of the Indian economy. [2] Together these companies account for more than sixty percent of total NSE trading value by market capitalisation. Any investor managing Indian equity exposure uses this index as the reference point, whether directly or as a comparison.

Managing a portfolio anchored to the NIFTY 50 is not just a matter of picking which stocks to buy. The harder question is always how much capital to put in each stock at each point in time, and how to adjust those weights as market conditions change. This is the portfolio allocation problem, and it is considerably more difficult than stock selection alone. Getting it consistently right across different market phases, bull runs, sideways markets, sharp corrections, is something even experienced fund managers find difficult.

The classical theory for handling this comes from Markowitz's work published in 1952. [1] He showed that if the expected returns and covariance of a set of assets are known, it is mathematically possible to find the allocation that minimises risk for a given level of expected return. The resulting set of solutions, the efficient frontier, became the central concept in modern portfolio theory. The framework is still taught today.

In practice, these assumptions create real problems for equity markets. One assumption is that returns follow a normal distribution. Indian stock returns do not. Sharp one-day drops and sudden recoveries happen far more often than a normal distribution would predict, particularly around election results, interest rate decisions, or sector-level regulatory announcements. [4] Another assumption is that correlations between stocks remain constant over time. In practice, correlations spike sharply during market downturns. Stocks that moved independently during normal conditions suddenly move together when there is a sell-off, and that is exactly when diversification was supposed to protect the portfolio. So the theoretical protection fails at the moment it matters most.

A separate estimation problem makes this worse. For fifty stocks, a full covariance matrix requires estimating 1225 pairwise correlations. Small errors in these estimates, which are unavoidable when using historical data, can shift the computed optimal portfolio dramatically. The method is sensitive to input quality in a way that makes it fragile outside controlled academic settings.

Research over the past decade has moved toward learning-based methods for portfolio management. Reinforcement learning, in particular, has been explored for this problem. [9] A reinforcement learning agent does not require a pre-specified return model. It learns a policy by interacting with historical data, taking allocation decisions, observing portfolio outcomes, and adjusting over time. This trial-and-error process, when applied to financial time series, can discover allocation strategies that adapt to market conditions rather than assuming stability. Also, it can be designed to handle constraints like maximum position limits and drawdown controls in a natural way.

Price data and technical indicators alone, however, leave out two important information dimensions. The first is the structural relationship between companies. Stocks within the same sector tend to move together when sector-specific news arrives. A change in banking regulation affects HDFCBANK, ICICIBANK, KOTAKBANK, and AXISBANK in similar ways. Supply chain connections create more specific dependencies: if raw material prices rise for TATASTEEL, that affects the input costs and margins of manufacturers like MARUTI SUZUKI. These relationships cannot be modelled by treating each stock as an independent series. Graph neural networks provide a way to represent them formally. [19, 20] A graph where stocks are nodes and edges represent sector membership, supply chain links, or return correlations allows a neural network to learn representations that include relational context alongside individual price history.

The second missing dimension is news sentiment. Markets respond to information, and a large part of that information arrives as text before it appears in price movements. Earnings announcements, central bank statements, sector-specific regulatory changes, governance disclosures, all of these shift investor behaviour before the price chart shows anything. Pre-trained language models adapted for financial text, such as FinBERT [26], can process news headlines and assign a sentiment score ranging from negative to positive for each company. Including this as an input to a portfolio model gives the system access to a signal that historical prices alone cannot provide.

A separate challenge arises when considering how multiple institutional participants could collaborate on training better models. A fund managing banking stocks and a fund managing technology stocks would both benefit from a shared, more general portfolio model. But they cannot share their actual holdings or transaction data. Federated learning [28] addresses this by allowing each participant to train locally and share only model weight updates. A central server aggregates these updates into a global model that is better than what any single participant could produce alone, without ever seeing the raw data of any client.

These four ideas — RL portfolio allocation, graph-based relational stock modelling, text sentiment signals, and federated learning — exist as separate contributions across the research literature. No open system has brought all four together for the NIFTY 50 universe and presented the combined output through a dashboard that non-specialists can actually use. FINQUANT-NEXUS is an attempt to build that system.

---

## 1.2 Motivation

A specific observation shaped this project. After 2020, retail investor participation in Indian equity markets grew at a pace that had not been seen before. SEBI data shows the number of active demat accounts went from around 40 million in 2019 to over 140 million by 2024. [3] Most of these are first-time investors without formal finance training. They make decisions based on news, tips, or simple heuristics, none of which are reliable for managing a diversified equity portfolio over time.

At the same time, the tools available to researchers studying portfolio optimisation for Indian markets are fragmented. Sophisticated methods like RL-based portfolio allocation, graph-based stock relationship modelling, and federated learning exist in academic literature, but there is no open platform that implements all of them together for NIFTY 50 stocks and makes the results interpretable through a working interface.

This gap motivated the design of FINQUANT-NEXUS. The system is not a trading product. It is a research and demonstration platform that shows what becomes possible when multiple machine learning components are integrated properly for the Indian market. The goal is to make the output of these methods accessible, not just to quantitative researchers, but to anyone who wants to understand how an AI-driven portfolio system works and what it produces.

---

## 1.3 Problem Statement

Existing approaches to equity portfolio management for Indian markets handle one problem at a time. Reinforcement learning systems for portfolio allocation typically treat each stock independently and ignore how companies relate to each other through their sector or supply chain. Graph-based models can encode these relationships but do not learn dynamic portfolio policies. Sentiment tools extract signals from news but have no direct connection to portfolio weight decisions. Federated learning systems for finance exist, but not in the context of portfolio optimisation across Indian sector-based clients.

The specific problem is that no single available system, open or commercial, integrates all four components for the NIFTY 50 stock universe. No system models stocks as a multi-relational graph, trains portfolio weights through reinforcement learning using combined price and sentiment inputs, applies differential privacy to sector-based federated training, and presents all of this through an interactive dashboard with stress testing and forward simulation. This integration gap is what the present work addresses.

---

## 1.4 Objectives of the Work

Seven objectives were defined for this dissertation:

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

This dissertation covers the design, implementation, and evaluation of FINQUANT-NEXUS as a locally deployed simulation system. The stock universe is limited to forty-four NIFTY 50 constituent stocks for which a complete price history was available through the Yahoo Finance API. Six index constituents were excluded because of incomplete data. The historical dataset spans January 2015 to December 2025, split into training (2015 to 2021), validation (2022 to 2023), and test (2024 to 2025) periods.

The entire system runs on a single local machine. There is no connection to any live brokerage, and no real-money execution is implemented. All portfolio performance values in this dissertation come from historical simulation. Transaction costs are modelled at 0.1% per trade and slippage at 0.05%. Market impact and bid-ask spread are not modelled.

The federated learning module simulates four sector clients on the same physical machine with logically separated data. Physical network distribution and real inter-institution communication are not part of this work.

Two components present in the broader codebase, a Quantum Approximate Optimisation Algorithm (QAOA) module and a Neural Architecture Search (NAS/DARTS) module, are outside the scope of this dissertation. Both require dedicated theoretical treatment and are not part of the eight-tab dashboard interface this work covers.

---

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
