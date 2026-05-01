# Chapter 2 — Literature Review
## Target: 15–20 pages | Status: [x] Done — Topic-wise, focused (6–7 key papers in depth)
## Word count: ~5000 words (~17 pages at 300 words/page)

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 2
# LITERATURE REVIEW

---

This chapter surveys existing research across six technical areas that form the foundation of FINQUANT-NEXUS. These are classical portfolio optimisation, deep reinforcement learning applied to finance, graph neural networks for stock market modelling, financial sentiment analysis, federated learning in financial systems, and Monte Carlo methods for risk management. Within each area, the focus stays on the one or two papers whose methods or findings are most directly connected to design decisions made in this dissertation. Supporting work is referenced where it provides necessary context. The chapter closes with a research gap analysis and a comparison table.

---

## 2.1 Classical Portfolio Optimisation

Before 1952, diversification was investment practice based on instinct. Practitioners knew that holding multiple assets was safer than holding one, but could not say precisely how much safer, or how to find the best possible combination. Markowitz changed that in 1952. [1] His contribution was to show that portfolio selection is a constrained mathematical optimisation problem, and to give it a precise formulation that anyone could actually compute.

The core insight is that portfolio risk, measured as variance of returns, is not the weighted average of individual asset variances. It depends on how each asset's returns move relative to every other. When two assets tend to move in opposite directions, holding both lowers total portfolio variance below what either asset produces alone. Markowitz encoded this in the covariance matrix. Given N assets, each with an expected return and pairwise covariances with all others, the minimum-variance portfolio for any target return level is found by solving a quadratic program. Solving this across all feasible return targets traces the efficient frontier, which is the set of portfolios that cannot be improved without accepting greater risk.

The practical effect was real. It gave diversification a mathematical proof rather than an intuition, and defined the risk-return trade-off in computable terms. The Sharpe ratio and the entire quantitative portfolio management field trace back directly to this 1952 formulation.

For real Indian equity markets, the Markowitz framework hits three hard limits that shaped the design of FINQUANT-NEXUS. The first is the distributional assumption. Asset returns are assumed to be normally distributed, so mean and variance describe the distribution completely. NIFTY 50 returns do not follow a normal distribution. Single-day drops of 5 to 10 percent driven by RBI policy decisions, SEBI regulatory changes, or global risk-off events happen far more often than any normal distribution would predict. These tail events are exactly the ones that matter most for risk, and the framework cannot handle them properly. In simple terms, the model was not built for markets like India's.

The second limit is input sensitivity. For 44 stocks, the covariance matrix needs 946 unique pairwise values estimated from historical data. Small estimation errors compound into large shifts in computed weights. A 1 percent error in one correlation can move optimal weights by 5 percent or more in a concentrated portfolio, making the output unreliable in practice.

Third, and most serious, is stationarity. The efficient frontier is computed once, using fixed historical estimates. It does not update as markets change. During the 2020 COVID crash and the 2022 rate hike cycle, correlations between NIFTY 50 stocks shifted sharply. Stocks that moved independently under normal conditions suddenly moved together in the sell-off. A static allocation computed before the crisis had no way to respond to this.

FINQUANT-NEXUS addresses all three. The RL environment learns allocation policies from market interactions rather than from a fixed return and covariance model, so it adjusts to non-stationarity. The T-GAT captures structural stock relationships that covariance matrices miss. The stress testing module builds non-normal tail scenarios calibrated to actual Indian and global crises. The Sharpe ratio stays as the central evaluation metric and reward signal, preserving the connection to Markowitz's risk-adjusted return framework while removing the restrictive assumptions underneath it.

---

## 2.2 Deep Reinforcement Learning in Finance

Reinforcement learning treats portfolio management as a sequential decision problem rather than a one-shot optimisation. An agent observes the current state of the market and portfolio, selects a portfolio weight allocation, receives a reward based on the resulting performance, and repeats. Through interaction with historical data, the agent learns a policy, which is a mapping from states to actions, that maximises cumulative risk-adjusted return. No pre-specified return or covariance model is needed. The agent finds what works by trying and failing on actual market data.

**Proximal Policy Optimization (PPO), Schulman et al., 2017 [13]**

The most practically useful contribution to stable policy gradient training came from Schulman and collaborators in 2017. Policy gradient methods had a known instability problem. Gradient steps in the wrong direction could collapse a well-trained policy in a single update because no constraint existed on step size. Trust Region Policy Optimization (TRPO) fixed this by imposing a hard limit on policy change per update, which stabilised training but required expensive second-order optimisation that was difficult to implement and tune.

PPO solved the same problem with a simpler mechanism. It uses a clipped surrogate objective. For each gradient step, a ratio r_t measures how much the new policy differs from the old one for each sampled action. The objective clips this ratio to the range [1 minus epsilon, 1 plus epsilon], preventing the policy from moving too aggressively in either direction:

    L_CLIP = E[ min( r_t(θ) * A_t ,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

where A_t is the advantage estimate at time step t and epsilon is typically set to 0.2. The minimum operation ensures that if the unclipped objective would encourage a very large update, the clipped version limits it. This achieves training stability comparable to TRPO without second-order methods, making PPO faster and easier to apply. On benchmark continuous control tasks in MuJoCo, PPO matched or exceeded TRPO, A3C, and DDPG while requiring less computation. It became the standard choice for continuous-action RL applications from 2017 onward.

In FINQUANT-NEXUS, PPO is one of five RL algorithms trained in the Gymnasium-compatible portfolio environment. Its on-policy nature, meaning it only learns from data collected by the current policy version, makes it sample-efficient compared to replay-buffer methods. This matters when training data is limited to 1,757 daily sessions from 2015 to 2021. PPO achieved a Sharpe Ratio of 0.7829 and a total return of 15.22% on the 2024 to 2025 held-out test period. Its conservative on-policy behaviour provides a stable baseline against which the off-policy algorithms (SAC, DDPG, TD3) can be compared.

**FinRL, Liu et al., 2021 [17]**

The most directly related prior work to FINQUANT-NEXUS is the FinRL library published by Liu and collaborators in 2021. FinRL was the first open-source framework for applying deep reinforcement learning to financial portfolio management in a standardised way. Its design wraps real market data inside a Gymnasium-compatible environment, implements the same five RL algorithms used in FINQUANT-NEXUS (PPO, SAC, TD3, A2C, DDPG) through Stable-Baselines3, and provides pre-built pipelines for portfolio allocation tasks. On the Dow Jones 30 allocation problem, trained on 2009 to 2020 data and tested over the COVID-19 crash period, the FinRL Ensemble agent achieved an annualised Sharpe Ratio of approximately 0.98, outperforming passive buy-and-hold and classical minimum-variance strategies.

However, the method did not address several things that matter for this work. FinRL's observation space contains only price-derived technical indicators. Each stock is treated as an independent time series with no representation of how stocks relate structurally through sector membership or supply chain links. There is no sentiment module and no federated learning component. FinRL is also built and benchmarked entirely on US markets, with no support for NIFTY 50 stocks or Indian market-specific events. FINQUANT-NEXUS extends the FinRL concept into a richer, Indian-market setting. Same environment structure and algorithm set, but with T-GAT graph embeddings and FinBERT sentiment scores added to the observation space, a new federated training framework for sector-based clients, and a full NIFTY 50 dataset covering 2015 to 2025.

---

## 2.3 Graph Neural Networks for Stock Market Modelling

Standard time series models treat each stock as an independent series. But companies are not independent. Stocks in the same sector respond to the same regulatory and macroeconomic events. Supply chain relationships create directed dependencies between companies in different sectors. Return correlations reflect clusters of stocks that behave similarly during market stress. Capturing this relational structure requires modelling graphs, which are networks where nodes are stocks and edges represent the relationships between them.

**Graph Attention Networks, Velickovic et al., 2018 [20]**

The central paper in this area for FINQUANT-NEXUS is the Graph Attention Network proposed by Velickovic and collaborators in 2018. Earlier graph neural network methods, particularly the Graph Convolutional Network of Kipf and Welling (2017), updated each node's representation by aggregating its neighbours' features with equal weight. Equal-weight aggregation works when all connections are equally informative, but this fails in stock graphs where some relationships carry far more signal than others. The correlation between HDFCBANK and ICICIBANK during a banking sector event is qualitatively different from the correlation between either of them and a consumer goods stock. So a fixed aggregation scheme does not work here.

GAT replaces fixed equal weights with learned attention coefficients. For each edge (i, j) in the graph, the model computes a scalar attention score e_ij by applying a shared learnable weight vector a to the concatenation of the transformed feature vectors of nodes i and j:

    e_ij = LeakyReLU( aᵀ [ W * h_i  ‖  W * h_j ] )

These raw scores are normalised across all neighbours of node i using softmax:

    α_ij = exp(e_ij) / Σ over k in N_i of exp(e_ik)

The updated representation of node i is then the weighted combination of its neighbours' transformed features:

    h'_i = σ ( Σ over j in N_i of α_ij * W * h_j )

Multi-head attention, where K independent attention mechanisms whose outputs are concatenated or averaged, stabilises the learning and increases expressive capacity. On the Cora, Citeseer, and PPI node classification benchmarks, GAT outperformed GCN and several other graph methods. On PPI specifically, GAT achieved 97.3% micro-F1 against 88.1% for GraphSAGE. That is a large gap on a complex multi-label task.

Two limitations are directly relevant to the stock market use case. The original GAT handles only a single edge type. A stock graph naturally has multiple structurally different relationship types, such as sector co-membership, supply chain links, and statistical return correlations, each carrying different economic meaning. Treating them as a single homogeneous edge set throws away information that distinguishes fundamentally different kinds of stock dependency. Also, standard GAT has no temporal component. Stock relationships and node features change over time, and a single static attention layer cannot capture how the relative importance of one stock's signal for another shifts across different market regimes.

The T-GAT model in FINQUANT-NEXUS addresses both directly. Separate graph attention layers are applied for each of the three edge types, namely sector membership (79 edges), supply chain linkage (24 edges), and 60-day rolling return correlation (147 edges), letting the model learn distinct attention patterns for structurally different connections. A GRU temporal encoder then processes the per-timestep node representations to capture how relational importance evolves over time. The output is a 32-dimensional embedding per stock per time step that encodes both structural position in the three-layer relationship graph and temporal co-movement dynamics. These embeddings form one of three components of the RL observation vector.

---

## 2.4 Financial Sentiment Analysis

Price data and technical indicators capture what has already happened in the market. A large share of price-relevant information arrives as text, in the form of earnings announcements, central bank statements, sector-level regulatory changes, and corporate governance disclosures, before it appears in price movements. Financial sentiment analysis tries to quantify the directional signal in this text.

**FinBERT, Araci, 2019 [26]**

Araci proposed FinBERT in 2019, and it remains the most important contribution to financial sentiment analysis for this kind of work. The specific implementation used in the proposed system is the ProsusAI/finbert variant hosted on HuggingFace. The base model is BERT (Devlin et al., 2019), a bidirectional transformer pre-trained on 3.3 billion words. BERT processes text bidirectionally, meaning every word's representation depends on both its left and right context simultaneously, which gives it substantially better sentence-level understanding than unidirectional language models.

General BERT performs poorly on financial text because the pre-training corpus (Wikipedia and BooksCorpus) builds associations between words and sentiment that do not transfer to financial language. The word "liability" sounds negative in general English but is neutral in a balance sheet context. "Outstanding shares" sounds positive in everyday speech but is a routine accounting term. These mismatches cause general BERT to misclassify financial sentences consistently. Araci fixed this by continuing BERT's pre-training on 1.8 million financial articles, news and earnings call transcripts, before fine-tuning on the Financial PhraseBank dataset (4,845 financial news sentences annotated by financial professionals as positive, neutral, or negative).

The accuracy gain was clear. On the Financial PhraseBank test set, FinBERT reached 97.42% accuracy against 88.1% for standard BERT fine-tuned on the same task. Gains were largest on negative sentiment, where domain-specific language consistently confused the general model. For any input headline, FinBERT returns three probability scores, namely P(positive), P(neutral), P(negative). The net sentiment score used in FINQUANT-NEXUS is:

    sentiment_score = P(positive) − P(negative)

This gives a value in [-1, +1]. The score is computed per stock per day by aggregating FinBERT outputs across all headlines fetched from Google News RSS, Yahoo Finance News, and Indian RSS feeds (Moneycontrol, Economic Times). SQLite caching avoids repeated inference on already-processed headlines, keeping the live dashboard responsive.

A limitation of this approach is that FinBERT is trained on English text from primarily US and UK financial sources. Indian financial news references different institutions (SEBI instead of SEC, NSE instead of NYSE), covers different event types (NBFC crises, demonetisation, RBI rate cycles), and reflects a different regulatory environment. No publicly available FinBERT variant has been fine-tuned specifically for Indian financial news, so accuracy on Indian market text is likely below the 97.42% benchmark. Beyond that, existing literature mostly uses sentiment scores as standalone features for return prediction or simple trading signals. In this work, the per-stock daily sentiment score is integrated directly into the RL observation vector alongside the 21 technical indicators and the 32-dimensional T-GAT embeddings, so the RL agent learns through training how much weight to give the sentiment signal relative to the other inputs.

---

## 2.5 Federated Learning in Financial Applications

Financial institutions deal with a structural collaboration problem. A fund managing banking equities and a fund managing pharmaceutical equities would both benefit from a shared portfolio model trained on both datasets. But neither can share holdings, trade records, or client data. Privacy obligations, regulatory restrictions, and competitive concerns all block centralised training. Federated learning was proposed as a solution to exactly this class of problem.

**FedAvg, McMahan et al., 2017 [28]**

McMahan and collaborators proposed Federated Averaging (FedAvg) in 2017 as the first practical solution to training shared models on distributed private data. The protocol is straightforward. At the start of each communication round, the central server sends the current global model weights w_t to all participating clients. Each client k receives w_t, runs stochastic gradient descent on its own local dataset D_k for E local epochs, and returns its updated local weights w_{t+1}^k, never the raw training data. The server produces the next global model as a weighted average of all client updates:

    w_{t+1} = Σ over k of  (n_k / n)  *  w_{t+1}^k

This cycle repeats for a fixed number of rounds. On MNIST and a language modelling task, McMahan et al. showed that FedAvg reduces required communication rounds by 10 to 100 times compared to sharing a single gradient step per round, while achieving comparable final model accuracy. No raw data leaves any client.

The critical limitation is the IID assumption built into FedAvg's convergence guarantees. The algorithm assumes each client's data is drawn from the same distribution. When this holds, local gradient updates from all clients point toward the same global optimum and averaging them makes sense. In financial applications across different sectors, this assumption fails badly. A banking sector client's portfolio data has fundamentally different statistical properties from a pharmaceutical sector client's data. Return distributions, volatility levels, correlation structures, and responses to macroeconomic events all differ. When client data is this heterogeneous, local updates diverge during local training, and the server-side average of diverged updates is a poor model for any individual client.

**FedProx, Li et al., 2020 [29]**

Li and collaborators proposed FedProx in 2020 to fix FedAvg's failure on heterogeneous data. [29] The modification is a single additional term in each client's local optimisation objective. Instead of minimising only its own local loss L_k(w), each client minimises:

    h_k(w, w_t) = L_k(w)  +  (μ/2) * ‖w - w_t‖²

where w_t is the current global model received from the server and μ > 0 is the proximal parameter. The quadratic penalty pulls the locally updated model toward the global model throughout local training. Even as the client follows its own local gradient direction, which may differ substantially from other clients, the proximal term prevents it from drifting so far that the resulting update harms other clients. Setting μ = 0 recovers FedAvg exactly. Increasing μ progressively constrains local updates closer to the global starting point.

Li et al. proved convergence for FedProx under non-IID data without requiring the IID assumption FedAvg's analysis depends on. Empirically, on benchmark federated datasets with controlled heterogeneity and on real heterogeneous datasets including Amazon Reviews and non-IID MNIST, FedProx converged more reliably and achieved higher final accuracy than FedAvg, especially when only a fraction of clients participated in each round.

In FINQUANT-NEXUS, the federated system simulates four sector-based clients. These are Banking and Finance (10 stocks, roughly 23% of portfolio weight), IT and Telecom (6 stocks), Pharma and FMCG (8 stocks), and Energy, Auto and Others (20 stocks). The return distributions and volatility profiles of these groups differ substantially, which is exactly the non-IID condition FedProx was designed for. With μ = 0.01 and DP-SGD differential privacy at epsilon = 8.0, the FedProx global model reached a Sharpe Ratio of 0.729 after 50 communication rounds. Three of four sector clients showed individual Sharpe improvement over their locally-trained baselines. The Energy sector client was the exception, showing a small decline consistent with FedProx's known difficulty when one client's distribution diverges substantially from the global average.

---

## 2.6 Monte Carlo Methods in Risk Management

Monte Carlo simulation generates a large number of random return scenarios, runs the portfolio forward through each, and measures the resulting outcome distribution. Two metrics dominate portfolio risk reporting. Value at Risk (VaR 95%) is the loss threshold exceeded only by the worst 5% of scenarios. Conditional Value at Risk (CVaR 95%), also called Expected Shortfall, is the average loss across those worst 5%. CVaR is more informative than VaR because it measures tail severity rather than just the threshold. Rockafellar and Uryasev (2002) [7] showed that CVaR is a coherent risk measure, making it preferable for optimisation and regulatory purposes.

Stress testing extends standard Monte Carlo by calibrating parameters to historical crisis periods rather than long-run averages. FINQUANT-NEXUS implements 1,000 simulation paths under eight scenarios. These are the 2008 Financial Crisis, COVID-19 Crash, Dot-Com 2000, India Bear Market 2015 to 2016, NBFC Crisis 2018, Demonetisation Shock 2016, Rate Hike Cycle 2022, and Flash Crash. Each scenario is calibrated to the actual Indian or global market behaviour of that period. This Indian-market-specific scenario library fills a gap in existing literature, where most published work calibrates only to US or European crisis data.

---

## 2.7 Research Gap and Contribution

Looking across all six areas reviewed above, the pattern is consistent. Each area has made real and lasting progress. Portfolio theory has Markowitz's efficient frontier and the Sharpe ratio as a well-established evaluation framework. Reinforcement learning for finance has PPO and FinRL as practical, working tools. Graph neural networks have GAT as a strong, flexible architecture for relational learning. Financial sentiment analysis has FinBERT as a domain-adapted transformer that substantially outperforms general classifiers. Federated learning has FedAvg as the baseline protocol and FedProx as a more reliable extension for heterogeneous data. Monte Carlo simulation has VaR and CVaR as industry-standard risk metrics.

But this progress has happened in separate lanes. FinRL, the closest prior work, uses RL for portfolio allocation without any graph structure or sentiment input. GNN-based stock modelling papers improve prediction accuracy but do not connect graph output to a portfolio allocation policy. FinBERT-based sentiment work produces useful signals but does not feed them into an RL agent's observation space. Federated learning in finance has focused almost entirely on classification tasks like credit scoring and fraud detection, not portfolio weight optimisation. None of this work, individually or combined, targets the Indian NIFTY 50 market.

No existing open system for Indian equity markets integrates all four components, namely RL for portfolio allocation, graph neural networks for relational stock modelling, FinBERT-based sentiment analysis, and privacy-preserving federated learning, in a single platform with stress testing and an interactive dashboard. That specific integration gap is what FINQUANT-NEXUS fills.

The concrete contributions of this dissertation are:

1. A Temporal Graph Attention Network with separate attention layers per edge type (sector membership, supply chain, return correlation) and a GRU temporal encoder, producing 32-dimensional stock embeddings for 44 NIFTY 50 equities. This is the first multi-relational GNN applied to Indian equities.

2. An ensemble of five RL algorithms trained in a shared Gymnasium environment on NIFTY 50 data, with a Sharpe-based reward and hard risk constraints, evaluated on completely held-out 2024 to 2025 test data.

3. A FinBERT sentiment pipeline integrated directly into the RL observation space, covering all 44 NIFTY 50 stocks daily with SQLite caching for live dashboard use.

4. A FedProx federated training system with DP-SGD (epsilon = 8.0) across four sector-based NIFTY 50 clients, showing that cross-sector knowledge transfer under differential privacy is feasible for portfolio optimisation.

5. A complete research platform with a FastAPI backend and eight-tab React 19 dashboard making all system outputs interactively accessible.

---

## Table 2.1 — Comparison of Related Works

| Author, Year | Primary Method | Market | RL | GNN | Sentiment | FL | Key Limitation vs This Work |
|-------------|---------------|--------|----|----|-----------|----|-----------------------------|
| Markowitz [1], 1952 | Mean-Variance Optimisation | US | No | No | No | No | Static, assumes normality, no adaptation to market regimes |
| Liu et al. [17], 2021 | FinRL, DRL Portfolio | US Markets | Yes | No | No | No | No relational modelling, no sentiment, no FL, no Indian market |
| Velickovic et al. [20], 2018 | Graph Attention Networks | General | No | Yes | No | No | Single edge type, no temporal component, no portfolio policy |
| Araci [26], 2019 | FinBERT Sentiment | US Financial Text | No | No | Yes | No | No portfolio integration, no Indian financial news fine-tuning |
| McMahan et al. [28], 2017 | FedAvg | General | No | No | No | Yes | Fails on non-IID data, not applied to portfolio optimisation |
| Li et al. [29], 2020 | FedProx | General | No | No | No | Yes | Not applied to Indian equity market or portfolio task |
| **This Work** | **RL + T-GAT + FinBERT + FedProx** | **NIFTY 50** | **Yes** | **Yes (3 edges)** | **Yes** | **Yes** | Simulation only, no live brokerage execution |

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
