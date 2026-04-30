# Chapter 2 — Literature Review
## Target: 15–20 pages | Status: [x] Done — Topic-wise, focused (6–7 key papers in depth)
## Word count: ~5000 words (~17 pages at 300 words/page)

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 2
# LITERATURE REVIEW

---

This chapter reviews the existing research across six technical areas that underpin FINQUANT-NEXUS: classical portfolio optimisation, deep reinforcement learning applied to finance, graph neural networks for stock market modelling, financial sentiment analysis, federated learning in financial systems, and Monte Carlo methods for risk management. Within each area, the focus stays on the one or two papers whose methods or findings are most directly relevant to the design choices made in this dissertation. Supporting work is referenced where it provides necessary context. The chapter closes with a research gap analysis and a comparison table.

---

## 2.1 Classical Portfolio Optimisation

The mathematical basis for portfolio construction as it is understood today comes from the work of Harry Markowitz, published in 1952. [1] Before this paper, diversification was standard advice in investment practice, but it rested entirely on intuition. Practitioners knew that holding multiple assets was safer than holding one, but could not say precisely how much safer, or how to find the best possible combination. Markowitz's contribution was to show that portfolio selection is a constrained mathematical optimisation problem, and to give it a precise formulation.

The core insight is that portfolio risk, measured as variance of returns, is not simply the weighted average of the individual asset variances. It depends on how each asset's returns move relative to every other asset in the portfolio. When two assets have returns that tend to move in opposite directions, holding both reduces total portfolio variance below what either asset produces alone. Markowitz formalised this through the covariance matrix: given a set of N assets, each with an expected return and pairwise covariances with all other assets, the minimum-variance portfolio for any target return level can be found by solving a quadratic program. Solving this across all feasible return targets traces the efficient frontier — the set of portfolios that cannot be improved upon without accepting greater risk.

The practical impact of this work was significant. It gave diversification a mathematical proof rather than an intuition, and it defined the trade-off between risk and return in terms of computable quantities. The Sharpe ratio — which measures return per unit of risk — and the entire field of quantitative portfolio management trace directly to this 1952 formulation.

For real Indian equity markets, however, the Markowitz framework encounters three hard limitations that motivated the design of FINQUANT-NEXUS. The first is the distributional assumption. The framework assumes that asset returns are normally distributed, so that mean and variance fully describe the return distribution. NIFTY 50 returns do not follow a normal distribution. Single-day drops of 5–10% driven by RBI policy decisions, SEBI regulatory changes, or global risk-off events occur far more frequently than a normal distribution predicts. These tail events are exactly the ones that matter most for risk management, and the Markowitz framework cannot account for them properly.

The second limitation is input sensitivity. For 44 stocks, the covariance matrix requires estimating 946 unique pairwise values from historical data. Small estimation errors in these inputs compound into large shifts in the computed optimal portfolio weights. A 1% error in one correlation estimate can move optimal weights by 5% or more in a concentrated portfolio, making the output unreliable in practice.

The third and most critical limitation is stationarity. The efficient frontier is computed once using fixed historical estimates of returns and covariances. It does not update as market conditions change. During the 2020 COVID crash and the 2022 rate hike cycle, correlations between NIFTY 50 stocks shifted dramatically. Stocks that moved independently under normal conditions suddenly moved together in the sell-off. A static allocation computed from pre-crisis covariance estimates had no way to respond to this change.

FINQUANT-NEXUS addresses all three limitations. The RL environment learns allocation policies from market interactions rather than requiring a pre-specified return and covariance model, so it adapts to non-stationarity. The T-GAT graph model captures structural relationships between stocks — sector membership, supply chain links — that covariance matrices miss entirely. The stress testing module explicitly constructs non-normal tail scenarios calibrated to historical Indian and global crises. The Sharpe ratio remains the central evaluation metric and reward signal, keeping the connection to Markowitz's risk-adjusted return framework while removing its restrictive assumptions.

---

## 2.2 Deep Reinforcement Learning in Finance

Reinforcement learning frames portfolio management as a sequential decision problem rather than a one-shot optimisation. An agent observes a state representation of the market at each time step, selects a portfolio weight allocation as its action, and receives a reward based on the resulting portfolio performance. Through repeated interaction with historical data, the agent learns a policy — a mapping from states to actions — that maximises cumulative risk-adjusted return. This approach does not require a pre-specified model of expected returns or covariances. It discovers what works by trial and error on actual market data.

**Proximal Policy Optimization (PPO) — Schulman et al., 2017 [13]**

The most practically important contribution to stable policy gradient training came from Schulman and collaborators in 2017. Policy gradient methods had a known instability problem: gradient steps in the wrong direction could collapse a well-trained policy in a single update because there was no constraint on how large a step could be. Trust Region Policy Optimization (TRPO) imposed a hard constraint on policy change per update, which stabilised training but required expensive second-order optimisation that was difficult to implement and tune.

PPO solved the same problem with a much simpler mechanism. It introduces a clipped surrogate objective: for each gradient step, a ratio r_t measures how much the new policy differs from the old policy for each sampled action. The objective clips this ratio to the range [1 − ε, 1 + ε], preventing the policy from moving too aggressively in either direction:

    L_CLIP = E[ min( r_t(θ) · Â_t ,  clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]

where Â_t is the advantage estimate at time step t and ε is typically set to 0.2. The minimum operation ensures that if the unclipped objective would encourage a very large update, the clipped version limits it. This achieves training stability comparable to TRPO without second-order methods, making PPO significantly faster and easier to apply. On benchmark continuous control tasks in MuJoCo, PPO matched or exceeded the performance of TRPO, A3C, and DDPG while requiring less computation and simpler code. It became the standard choice for continuous-action RL applications from 2017 onward.

In FINQUANT-NEXUS, PPO is one of five RL algorithms trained in the Gymnasium-compatible portfolio environment. Its on-policy nature — it can only learn from data collected by the current policy version — makes it sample-efficient compared to replay-buffer methods, which is relevant when training data is limited to 1,757 daily sessions from 2015 to 2021. PPO achieved a Sharpe Ratio of 0.7829 and a total return of 15.22% on the 2024–2025 held-out test period. Its conservative on-policy behavior provides a stable baseline against which the off-policy algorithms (SAC, DDPG, TD3) can be compared.

**FinRL — Liu et al., 2021 [17]**

The most directly related prior work to FINQUANT-NEXUS is the FinRL library published by Liu and collaborators in 2021. FinRL was the first standardised, open-source framework for applying deep reinforcement learning to financial portfolio management. Its design wraps real market data inside a Gymnasium-compatible environment, implements the same five RL algorithms used in FINQUANT-NEXUS (PPO, SAC, TD3, A2C, DDPG) through Stable-Baselines3, and provides pre-built pipelines for portfolio allocation tasks. On the Dow Jones 30 portfolio allocation problem, with training data from 2009 to 2020 and test data covering the COVID-19 crash period, FinRL's ensemble agent achieved an annualised Sharpe Ratio of approximately 0.98, outperforming passive buy-and-hold and classical minimum-variance strategies.

The limitations of FinRL define what FINQUANT-NEXUS needed to add. FinRL's observation space contains only price-derived technical indicators. Each stock is treated as an independent time series — there is no representation of how stocks relate to each other structurally, through sector membership or supply chain links. There is no sentiment module. There is no federated learning component. And FinRL is built and benchmarked entirely on US equity markets, with no support for NIFTY 50 stocks or Indian market-specific events. FINQUANT-NEXUS is best understood as extending the FinRL concept into a richer, Indian-market setting: same environment structure and algorithm set, but with T-GAT graph embeddings and FinBERT sentiment scores added to the observation space, a new federated training framework for sector-based clients, and a full NIFTY 50 dataset covering 2015 to 2025.

---

## 2.3 Graph Neural Networks for Stock Market Modelling

Standard time series models treat each stock as an independent series. This misses a fundamental aspect of equity markets: stocks are not independent. Companies in the same sector respond to the same regulatory and macroeconomic events. Supply chain relationships create directed dependencies between companies in different sectors. Return correlations reflect clusters of stocks that behave similarly during market stress. Incorporating this relational structure requires a way to model graphs — networks where nodes are stocks and edges represent relationships between them.

**Graph Attention Networks — Velickovic et al., 2018 [20]**

The central paper in this area for FINQUANT-NEXUS is the Graph Attention Network proposed by Velickovic and collaborators in 2018. Earlier graph neural network methods — particularly the Graph Convolutional Network of Kipf and Welling (2017) — updated each node's representation by aggregating its neighbours' features with equal weight. Equal-weight aggregation is appropriate when all connections are equally informative, but this fails in stock graphs where some relationships carry far more signal than others. The correlation between HDFCBANK and ICICIBANK during a banking sector event is qualitatively different in importance from the correlation between either of them and a consumer goods stock.

GAT replaces fixed equal weights with learned attention coefficients. For each edge (i, j) in the graph, the model computes a scalar attention score e_ij by applying a shared learnable weight vector a to the concatenation of the transformed feature vectors of nodes i and j:

    e_ij = LeakyReLU( a^T [ W · h_i  ||  W · h_j ] )

These raw scores are normalised across all neighbours of node i using softmax:

    α_ij = exp(e_ij) / Σ_{k ∈ N_i} exp(e_ik)

The updated representation of node i is then the weighted combination of its neighbours' transformed features:

    h'_i = σ ( Σ_{j ∈ N_i} α_ij · W · h_j )

Multi-head attention — K independent attention mechanisms whose outputs are concatenated or averaged — stabilises the learning process and increases the model's expressive capacity. On the Cora, Citeseer, and PPI node classification benchmarks, GAT outperformed GCN and several other graph methods. On the PPI dataset, GAT achieved 97.3% micro-F1 against 88.1% for GraphSAGE, a substantial improvement on a complex multi-label classification task.

Two limitations are directly relevant to the stock market use case. First, the original GAT handles only a single edge type. A stock graph naturally contains multiple structurally different relationship types — sector co-membership, supply chain links, and statistical return correlations — that carry different economic meanings. Treating them as a single homogeneous edge set discards information that separates fundamentally different kinds of stock dependency. Second, standard GAT has no temporal component. Stock relationships and node features evolve over time, and a single static attention layer cannot capture how the relative importance of one stock's signal for another changes across different market regimes.

The T-GAT model in FINQUANT-NEXUS addresses both limitations directly. Separate graph attention layers are applied for each of the three edge types — sector membership (79 edges), supply chain linkage (24 edges), and 60-day rolling return correlation (147 edges) — allowing the model to learn distinct attention patterns for structurally different connections. A GRU temporal encoder then processes the sequence of per-timestep node representations to capture how relational importance evolves over time. The output is a 32-dimensional embedding per stock per time step that encodes both structural position in the three-layer relationship graph and temporal co-movement dynamics. These embeddings form one of three components of the RL observation vector.

---

## 2.4 Financial Sentiment Analysis

Price data and technical indicators capture what has already happened in the market. A large share of price-relevant information, however, arrives as text — earnings announcements, central bank statements, sector-specific regulatory changes, corporate governance disclosures — before it appears in price movements. Financial sentiment analysis attempts to quantify the directional signal in this text.

**FinBERT — Araci, 2019 [26]**

The most important contribution to financial sentiment analysis in recent years is FinBERT, proposed by Araci in 2019, and the specific implementation used in FINQUANT-NEXUS is the ProsusAI/finbert variant hosted on HuggingFace. The foundational model is BERT (Devlin et al., 2019), a bidirectional transformer pre-trained on 3.3 billion words. BERT processes text bidirectionally — every word's representation is conditioned on both its left and right context simultaneously — which gives it substantially better understanding of sentence meaning than unidirectional language models.

General BERT performs poorly on financial text because the pre-training corpus (Wikipedia and BooksCorpus) creates associations between words and sentiment that do not carry over to financial language. The word "liability" is semantically negative in general English but neutral in a balance sheet context. "Outstanding shares" sounds positive in everyday language but is a neutral accounting term. These mismatches consistently cause general BERT to misclassify financial sentences. Araci addressed this by continuing BERT's pre-training on a 1.8-million article financial corpus — financial news and earnings call transcripts — before fine-tuning on the Financial PhraseBank dataset, which contains 4,845 financial news sentences annotated by financial professionals as positive, neutral, or negative.

The result was a substantial accuracy improvement. On the Financial PhraseBank test set, FinBERT achieved 97.42% accuracy compared to 88.1% for standard BERT fine-tuned directly on the same task. The gains were largest on negative sentiment classification, where domain-specific language consistently misled the general model. For any given input headline, FinBERT returns three probability scores: P(positive), P(neutral), P(negative). The net sentiment score used in FINQUANT-NEXUS is:

    sentiment_score = P(positive) − P(negative)

This produces a continuous value in [−1, +1]. The score is computed per stock per day by aggregating FinBERT outputs across all headlines fetched from Google News RSS, Yahoo Finance News, and Indian RSS feeds (Moneycontrol, Economic Times). SQLite caching prevents repeated inference on headlines already processed, keeping the live dashboard responsive.

Two limitations are relevant to this dissertation. FinBERT is trained on English text from primarily US and UK financial sources. Indian financial news uses different institutional references (SEBI instead of SEC, NSE instead of NYSE), covers different event types (NBFC crises, demonetisation, RBI rate cycles), and reflects a different regulatory environment. There is no publicly available FinBERT variant fine-tuned specifically for Indian financial news, so the accuracy on Indian market text is likely below the 97.42% benchmark on Western financial corpora. Second, existing literature uses sentiment scores primarily as standalone features for return prediction or simple trading signals. FINQUANT-NEXUS integrates the per-stock daily sentiment score directly into the RL observation vector alongside the 21 technical indicators and the 32-dimensional T-GAT embeddings, so the RL agent can learn — through interaction with the training data — when and how much weight to give the sentiment signal relative to the other features.

---

## 2.5 Federated Learning in Financial Applications

Financial institutions deal with a structural collaboration problem. A fund managing banking sector equities and a fund managing pharmaceutical sector equities would both benefit from a shared portfolio model trained on both datasets. But neither can share its historical holdings, trade records, or client data with the other, because of privacy obligations, regulatory restrictions, and competitive concerns. Federated learning was proposed as a solution to exactly this class of problem.

**FedAvg — McMahan et al., 2017 [28]**

McMahan and collaborators proposed the Federated Averaging (FedAvg) algorithm in 2017 as the first practical solution to training shared models on distributed private data. The protocol is straightforward. At the start of each communication round, the central server sends the current global model weights w_t to a set of participating clients. Each client k receives w_t, runs stochastic gradient descent on its own local dataset D_k for E local epochs, and returns its updated local weights w_{t+1}^k — never the raw training data. The server produces the next global model by computing a weighted average of all client updates, weighted by dataset size:

    w_{t+1} = Σ_k  (n_k / n)  ·  w_{t+1}^k

This cycle repeats for a fixed number of rounds. On MNIST and a language modelling task, McMahan et al. showed that FedAvg reduces the number of required communication rounds by 10 to 100 times compared to the baseline of sharing a single gradient step per round, while achieving comparable final model accuracy. No raw data leaves any client's local system.

The critical limitation is the IID assumption embedded in FedAvg's convergence guarantees. The algorithm is designed and analysed under the assumption that each client's data is drawn from the same distribution. When this holds, the local gradient updates from all clients point toward the same global optimum, and averaging them produces a meaningful global update. In financial applications across different sectors, this assumption fails badly. A banking sector client's portfolio data has fundamentally different statistical properties from a pharmaceutical sector client's data — different return distributions, volatility levels, correlation structures, and responses to macroeconomic events. When client data is this heterogeneous, local updates diverge during the local training phase, and the server-side average of diverged updates is a poor model for any individual client.

**FedProx — Li et al., 2020 [29]**

Li and collaborators proposed FedProx in 2020 to address FedAvg's failure on heterogeneous data directly. [29] The modification is a single additional term in each client's local optimisation objective. Instead of minimising only its own local loss L_k(w), each client minimises:

    h_k(w, w_t) = L_k(w)  +  (μ/2) · ||w − w_t||²

where w_t is the current global model received from the server and μ > 0 is the proximal parameter. The quadratic penalty term (μ/2)||w − w_t||² pulls the locally updated model toward the global model throughout local training. Even as the client follows its own local gradient direction — which may differ substantially from gradients at other clients — the proximal term prevents it from drifting so far from the global model that the resulting update is harmful to other clients. Setting μ = 0 recovers FedAvg exactly; increasing μ progressively constrains local updates closer to the global starting point.

Li et al. proved convergence for FedProx under non-IID data conditions without requiring the IID assumption that FedAvg's analysis depends on. Empirically, on benchmark federated datasets with controlled heterogeneity and on real heterogeneous datasets including Amazon Reviews (text classification) and non-IID MNIST, FedProx converged more reliably and achieved higher final accuracy than FedAvg, especially in low-participation rounds where only a fraction of clients update.

In FINQUANT-NEXUS, the federated system simulates four sector-based clients: Banking and Finance (10 stocks, ~23% of portfolio weight), IT and Telecom (6 stocks), Pharma and FMCG (8 stocks), and Energy, Auto and Others (20 stocks). The return distributions and volatility profiles of these sector groups are substantially different, which is exactly the non-IID condition FedProx was designed for. With μ = 0.01 and DP-SGD differential privacy at ε = 8.0, the FedProx global model reached a Sharpe Ratio of 0.729 after 50 communication rounds, with three of four sector clients showing individual Sharpe improvement over their locally-trained baselines. The Energy sector client was the exception — a small decline consistent with FedProx's known limitation when one client's data distribution diverges significantly from the global average.

---

## 2.6 Monte Carlo Methods in Risk Management

Monte Carlo simulation generates a large number of random return scenarios, runs the portfolio forward through each, and measures the resulting outcome distribution. Two metrics dominate portfolio risk reporting. Value at Risk (VaR 95%) is the loss threshold exceeded by only the worst 5% of scenarios. Conditional Value at Risk (CVaR 95%), also called Expected Shortfall, is the average loss across those worst 5% — more informative than VaR because it characterises tail severity rather than just the threshold. Rockafellar and Uryasev (2002) [7] showed that CVaR is a coherent risk measure, making it preferable for optimisation and regulatory purposes.

Stress testing extends standard Monte Carlo by calibrating parameters to historical crisis periods rather than long-run averages. FINQUANT-NEXUS implements 1,000 simulation paths under eight scenarios: the 2008 Financial Crisis, COVID-19 Crash, Dot-Com 2000, India Bear Market 2015–2016, NBFC Crisis 2018, Demonetisation Shock 2016, Rate Hike Cycle 2022, and Flash Crash. Each scenario is calibrated to the actual Indian or global market behaviour of that period. This Indian-market-specific scenario library addresses a gap in existing literature, where most published work calibrates only to US or European crisis data.

---

## 2.7 Research Gap and Contribution

Looking across all six areas reviewed above, the same pattern appears at every level. Each area has made genuine and significant progress. Portfolio theory has Markowitz's efficient frontier and the Sharpe ratio as a well-established evaluation framework. Reinforcement learning for finance has PPO and FinRL as practical, working tools. Graph neural networks have GAT as a strong, flexible architecture for relational learning. Financial sentiment analysis has FinBERT as a domain-adapted transformer that substantially outperforms general classifiers. Federated learning has FedAvg as the baseline protocol and FedProx as a robust extension for heterogeneous data. Monte Carlo simulation has VaR and CVaR as industry-standard risk metrics.

But this progress has happened in isolation. FinRL, the closest prior work, uses RL for portfolio allocation without any graph structure or sentiment input. GNN-based stock modelling papers demonstrate improved prediction accuracy but do not connect the graph output to a portfolio allocation policy. FinBERT-based sentiment work produces useful signals but does not integrate them into an RL agent's observation space. Federated learning in finance has focused almost entirely on classification tasks — credit scoring and fraud detection — not on portfolio weight optimisation. And none of this work, individually or combined, targets the Indian NIFTY 50 market.

No existing open system for Indian equity markets integrates all four components — reinforcement learning for portfolio allocation, graph neural networks for relational stock modelling, FinBERT-based sentiment analysis, and privacy-preserving federated learning — in a single platform with stress testing and an interactive dashboard. This specific integration gap is what FINQUANT-NEXUS fills.

The concrete contributions of this dissertation are:

1. A Temporal Graph Attention Network with separate attention layers per edge type (sector membership, supply chain, return correlation) and a GRU temporal encoder, producing 32-dimensional stock embeddings for 44 NIFTY 50 equities — the first multi-relational GNN applied to Indian equities.

2. An ensemble of five RL algorithms trained in a shared Gymnasium environment on NIFTY 50 data, with a Sharpe-based reward and hard risk constraints, evaluated on completely held-out 2024–2025 test data.

3. A FinBERT sentiment pipeline integrated directly into the RL observation space, covering all 44 NIFTY 50 stocks daily with SQLite caching for live dashboard use.

4. A FedProx federated training system with DP-SGD (ε = 8.0) across four sector-based NIFTY 50 clients, demonstrating that cross-sector knowledge transfer under differential privacy is feasible for portfolio optimisation.

5. A complete research platform with a FastAPI backend and eight-tab React 19 dashboard making all system outputs interactively accessible.

---

## Table 2.1 — Comparison of Related Works

| Author, Year | Primary Method | Market | RL | GNN | Sentiment | FL | Key Limitation vs This Work |
|-------------|---------------|--------|----|----|-----------|----|-----------------------------|
| Markowitz [1], 1952 | Mean-Variance Optimisation | US | No | No | No | No | Static, assumes normality, no adaptation to market regimes |
| Liu et al. [17], 2021 | FinRL — DRL Portfolio | US Markets | Yes | No | No | No | No relational modelling, no sentiment, no FL, no Indian market |
| Velickovic et al. [20], 2018 | Graph Attention Networks | General | No | Yes | No | No | Single edge type, no temporal component, no portfolio policy |
| Araci [26], 2019 | FinBERT Sentiment | US Financial Text | No | No | Yes | No | No portfolio integration, no Indian financial news fine-tuning |
| McMahan et al. [28], 2017 | FedAvg | General | No | No | No | Yes | Fails on non-IID data, not applied to portfolio optimisation |
| Li et al. [29], 2020 | FedProx | General | No | No | No | Yes | Not applied to Indian equity market or portfolio task |
| **This Work** | **RL + T-GAT + FinBERT + FedProx** | **NIFTY 50** | **Yes** | **Yes (3 edges)** | **Yes** | **Yes** | Simulation only, no live brokerage execution |

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
