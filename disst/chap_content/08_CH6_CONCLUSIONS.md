# Chapter 6 — Conclusions and Future Work
## Target: 5–7 pages | Status: [x] Done — Written per prompt.md
## Word count: ~1900 words

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 6
# CONCLUSIONS AND FUTURE WORK

---

## 6.1 Summary of Work Done

FINQUANT-NEXUS was built to fill a gap that kept appearing when looking at existing research for Indian equity portfolio management: no open system brought together reinforcement learning for portfolio allocation, graph neural networks for relational stock modelling, transformer-based financial sentiment analysis, and privacy-preserving federated learning into a single working platform. This dissertation covered the full pipeline from raw data ingestion to an eight-tab interactive dashboard, all implemented using publicly available open-source tools.

The data pipeline collected eleven years of daily OHLCV history for 44 NIFTY 50 constituent stocks through the Yahoo Finance API, covering January 2015 to December 2025. From this raw data, 21 technical indicators were computed per stock, normalised using a 252-day rolling Z-score window, and organised into a feature matrix of shape (2,761, 44, 21). A separate sentiment pipeline used the ProsusAI FinBERT model to produce daily sentiment scores in the range of minus 1 to plus 1 for each stock, with SQLite caching to prevent repeated inference on already-processed headlines. Together, the price-derived indicators and news-derived sentiment provided two distinct input channels for the downstream portfolio model.

The stock relationship layer added a third input channel. A multi-relational graph was constructed with 44 stock nodes and 250 edges across three types: 79 sector membership edges, 24 supply chain linkage edges, and 147 rolling correlation edges updated every 60 days. A Temporal Graph Attention Network was trained on this graph to produce 32-dimensional embeddings for each stock, encoding both structural position and temporal return dynamics. These embeddings were concatenated with the 21 indicators and the sentiment score to form the full observation vector for the reinforcement learning environment.

Five RL algorithms were then trained in a Gymnasium-compatible portfolio environment: PPO, SAC, TD3, A2C, and DDPG. Each ran for 500,000 steps on the 2015 to 2021 training data and was evaluated on the completely held-out 2024 to 2025 test period. A sixth Ensemble agent averaged the weight outputs of all five at inference time. The environment enforced hard risk constraints throughout: maximum single-stock position of 12 percent, stop-loss per trade of minus 3 percent, and portfolio maximum drawdown of minus 12 percent. A Monte Carlo stress testing module ran 1,000 simulation paths each under eight historical crisis scenarios, including the 2008 Financial Crisis, COVID-19 Crash, and Dot-Com 2000 collapse, reporting VaR, CVaR, and survival rates for each.

The federated learning module simulated collaborative training across four sector-based clients (Banking and Finance, IT and Telecom, Pharma and FMCG, and Energy, Auto and Others) using the FedProx aggregation algorithm with DP-SGD differential privacy at a budget of epsilon equal to 8.0 over 50 communication rounds. All of this was made accessible through a FastAPI REST backend and a React 19 dashboard with eight interactive pages covering portfolio analytics, RL agent control, sentiment analysis, graph visualisation, stress testing, federated learning, pipeline workflow, and forward prediction.

---

## 6.2 Key Contributions

Five contributions came out of this work:

**1. Multi-relational stock graph for NIFTY 50**

A stock relationship graph with three distinct edge types (sector membership, supply chain linkage, and rolling return correlation) was built for 44 Indian equity stocks. Prior work on graph-based stock modelling for Indian markets typically uses a single edge type. The three-type graph captures structural economic dependencies alongside statistical co-movement, giving the downstream model richer context than a correlation-only graph can provide.

**2. Temporal Graph Attention Network for stock embeddings**

A T-GAT architecture was designed with separate graph attention layers per edge type and a GRU encoder to capture temporal dynamics across the graph. The resulting 32-dimensional embeddings per stock encode both the stock's position in the industrial relationship network and its historical co-movement behaviour with connected stocks.

**3. Multi-algorithm RL ensemble for portfolio allocation**

Five distinct reinforcement learning algorithms were trained and evaluated in a common Gymnasium-compatible environment with identical observation space, action space, and risk constraints. The Ensemble agent, which averages the outputs of all five, consistently gave a better Sharpe-to-drawdown balance than any individual algorithm. This shows the benefit of diversification at the algorithm level, not just the asset level.

**4. Sector-based federated learning with differential privacy**

A FedProx federated training setup with four sector-based clients was implemented for the Indian equity market context. DP-SGD provided formal differential privacy guarantees during client-to-server weight sharing. Three of the four sector clients showed Sharpe Ratio improvement after federated training, confirming that cross-sector knowledge transfer under privacy constraints is achievable for portfolio optimisation.

**5. Integrated research and demonstration platform**

FINQUANT-NEXUS brings together all four components in a single system with a working API and interactive dashboard. The eight-tab React interface makes results from quantitative models readable without requiring programming expertise, which is an important practical contribution for research demonstration and financial literacy education in the Indian context.

---

## 6.3 Conclusions

The test period results indicate that RL-based portfolio management, when combined with graph embeddings and sentiment signals alongside standard technical indicators, can produce risk-adjusted returns meaningfully above what passive NIFTY 50 index tracking would have returned over the same period. The Ensemble agent achieved an annualised Sharpe Ratio of 0.8316 on the 2024 to 2025 test data, with a return of 16.75 percent and maximum drawdown of minus 17.80 percent. Against the NIFTY 50 index return of 0.65 percent for the benchmark evaluation window, the portfolio returned 8.27 percent, outperforming both the index and a 7 percent fixed deposit baseline. These numbers do not prove the system is ready for live deployment, but they do confirm that integrating multiple ML components adds measurable value over simple buy-and-hold in the tested conditions.

The federated learning component shows that sector-based collaborative model training is achievable with reasonable privacy costs. FedProx aggregation with DP-SGD at epsilon equal to 8.0 allowed the global model to reach a Sharpe Ratio of 0.729 across 50 rounds, with three of four sector clients improving their individual model quality through participation. The Energy sector client was the exception, showing a small decline, which points to a known limitation of standard federated averaging when one client's data distribution differs a great deal from the global average. A real-world implementation between Indian financial institutions would need to account for this sector heterogeneity, possibly through personalised federated learning extensions.

The stress testing results put the portfolio's behaviour in crisis scenarios in proper perspective. Under the Flash Crash scenario, 76.4 percent of the 1,000 simulation paths ended positively because the crash was brief and stop-loss constraints limited exposure. Under prolonged systemic crises like 2008 and the Dot-Com collapse, survival rates fell to 1.2 and 0.9 percent respectively. This is not a failure of the RL strategy specifically. It reflects the reality that no equity portfolio strategy survives a 40 to 60 percent sustained market decline intact. The value of the stress testing module is in making this fact visible and measurable, rather than leaving it as an implicit assumption.

Taken together, this work fills the specific gap identified in the Chapter 2 literature review: the absence of a single open system for Indian equity markets that integrates reinforcement learning, graph neural networks, financial sentiment analysis, and federated learning in one platform with an interpretable interface. The system as built is a research platform, not a commercial product. But it demonstrates what is technically achievable within the open-source ecosystem on modest hardware, and the modular design means individual components can be improved or replaced without rebuilding the whole system.

---

## 6.4 Future Work

Several concrete directions would extend this work in meaningful ways.

**Live brokerage integration.** The most natural next step is connecting the RL portfolio agent to a live brokerage API such as Zerodha Kite or Upstox. This would require real-time order management, slippage handling, and position sizing that accounts for actual market liquidity. The current simulation framework could serve as the training and backtesting environment, with a separate deployment layer for live execution. SEBI regulations around algorithmic trading would also need to be factored into the implementation plan.

**Multilingual sentiment coverage.** The current FinBERT implementation processes English headlines only, which creates uneven signal quality across sectors. Incorporating Hindi and other regional language financial news using multilingual models such as IndicBERT, or a FinBERT variant fine-tuned on a Hindi financial news corpus, would improve sentiment signal quality for companies whose primary news coverage is not in English. This matters most for FMCG, energy, and manufacturing stocks with large domestic retail investor bases.

**Dynamic supply chain graph.** The 24 supply chain edges in the current graph were manually defined and remain fixed. Connecting the graph to structured corporate filings data or supply chain databases would allow automatic edge updates when business relationships change. If a steel manufacturer shifts its primary customer base or a pharmaceutical company changes raw material suppliers, the graph should reflect this without manual intervention.

**Market regime-aware algorithm switching.** The Ensemble agent averages the five algorithms with equal weights at all times. A hidden Markov model or clustering layer trained to detect the current market regime, trending, volatile, or sideways, could make this averaging dynamic: giving more weight to SAC during high-volatility periods and more weight to PPO or DDPG during trending conditions. This would directly address the finding in Chapter 5 that no single algorithm dominates all market phases.

**Broader stock universe.** Extending coverage from the current 44 NIFTY 50 stocks to the BSE 200 would allow the system to access mid-cap and small-cap stocks that sometimes offer higher returns than large-cap index constituents. This would require scaling the T-GAT graph to 200 nodes, adding substantially more supply chain edges, and retraining all RL agents on the expanded observation space. Computational resources would need to scale accordingly, which is another reason a GPU-enabled training setup would be valuable in future iterations.

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
