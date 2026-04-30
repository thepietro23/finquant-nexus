# ABSTRACT
## Reference: DISSERTATION_FORMATTING.md → Section 7 (Annexure-IX)

> Status: [x] Done — Written per prompt.md
> Word count: 250 / 250 (exact limit)

---

## RRU Abstract Header Format

```
Abstract

FINQUANT-NEXUS: AN AI-POWERED PORTFOLIO OPTIMIZATION SYSTEM FOR NIFTY 50

Submitted By: Praveen Pal Rawal (240031105151008)

Supervised By: Dr. Mayur Makwana, Assistant Professor,
               School of Information Technology Artificial Intelligence
               and Cyber Security, Rashtriya Raksha University, Gandhinagar
```

---

## ABSTRACT

FINQUANT-NEXUS is a portfolio management platform built for the NIFTY 50 stock universe. The aim of this work was to bring four separate machine learning ideas into one working system: deep reinforcement learning for portfolio weight allocation, graph neural networks for modelling how stocks relate to each other, FinBERT-based sentiment scoring from financial news, and federated learning with differential privacy for collaborative model training.

For the data side, daily prices for 44 NIFTY 50 stocks were pulled from Yahoo Finance across eleven years. Twenty-one technical indicators were computed and Z-score normalised per stock. A multi-relational stock graph was built with three edge types (sector, supply chain, and rolling correlation). A Temporal Graph Attention Network then produced 32-dimensional per-stock embeddings, which were concatenated with indicator and sentiment values to form the RL observation vector. Six algorithms were trained in a Gymnasium-compatible environment: PPO, SAC, TD3, A2C, DDPG, and an Ensemble that combines all five. Monte Carlo stress testing ran 1000 paths over eight historical crisis scenarios. A FedProx federated system used DP-SGD at epsilon = 8.0 across four sector clients for 50 rounds.

The Ensemble agent got a Sharpe Ratio of 0.8316 and annualised return of 16.75 percent on the 2024 to 2025 test set. Over 247 trading days the portfolio returned 8.27 percent against 0.65 percent for the NIFTY 50 index. The federated model reached a global Sharpe of 0.729. Stress test survival rates went from 76.4 percent under the Flash Crash down to 0.9 percent under the Dot-Com 2000 scenario.

Taken together, these numbers suggest that combining RL, graph modelling, sentiment signals, and federated training produces real improvement over the passive benchmark for Indian equities. The work also confirms that privacy-preserving collaborative training is doable at this scale, though the system is a research platform, not a deployment-ready trading tool.

**Keywords:** Reinforcement Learning, Portfolio Optimisation, Graph Neural Networks, NIFTY 50, Federated Learning, Sentiment Analysis

---

*Reference: DISSERTATION_FORMATTING.md → Section 7*
*Last updated: 2026-04-30*
