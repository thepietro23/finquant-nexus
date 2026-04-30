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

FINQUANT-NEXUS is an AI-powered platform built for equity portfolio management on the NIFTY 50 stock universe. The objective was to combine deep reinforcement learning for portfolio allocation, graph neural networks for inter-stock relationship modelling, FinBERT-based sentiment analysis, and federated learning with differential privacy into a single system.

Daily price data for 44 NIFTY 50 constituent stocks was collected for eleven years through Yahoo Finance. Twenty-one technical indicators were computed per stock with Z-score normalization. A multi-relational stock graph with three edge types (sector, supply chain, and rolling correlation) was constructed. A Temporal Graph Attention Network generated 32-dimensional embeddings per stock, concatenated with indicators and sentiment as the RL observation vector. Six algorithms (PPO, SAC, TD3, A2C, DDPG, and Ensemble) were trained in a Gymnasium-compatible environment. Monte Carlo stress testing was applied across eight historical crisis scenarios. A FedProx federated system with DP-SGD at epsilon = 8.0 was trained over four sector clients for 50 rounds.

The Ensemble agent achieved a Sharpe Ratio of 0.8316 and annualized return of 16.75 percent on the 2024 to 2025 test period. Portfolio return was 8.27 percent over 247 trading days, compared to 0.65 percent for the NIFTY 50 index. The federated model reached a global Sharpe of 0.729. Survival rates under stress testing ranged from 76.4 percent (Flash Crash) to 0.9 percent (Dot-Com 2000).

These results show that a multi-modal AI portfolio system is practically achievable for Indian equity markets and produces measurable risk-adjusted outperformance over passive benchmarks while enabling privacy-preserving collaborative model training.

**Keywords:** Reinforcement Learning, Portfolio Optimisation, Graph Neural Networks, NIFTY 50, Federated Learning, Sentiment Analysis

---

*Reference: DISSERTATION_FORMATTING.md → Section 7*
*Last updated: 2026-04-30*
