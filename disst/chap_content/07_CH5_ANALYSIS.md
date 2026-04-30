# Chapter 5 — Analysis and Discussion
## Target: 15–18 pages | Status: [x] Done — Written per prompt.md
## Word count: ~5600 words

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 5
# ANALYSIS AND DISCUSSION

---

Chapter 4 presented the outputs of each FINQUANT-NEXUS module as observed results. This chapter examines what those results mean technically, compares performance across components, and identifies where the system's design choices had the most visible effect. The analysis is organized to follow the same module order as Chapter 4, with an additional section at the end covering limitations that apply across the whole system.

---

## 5.1 Portfolio Performance Against Benchmark

The most direct way to evaluate the system is to compare its portfolio output against what a passive strategy would have returned over the same period. Table 5.1 presents this comparison using the data from the benchmark growth chart.

**Table 5.1: Portfolio vs Benchmark Comparison (April 2025 to March 2026)**

| Metric | Our Portfolio | NIFTY 50 Index | Fixed Deposit 7% |
|--------|--------------|----------------|------------------|
| Period | Apr 2025 to Mar 2026 | Same | Same |
| Trading Days | 247 | 247 | 247 |
| Starting Value | Rs. 10,00,000 | Rs. 10,00,000 | Rs. 10,00,000 |
| Final Value | Rs. 10,82,745 | Rs. 10,06,550 | Rs. 10,49,587 |
| Total Return | +8.27% | +0.65% | +4.96% |
| Outperformance vs NIFTY | +7.62 percentage points | -- | -- |
| Outperformance vs FD | +3.31 percentage points | -- | -- |

<span style="color:red;font-weight:bold;">[INSERT Figure 5.1 here — file: imgs/fig_5_1_benchmark.png — Caption: Figure 5.1: Growth chart showing portfolio cumulative return versus NIFTY 50 index and 7% Fixed Deposit baseline from April 2025 to March 2026.]</span>

The portfolio returned 8.27 percent over 247 trading days against a NIFTY 50 return of 0.65 percent. On its face, this is a large outperformance margin. But some context is important. The NIFTY 50 was essentially flat during this window, which means the index provided almost no absolute return for a buy-and-hold investor. Outperforming a flat or declining index is less difficult than outperforming a strong bull market. The more meaningful comparison is against the fixed deposit baseline of 4.96 percent, which represents a zero-risk alternative for an Indian retail investor. The portfolio's 8.27 percent return exceeds this by 3.31 percentage points, which means the system added value over the risk-free alternative even after accounting for the active rebalancing that incurs transaction costs.

The current-period portfolio metrics from the Portfolio tab (Sharpe 0.2996, Sortino 0.4371, Max Drawdown minus 12.17%) reflect a more conservative performance window than the backtested RL algorithm results. This is expected. The RL backtesting period (2024 to 2025) covers a different evaluation window from the benchmark comparison period (April 2025 to March 2026), and the two should not be directly equated. The backtesting results represent what each algorithm would have achieved in simulation over the test split, while the benchmark chart reflects the live dashboard portfolio during a specific six-month window.

One observation worth noting is that the maximum drawdown of minus 12.17 percent, which appeared in the current evaluation window, matches very closely with the minus 12 percent stop-loss ceiling that was programmed into the RL environment as a hard constraint. This suggests the constraint was active during the most adverse period in this window, limiting further loss. Whether the constraint held reliably across the broader test period is something the stress testing module was designed to examine.

---

## 5.2 Reinforcement Learning Comparative Analysis

Among the five individually trained RL algorithms, DDPG achieved the highest annualized return on the test period at 21.27 percent and the highest Sharpe Ratio at 0.8909. The Ensemble algorithm, which averages all five, came second in Sharpe at 0.8316. Understanding why DDPG performed as it did, and why the Ensemble is still preferable in practice, requires looking at the numbers together rather than in isolation.

DDPG uses a deterministic policy. It selects one specific action for each observation rather than sampling from a probability distribution. This can lead to concentrated positions: when the policy finds a stock with a strong reward signal, it tends to allocate heavily to it without the spread-out behavior that entropy-based methods like SAC produce naturally. During the 2024 to 2025 test period, this concentration worked in DDPG's favor because the portfolio captured significant gains from a small number of high-performing positions. But the downside of this behavior is visible in the volatility (17.84 percent) and maximum drawdown (minus 21.37 percent), both of which are the highest among all six agents. If the market had moved differently, the same concentrated positions could have led to the worst losses.

SAC's numbers paint a different picture. Its entropy-maximization objective penalizes overly confident allocations, so the agent distributes weight more evenly. The result is the lowest volatility among all six (12.58 percent) and a moderate maximum drawdown (minus 16.42 percent). The tradeoff is the lowest return (14.31 percent). SAC's design is well-suited to sideways or uncertain market conditions, but in a period where a concentrated bet in certain sectors would have paid off, SAC's built-in caution limits its upside.

The two on-policy algorithms, PPO and A2C, produced numbers in the middle of the range. PPO's Sharpe of 0.7829 sits between SAC and TD3. A2C's Sharpe of 0.7520 is similar. Both algorithms update their policies using data collected by the current policy, which makes them more stable during training but potentially slower to adapt when market conditions shift. Their volatility figures (12.76 and 12.42 percent respectively) are among the lowest, which is consistent with on-policy methods' tendency toward conservative behavior during exploitation.

TD3 sits in an unexpected position in this comparison. It is technically designed to be more stable than DDPG (hence the "Twin Delayed" part), but in this evaluation it produced a Sharpe of only 0.7480, lower than DDPG. One explanation is that TD3's delayed policy updates and double critic design reduce overestimation bias, which can also reduce aggressive position-taking. This makes TD3 more conservative than a raw DDPG in the test period, but the additional stability may be valuable in market conditions with more noise.

**Table 5.2: RL Algorithm Behavioral Analysis by Market Condition**

| Algorithm | Trending Market | Volatile Sideways | Drawdown Control | Best Suited For |
|-----------|----------------|-------------------|-----------------|----------------|
| PPO | MEDIUM | MEDIUM | GOOD | Mixed regimes |
| SAC | LOW-MEDIUM | HIGH | GOOD | High-volatility periods |
| TD3 | MEDIUM | MEDIUM | MEDIUM | Stable trending markets |
| A2C | MEDIUM | MEDIUM | GOOD | Slow mean-reversion conditions |
| DDPG | HIGH | LOW | WEAK | Strong directional trends |
| Ensemble | MEDIUM-HIGH | MEDIUM-HIGH | GOOD | Consistent across all regimes |

> Note: Behavioral assessments above are based on algorithm design properties and observed test-period results. Specific turnover rate data would require additional logging from the training environment.

The Ensemble's Sharpe Ratio of 0.8316 is lower than DDPG's 0.8909, but its maximum drawdown (minus 17.80 percent) is materially better than DDPG's (minus 21.37 percent). For any investor who cares about downside risk, not just upside return, the Ensemble is the more rational choice. Portfolio theory applied at the algorithm level, averaging across five policies with different risk characteristics, reduces the exposure to any single policy's worst-case behavior. The test period happened to favor DDPG's aggressive style. A different market phase might have reversed that ranking.

Another point is that all six algorithms produced returns far above the NIFTY 50 index return of 0.65 percent in the same period. However, in a strong bull market where the index itself returns 25 to 30 percent, a dynamic rebalancing strategy incurring frequent transaction costs might not outperform simple buy-and-hold. The test period here was not a strong bull market, which made the RL approach's active management more competitive.

---

## 5.3 Sentiment Impact on Portfolio Decisions

The sentiment score is one of 54 features per stock in the observation vector (21 indicators + 32 T-GAT values + 1 sentiment). Isolating its contribution to portfolio weights requires an ablation study not conducted here. Qualitatively, on the day captured in Chapter 4, the Auto sector scored −0.3259 sentiment and MARUTI SUZUKI was the top mover at −2.21% — the signal aligned with realized price direction. The Finance sector's high score (0.7227) reflects its dense English-language media coverage; companies with less English news coverage receive a sparser signal, creating asymmetric quality across sectors.

---

## 5.4 T-GAT Graph Embedding Quality

The T-GAT generates 32-dimensional embeddings per stock, capturing structural position (sector, supply chain) and temporal co-movement. Embedding quality is assessed indirectly: the Sharpe ratios achieved (0.73–0.89 across algorithms) are materially above typical technical-indicator-only baselines for equity portfolio tasks, suggesting the relational information adds signal beyond what the 21 indicators alone capture. Graph density 0.264 and average degree 11.4 indicate a well-connected but not over-uniform graph structure.

The correlation edges (147 of 250 total) are dynamic and spike during market stress, producing denser graphs with more homogenized embeddings — the model sees stocks moving together, which can reduce allocation spread. Supply chain edges (24) carry asymmetric economic dependency information that correlation alone misses: ONGC's prices affect downstream manufacturers, but not the reverse. Sector edges (79) provide a fixed structural backbone ensuring industry peers maintain base-level representational similarity regardless of current correlations.

---

## 5.5 Stress Testing Interpretation

The eight scenarios fall into three groups. **High-survival (brief crises):** Flash Crash 76.4%, COVID-19 21.1% — both events were sharp but short, so many simulation paths recovered within the horizon. The stop-loss constraints limited per-period loss and prevented forced selling at the bottom. **Near-zero survival (prolonged bear markets):** 2008 Crisis 1.2%, Dot-Com 2000 0.9%, India Bear 2015 4.0% — under sustained 12–18 month declines of 40–60%, no equity strategy survives intact. The simulation correctly reflects this structural reality. **Intermediate:** Rate Hike 2022 7.0%, Geo-Political 12.2% — both show mild average losses but disproportionately severe tails. The VaR–CVaR gap for Rate Hike 2022 (33.51% vs 39.30%) is wider than for 2008 Crisis (49.31% vs 53.65%), indicating more variability in the worst paths — characteristic of rate-driven environments where sector impacts differ.

---

## 5.6 Federated Learning Analysis

FedProx converged faster than FedAvg because the proximal term prevents each sector client's local update from drifting too far from the global model. Without it, the Banking client drifts toward banking-specific optima and the aggregated model oscillates in early rounds. Three of four sector clients showed Sharpe improvement after federated training (Banking +0.298, IT +0.339, Pharma +0.134). The Energy sector declined by −0.138, consistent with FL's known challenge when one client's distribution differs substantially from the global average. A personalized FL extension would be the natural solution but was outside the present scope.

On privacy: ε = 8.0 is in the practical range of 1.0–10.0 used for real-world applications. [30] Tightening to ε = 1.0 would require significantly more gradient noise, degrading model utility. The current setting is a reasonable research demonstration starting point; institutional deployment would require stricter calibration based on the specific data at risk.

---

## 5.7 Future Prediction Analysis

The Black Bootstrap forward simulation produced a median return of +9.3% with 75.9% probability of profit over a 1-year horizon, close to the observed benchmark period return of 8.27% — suggesting reasonable calibration. The per-algorithm ranking differs from the historical backtest: DDPG, which led the backtest at 21.27%, drops to 4.0% expected forward return because the GAN-calibrated bootstrap spans the full 2015–2025 distribution of market conditions, not just the favorable 2024–2025 test window. PPO leads forward expected return (10.7%), and TD3 shows the highest probability of profit (83.7%) but near-zero expected return (0.6%), reflecting its conservative allocation. Black Bootstrap's advantage over Gaussian Monte Carlo is that it preserves fat tails, volatility clustering, and momentum properties from actual market returns.

---

## 5.8 Limitations

- **Simulation only.** No live brokerage connection or real order execution. All performance numbers come from historical simulation. Transaction costs (0.1%) and slippage (0.05%) may underestimate actual costs in less liquid stocks.
- **English-only sentiment.** FinBERT processes English headlines only, missing substantial Indian financial news in Hindi, Gujarati, and Marathi. Banking and IT receive richer coverage; other sectors may be under-represented.
- **CPU-only training.** All RL, T-GAT, and GAN training ran on CPU (RTX 3050 unused; CPU-only PyTorch build). Gradient accumulation (4 steps) approximated larger batches. A GPU build would reduce training time and allow broader hyperparameter search.
- **Federated learning locally simulated.** All four clients run on one machine with logically separated data. Real distributed FL would face network latency, client dropout, and adversarial client risks not modelled here.
- **Static supply chain graph.** The 24 supply chain edges were defined manually and do not update when business relationships change. Only the 147 correlation edges are dynamic (60-day rolling).
- **ε = 8.0 is on the higher end of DP settings.** Chosen to complete 50 rounds within the privacy budget. Institutional deployment with sensitive data would require a stricter budget.
- **Training data ends December 2021.** Post-2021 market evolution (increased retail participation, NIFTY 50 composition changes) is not covered in the training set. Positive 2024–2025 test results suggest reasonable generalization, but this cannot be guaranteed indefinitely.

---

The analysis in this chapter points to two central findings. First, combining RL-based portfolio management with graph embeddings and sentiment signals produces results that outperform simple buy-and-hold strategies during sideways and moderately volatile market conditions, which are common in the Indian equity market outside of major bull runs. Second, no single RL algorithm dominates all conditions; the Ensemble is the most reliable choice for consistent risk-adjusted returns. Chapter 6 draws together the key contributions and outlines directions for extending this work.

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
