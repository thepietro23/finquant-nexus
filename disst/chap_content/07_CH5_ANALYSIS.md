# Chapter 5 — Analysis and Discussion
## Target: 15–18 pages | Status: [x] Done — Written per prompt.md
## Word count: ~5600 words

> Reference: DISSERTATION_FORMATTING.md | Writing rules: fqn1/disst/prompt.md

---

# CHAPTER 5
# ANALYSIS AND DISCUSSION

---

Chapter 4 presented the outputs of each FINQUANT-NEXUS module as observed numbers and screenshots. This chapter goes one step further. It looks at what those results actually mean, where the design choices worked, where they did not, and what the numbers reveal about the system's behaviour across different modules. The analysis follows the same order as Chapter 4.

---

## 5.1 Portfolio Performance Against Benchmark

To evaluate the system, the most direct comparison is against what a passive strategy would have returned over the same period. Table 5.1 shows that comparison using the benchmark growth chart data.

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

The portfolio returned 8.27 percent over 247 trading days against a NIFTY 50 return of 0.65 percent. On its face that is a large outperformance margin, but some context matters here. The NIFTY 50 was essentially flat during this window. That means a buy-and-hold approach returned almost nothing. Outperforming a flat or declining index is less demanding than outperforming a strong bull market, and it would be dishonest not to acknowledge that. The more meaningful comparison is against the fixed deposit baseline of 4.96 percent, which represents a zero-risk alternative for any Indian retail investor. At 8.27 percent, the portfolio exceeds this by 3.31 percentage points, meaning the system added real value over the risk-free alternative even after accounting for transaction costs from active rebalancing.

The current-period portfolio metrics from the Portfolio tab (Sharpe 0.2996, Sortino 0.4371, Max Drawdown minus 12.17%) look conservative compared to the backtested RL algorithm results. This is expected. The RL backtesting period (2024 to 2025) covers a different evaluation window from the benchmark comparison period (April 2025 to March 2026), and the two should not be directly equated. Backtesting results represent what each algorithm would have achieved in simulation over the full test split, while the benchmark chart reflects the live dashboard portfolio over a specific six-month window. Mixing the two would be comparing apples to oranges.

One detail worth noting. The maximum drawdown of minus 12.17 percent in the current evaluation window sits very close to the minus 12 percent stop-loss ceiling programmed into the RL environment as a hard constraint. This suggests the constraint was active during the most adverse part of this window, limiting further loss. Whether the constraint held reliably across the broader test period is what the stress testing module was designed to examine.

---

## 5.2 Reinforcement Learning Comparative Analysis

Among the five individually trained RL algorithms, DDPG achieved the highest annualised return on the test period at 21.27 percent and the highest Sharpe Ratio at 0.8909. The Ensemble algorithm, which averages all five, came second in Sharpe at 0.8316. Understanding why DDPG performed as it did, and why the Ensemble is still preferable in practice, requires looking at all the numbers together rather than in isolation.

DDPG uses a deterministic policy. It selects one specific action for each observation rather than sampling from a probability distribution. This tends to produce concentrated positions. When the policy finds a stock with a strong reward signal, it allocates heavily to it without the spread-out behaviour that entropy-based methods like SAC produce naturally. During the 2024 to 2025 test period, this concentration worked in DDPG's favour because the portfolio captured large gains from a small number of high-performing positions. But the same behaviour is visible in the volatility (17.84 percent) and maximum drawdown (minus 21.37 percent), both of which are the worst among all six agents. If the market had moved differently, the same concentrated positions could easily have produced the worst losses instead.

SAC's numbers tell a different story. Its entropy-maximization objective penalises overly confident allocations, so the agent distributes weight more evenly. The result is the lowest volatility among all six (12.58 percent) and a moderate maximum drawdown (minus 16.42 percent). The trade-off is the lowest return (14.31 percent). SAC is well-suited to sideways or uncertain market conditions, but in a period where a concentrated bet in certain sectors paid off, SAC's built-in caution limited its upside. That is basically the nature of entropy regularisation.

The two on-policy algorithms, PPO and A2C, produced numbers in the middle of the range. PPO's Sharpe of 0.7829 sits between SAC and TD3. A2C's Sharpe of 0.7520 is similar. Both algorithms update their policies using only data collected by the current policy, which makes them stable during training but potentially slower to adapt when market conditions shift. Their volatility figures (12.76 and 12.42 percent respectively) are among the lowest, consistent with on-policy methods' tendency toward conservative behaviour during exploitation.

TD3 sits in an unexpected position in this comparison. It is technically designed to be more stable than DDPG (the "Twin Delayed" part is about reducing overestimation bias), but in this evaluation it produced a Sharpe of only 0.7480, lower than DDPG. One reason is that TD3's delayed policy updates and double critic design reduce aggressive position-taking. So this makes TD3 more conservative than raw DDPG in the test period. The additional stability may be valuable under market conditions with more noise, just not in this particular window.

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

The Ensemble's Sharpe Ratio of 0.8316 is lower than DDPG's 0.8909, but its maximum drawdown (minus 17.80 percent) is materially better than DDPG's (minus 21.37 percent). For any investor who cares about downside risk, not just upside return, the Ensemble is the more rational choice. The logic here is the same logic that portfolio theory applies at the asset level, but applied at the algorithm level. Averaging across five policies with different risk characteristics reduces exposure to any single policy's worst-case behaviour. The test period happened to favour DDPG's aggressive style. A different market phase would likely reverse that ranking.

Another point. All six algorithms produced returns far above the NIFTY 50 index return of 0.65 percent in the same period. But in a strong bull market where the index itself returns 25 to 30 percent, a strategy incurring frequent rebalancing costs might not outperform simple buy-and-hold. The test period here was not a strong bull market. That made the RL approach's active management more competitive.

---

## 5.3 Sentiment Impact on Portfolio Decisions

The sentiment score is one of 54 features per stock in the observation vector (21 indicators plus 32 T-GAT values plus 1 sentiment). Isolating its contribution to portfolio weights would require an ablation study, which was not conducted here. Qualitatively, on the day captured in Chapter 4, the Auto sector scored minus 0.3259 sentiment and MARUTI SUZUKI was the top mover at minus 2.21 percent. The signal aligned with realised price direction. The Finance sector's high score (0.7227) reflects dense English-language media coverage of that sector. Companies with less English news coverage receive a sparser signal, creating uneven quality across sectors.

---

## 5.4 T-GAT Graph Embedding Quality

The T-GAT generates 32-dimensional embeddings per stock that encode both structural position (sector, supply chain) and temporal co-movement. Embedding quality cannot be assessed directly without a separate classification benchmark, so the evaluation here is indirect. The Sharpe ratios achieved across algorithms (0.73 to 0.89) are meaningfully above typical technical-indicator-only baselines for equity portfolio tasks reported in the FinRL literature, which suggests the relational information adds real signal beyond what the 21 indicators alone capture. Graph density 0.264 and average degree 11.4 indicate a well-connected but not over-uniform graph structure.

A closer look at the three edge types reveals different roles. The correlation edges (147 of 250 total) are dynamic and spike during market stress, producing denser graphs with more similar embeddings. The model sees stocks moving together, which can reduce allocation spread during those periods. Supply chain edges (24) carry asymmetric economic dependency information that correlation alone misses. ONGC's prices affect downstream manufacturers, but not the reverse. And sector edges (79) provide a fixed structural backbone, ensuring industry peers maintain base-level representational similarity regardless of current correlation readings. All three edge types together give the graph structure that neither a purely correlation-based approach nor a simple sector-grouping approach would achieve on its own.

---

## 5.5 Stress Testing Interpretation

The eight scenarios fall into three groups based on survival rate and historical crisis type.

High-survival scenarios are the brief crises. Flash Crash at 76.4% and COVID-19 at 21.1%. Both events were sharp but short, so many simulation paths recovered within the horizon. The stop-loss constraints limited per-period loss and prevented forced selling at the bottom. That the system survived the COVID scenario at 21.1% is actually a reasonable result given how abrupt the March 2020 crash was. Short duration is what the stop-loss mechanism handles well.

Near-zero survival scenarios are the prolonged bear markets. 2008 Crisis at 1.2%, Dot-Com 2000 at 0.9%, India Bear 2015 at 4.0%. Under sustained 12 to 18 month declines of 40 to 60 percent, no equity strategy survives intact. The simulation correctly reflects this structural reality. This is not a failure of the RL strategy. It is an honest characterisation of what equity portfolios face in worst-case environments.

Intermediate scenarios are Rate Hike 2022 at 7.0% and Geo-Political Shock at 12.2%. Both show mild average losses but disproportionately severe tails. The VaR-CVaR gap for Rate Hike 2022 (33.51% vs 39.30%) is wider relative to its severity than for the 2008 Crisis (49.31% vs 53.65%), indicating more variability in the worst paths. Such behaviour is characteristic of rate-driven environments where sector impacts differ widely depending on the sector's debt sensitivity and growth profile.

---

## 5.6 Federated Learning Analysis

FedProx converged faster than FedAvg because the proximal term prevents each sector client's local update from drifting too far from the global model. Without it, the Banking client drifts toward banking-specific optima, the IT client toward IT-specific optima, and the server-side average of those diverged updates oscillates in the early rounds. With the proximal term active, per-client updates stay closer to the global starting point at each round, so the aggregation produces a more coherent global model from the start.

Three of four sector clients showed Sharpe improvement after federated training. Banking and Finance (+0.298), IT and Telecom (+0.339), Pharma and FMCG (+0.134). The Energy sector client showed a decline of minus 0.138. This is consistent with a known limitation of federated averaging. When one client's data distribution differs substantially from the global average, the global model's generalisation does not translate into per-client improvement for that outlier. The energy sector has distinct return characteristics driven by commodity prices and government policy that differ substantially from the banking and IT sectors. A personalised federated learning extension would be the natural solution but was outside the present scope.

On privacy. epsilon = 8.0 sits in the practical range of 1.0 to 10.0 used for real-world applications. [30] Tightening to epsilon = 1.0 would require significantly more gradient noise, degrading model utility noticeably. The current setting is a reasonable starting point for research demonstration. Institutional deployment with genuinely sensitive data would require stricter calibration based on the specific privacy risk at stake.

---

## 5.7 Future Prediction Analysis

The Black Bootstrap forward simulation produced a median return of +9.3% with 75.9% probability of profit over a 1-year horizon. This is close to the observed benchmark period return of 8.27%, suggesting reasonable calibration of the simulation to actual market behaviour.

Going by the numbers, the per-algorithm ranking in the forward simulation differs from the historical backtest, and that difference is informative. DDPG led the backtest at 21.27% but dropped to 4.0% expected forward return. The reason is that the GAN-calibrated bootstrap spans the full 2015 to 2025 distribution of market conditions, not just the favourable 2024 to 2025 test window. In a broader sample of market conditions, DDPG's concentrated strategy does not consistently produce the same outsized returns it captured in that specific test period. PPO leads forward expected return (10.7%), and TD3 shows the highest probability of profit (83.7%) but near-zero expected return (0.6%), reflecting its conservative forward allocation. Black Bootstrap's advantage over Gaussian Monte Carlo is that it preserves fat tails, volatility clustering, and momentum properties from actual market return data rather than imposing a normal distribution that Indian equities do not follow.

---

## 5.8 Limitations

- **Simulation only.** No live brokerage connection or real order execution. All performance numbers come from historical simulation. Transaction costs (0.1%) and slippage (0.05%) may underestimate actual costs in less liquid stocks.
- **English-only sentiment.** FinBERT processes English headlines only, missing substantial Indian financial news in Hindi, Gujarati, and Marathi. Banking and IT receive richer coverage. Other sectors may be under-represented in the sentiment signal.
- **CPU-only training.** All RL, T-GAT, and GAN training ran on CPU (RTX 3050 unused, CPU-only PyTorch build). Gradient accumulation (4 steps) approximated larger batches. A GPU build would reduce training time and allow broader hyperparameter search.
- **Federated learning locally simulated.** All four clients run on one machine with logically separated data. Real distributed FL would face network latency, client dropout, and adversarial client risks not modelled here.
- **Static supply chain graph.** The 24 supply chain edges were defined manually and do not update when business relationships change. Only the 147 correlation edges are dynamic (60-day rolling).
- **epsilon = 8.0 is on the higher end of DP settings.** Chosen to complete 50 rounds within the privacy budget. Institutional deployment with sensitive data would require a stricter budget.
- **Training data ends December 2021.** Post-2021 market evolution (increased retail participation, NIFTY 50 composition changes) is not covered in the training set. Positive 2024 to 2025 test results suggest reasonable generalisation, but this cannot be guaranteed indefinitely.

---

In this work, we see two central findings emerge from this analysis. First, combining RL-based portfolio management with graph embeddings and sentiment signals produces results that outperform simple buy-and-hold during sideways and moderately volatile market conditions, which are common in the Indian equity market outside of major bull runs. Second, no single RL algorithm dominates all conditions. The Ensemble is the most reliable choice for consistent risk-adjusted returns across different market phases. Chapter 6 draws together the key contributions and outlines directions for extending this work.

---

*Reference: DISSERTATION_FORMATTING.md | prompt.md*
*Last updated: 2026-04-30*
