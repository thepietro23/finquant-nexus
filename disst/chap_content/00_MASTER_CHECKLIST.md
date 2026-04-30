# MASTER CHECKLIST — Dissertation Submission
## Praveen Pal Rawal | 240031105151008 | MTech DS & ML Sem 4
## Rashtriya Raksha University | Supervisor: Dr. Mayur Makwana

> Reference: DISSERTATION_FORMATTING.md
> Update status as: [ ] = Pending | [x] = Done | [~] = In Progress

---

## PHASE 0 — Preparation (Before Writing)

### Information to Collect
- [x] Confirm exact dissertation title — FINQUANT-NEXUS: AN AI-POWERED PORTFOLIO OPTIMIZATION SYSTEM FOR NIFTY 50
- [x] Dr. Makwana's designation — Assistant Professor
- [x] School name — School of Information Technology Artificial Intelligence and Cyber Security
- [x] Submission month and year — May 2026
- [x] Hardware specs — Intel Core i5-13420H, 16GB RAM, NVIDIA RTX 3050 4GB (CPU-only PyTorch environment)

### Data & Results to Collect From System
- [x] Sharpe Ratios — PPO: 0.7829 | SAC: 0.7288 | TD3: 0.7480 | A2C: 0.7520 | DDPG: 0.8909 | Ensemble: 0.8316
- [x] Total Return % — PPO: +15.22% | SAC: +14.31% | TD3: +14.86% | A2C: +14.52% | DDPG: +21.27% | Ensemble: +16.75%
- [x] Benchmark data — Portfolio: +8.27% | NIFTY 50: +0.65% | FD 7%: +4.96%
- [x] Sortino Ratio per algorithm — PPO: 1.0721 | SAC: 1.0089 | TD3: 1.0212 | A2C: 1.0447 | DDPG: 1.1279 | Ensemble: 1.1086
- [x] Volatility per algorithm — PPO: 12.76% | SAC: 12.58% | TD3: 12.98% | A2C: 12.42% | DDPG: 17.84% | Ensemble: 13.76%
- [x] Max Drawdown per algorithm — PPO: -17.06% | SAC: -16.42% | TD3: -16.14% | A2C: -16.29% | DDPG: -21.37% | Ensemble: -17.80%
- [x] All 8 stress scenarios fully confirmed — see Table 4.5 in 06_CH4_IMPLEMENTATION.md
- [x] FL system — 50 rounds, epsilon=8.0, delta=0.00001, Global Sharpe 0.729, FedProx faster than FedAvg confirmed
- [x] FL client fairness — Banking +0.298, IT +0.339, Pharma +0.134, Energy -0.138
- [x] Graph stats — 44 nodes, 250 edges (Sector 79, Supply Chain 24, Correlation 147), Density 0.264, Avg Degree 11.4
- [x] Risk constraints — Max Position 12%, Stop Loss -3%, Max Drawdown -12%, Transaction Cost 0.1%, Slippage 0.05%
- [x] T-GAT embedding dimension CORRECTED — 32-dim (not 64-dim), updated in all files
- [x] pytest results — 244 passed, 1 failed (config value mismatch: expected 0.07 got 0.05), 1 skipped, 11 warnings
- [x] RL training steps — 500,000 steps per algorithm (PPO/SAC/TD3/A2C/DDPG) confirmed from configs/base.yaml (total_timesteps: 500000). Actual wall-clock time not captured in logs (logs only show test runs of 200 steps). Policy networks: PPO 45,835 params, SAC 117,070 params (confirmed from rl_agent.log).

### Screenshots
- [x] fig_4_2_portfolio.png — Portfolio tab
- [x] fig_4_3_rl_agent_ensemble.png — RL Agent (Ensemble selected)
- [x] fig_4_3_rl_comparison_table.png — RL comparison table
- [x] fig_4_5_sentiment1.png — Sentiment main view
- [x] fig_4_5_sentiment2.png — Sentiment portfolio impact
- [x] fig_4_5_sentiment3.png — Sentiment sector scores
- [x] fig_4_6_graph_all_edges.png — Graph all 3 edges
- [x] fig_4_6_graph_correlation_only.png — Correlation only
- [x] fig_4_7_stress_normal.png — Stress testing
- [x] fig_4_8_federated.png — Federated learning
- [x] fig_4_9_pipeline.png — Pipeline annotated
- [x] fig_4_9_pipeline2.png — Pipeline diagram
- [x] fig_4_10_future_prediction.png — Future Prediction tab
- [x] fig_5_1_benchmark.png — Growth Chart vs NIFTY 50
> ALL SCREENSHOTS COMPLETE

### Architecture Diagrams (for Chapter 3)
- [x] Mermaid source code written for all 5 diagrams — DIAGRAMS_MERMAID.md
- [x] fig_3_1_architecture.png — confirmed in imgs/ folder
- [x] fig_3_4_sentiment_pipeline.png — confirmed in imgs/ folder
- [x] fig_3_6_tgat.png — confirmed in imgs/ folder
- [x] fig_3_7_rl_env.png — confirmed in imgs/ folder
- [x] fig_3_8_fl.png — confirmed in imgs/ folder

### Research Papers to Find (Google Scholar)
- [ ] Markowitz H., "Portfolio Selection," Journal of Finance, 1952
- [ ] Mnih V. et al., "Human-level control through DRL," Nature, 2015
- [ ] Schulman J. et al., "Proximal Policy Optimization," arXiv, 2017
- [ ] Haarnoja T. et al., "Soft Actor-Critic," ICML, 2018
- [ ] Fujimoto S. et al., "TD3," ICML, 2018
- [ ] Velickovic P. et al., "Graph Attention Networks," ICLR, 2018
- [ ] Kipf T. & Welling M., "GCN," ICLR, 2017
- [ ] Araci D., "FinBERT," arXiv, 2019
- [ ] Devlin J. et al., "BERT," NAACL, 2019
- [ ] McMahan B. et al., "FedAvg," AISTATS, 2017
- [ ] Li T. et al., "FedProx," MLSys, 2020
- [ ] Abadi M. et al., "DP-SGD," CCS, 2016
- [ ] Mnih V. et al., "DQN," 2013
- [ ] Silver D. et al., "DDPG," ICML, 2014
- [ ] Mnih V. et al., "A3C/A2C," ICML, 2016
- [ ] Liu X. et al., "FinRL," arXiv, 2020
- [ ] Raffin A. et al., "Stable-Baselines3," JMLR, 2021
- [ ] Fey M. & Lenssen J., "PyTorch Geometric," 2019
- [ ] Beutel D. et al., "Flower (flwr)," 2020
- [ ] yfinance — cite as software/documentation
- [ ] Loughran T. & McDonald B., "Finance-specific sentiment," JoF, 2011
- [ ] Vaswani A. et al., "Attention is All You Need," NeurIPS, 2017
- [ ] Black F. & Litterman R., "Black-Litterman Model," 1992
- [ ] Merton R., "An intertemporal CAPM," 1973
- [ ] Sharpe W., "Capital Asset Pricing Model," 1964
- [ ] Find 20–25 more Indian stock market / NIFTY specific papers

---

## PHASE 1 — Writing Chapters

### Front Matter
- [x] Title Page — filled in 01_FRONT_MATTER.md
- [~] Declaration — template ready, fill with actual title before printing
- [x] Certificate — template complete in 01_FRONT_MATTER.md | physical signature from Dr. Makwana + School Director + RRU seal required at time of printing
- [x] Acknowledgements — written and pasted into 01_FRONT_MATTER.md
- [ ] Dedication — optional

### Chapter Writing
- [x] **CH1 — Introduction** (~8–10 pages) — 03_CH1_INTRODUCTION.md — DONE
  - [x] 1.1 Background of the Work
  - [x] 1.2 Motivation
  - [x] 1.3 Problem Statement
  - [x] 1.4 Objectives of the Work
  - [x] 1.5 Scope of the Work
  > Section 1.6 (Organization of Dissertation) removed — irrelevant filler section

- [x] **CH2 — Literature Review** (~15–20 pages) — 04_CH2_LITERATURE_REVIEW.md — DONE (Topic-wise, 6–7 key papers in depth)
  - [x] 2.1 Classical Portfolio Optimisation — Markowitz (1952) focused
  - [x] 2.2 Deep RL in Finance — PPO (Schulman 2017) + FinRL (Liu 2021) focused
  - [x] 2.3 GNN for Stock Markets — GAT (Velickovic 2018) focused
  - [x] 2.4 Financial Sentiment Analysis — FinBERT (Araci 2019) focused
  - [x] 2.5 Federated Learning — FedAvg (McMahan 2017) + FedProx (Li 2020) focused
  - [x] 2.6 Monte Carlo Methods in Risk Management
  - [x] 2.7 Research Gap and Contribution

- [x] **CH3 — System Design & Methodology** (~28–35 pages) — 05_CH3_METHODOLOGY.md — DONE
  - [x] 3.1 Overall System Architecture
  - [x] 3.2 Dataset Description
  - [x] 3.3 Data Preprocessing
  - [x] 3.4 Feature Engineering (21 indicators)
  - [x] 3.5 Sentiment Analysis Module
  - [x] 3.6 Stock Relationship Graph
  - [x] 3.7 T-GAT Model
  - [x] 3.8 RL Environment
  - [x] 3.9 RL Agents (PPO/SAC/TD3/A2C/DDPG/Ensemble)
  - [x] 3.10 Stress Testing Framework
  - [x] 3.11 Federated Learning System
  - [x] 3.12 REST API Design
  - [x] 3.13 Dashboard Design

- [x] **CH4 — Implementation & Results** (~25–30 pages) — 06_CH4_IMPLEMENTATION.md — DONE
  - [x] 4.1 Development Environment
  - [x] 4.2 Data Pipeline Results
  - [x] 4.3 Portfolio Analytics Dashboard
  - [x] 4.4 RL Training Results
  - [x] 4.5 Sentiment Results
  - [x] 4.6 Graph Visualization Results
  - [x] 4.7 Stress Testing Results
  - [x] 4.8 Federated Learning Results
  - [x] 4.9 Pipeline Workflow Visualization
  - [x] 4.10 Future Prediction Results
  - [x] 4.11 Testing & Validation

- [x] **CH5 — Analysis & Discussion** (~15–18 pages) — 07_CH5_ANALYSIS.md — DONE
  - [x] 5.1 Portfolio Performance vs Benchmark
  - [x] 5.2 RL Algorithm Comparative Analysis
  - [x] 5.3 Sentiment Impact
  - [x] 5.4 T-GAT Embedding Quality
  - [x] 5.5 Stress Testing Interpretation
  - [x] 5.6 Federated Learning Analysis
  - [x] 5.7 Future Prediction Analysis
  - [x] 5.8 Limitations

- [x] **CH6 — Conclusions** (~5–7 pages) — 08_CH6_CONCLUSIONS.md — DONE
  - [x] 6.1 Summary of Work
  - [x] 6.2 Key Contributions
  - [x] 6.3 Conclusions
  - [x] 6.4 Future Work

- [x] **References** (~4–5 pages) — 09_REFERENCES.md — DONE (47 refs, ordered by appearance)
  - [x] Collect all 50–60 references — 47 confirmed legit papers done; 6 Indian-market papers ([41],[43]–[47])
  - [x] Format in RRU superscript style
  - [ ] Cross-check all [REF] placeholders in chapters — pending final pass

- [x] **Abstract** (~1 page, MAX 250 words) — 02_ABSTRACT.md — DONE (exactly 250 words)
  - [x] Write LAST after all chapters done
  - [x] Must include: Objective, Work Done, Results, Conclusions
  - [x] Count words — 250 / 250

- [x] **Appendices** — 10_APPENDICES.md — DONE
  - [x] Appendix A: System Architecture Diagram — red marker for fig_3_1_architecture.png
  - [x] Appendix B: configs/base.yaml (full) — actual file pasted
  - [x] Appendix C: API Endpoint List — 18 endpoints listed
  - [x] Appendix D: Test Results Summary — 244 passed, 1 failed, 1 skipped confirmed
  - [x] Appendix E: List of Abbreviations — complete

---

## PHASE 2 — Formatting (In MS Word / LibreOffice)

### Typography
- [ ] Font: Times New Roman everywhere
- [ ] Body text: 12pt
- [ ] Chapter headings: 14pt Bold
- [ ] Section headings: 12pt Bold
- [ ] Line spacing: 1.5 throughout
- [ ] Paragraph spacing: consistent

### Page Setup
- [ ] Paper: A4
- [ ] Left margin: 38 mm (1.5 inch)
- [ ] Right margin: 25.4 mm (1 inch)
- [ ] Top margin: 25.4 mm (1 inch)
- [ ] Bottom margin: 25.4 mm (1 inch)
- [ ] Print: Single side only

### Page Numbering
- [ ] Front matter: Roman numerals (i, ii, iii...)
- [ ] Body chapters: Arabic numerals starting from Chapter 1 page 1
- [ ] References and Appendices: continues Arabic

### Figures & Tables
- [ ] Every figure has number + caption below
- [ ] Every table has number + title above
- [ ] List of Figures page complete
- [ ] List of Tables page complete
- [ ] All figures clear and readable (min 300 DPI)

### References
- [ ] All in-text citations are superscript numbers
- [ ] Reference list matches all in-text citations
- [ ] RRU citation format used

---

## PHASE 3 — Review

- [ ] Self-review: read entire dissertation once
- [ ] Check all [REF] placeholders replaced
- [ ] Check all [TO WRITE] placeholders removed
- [ ] Check figure/table numbers are sequential
- [ ] Spell check
- [ ] Grammar check
- [ ] Give draft to Dr. Makwana for review
- [ ] Incorporate supervisor feedback
- [ ] Final proofread

---

## PHASE 4 — Plagiarism Check

- [ ] Submit to Turnitin or iThenticate
- [ ] Review similarity report (target < 10–15%)
- [ ] Rewrite any flagged sections
- [ ] Show report to Dr. Makwana
- [ ] Submit to RRU Central Library for final check
- [ ] Obtain Plagiarism Verification Certificate
- [ ] Keep certificate — goes at back of hardbound

---

## PHASE 5 — Printing & Binding

- [ ] Final PDF ready and approved
- [ ] Print 2 complete copies — A4, 80–90 gsm, single side
- [ ] Get Navy Blue hardbound covers made
- [ ] Cover text in Golden color font
- [ ] Cover: Title, Name, Enrollment, Degree, Supervisor, School, University, Month & Year
- [ ] Paste Plagiarism Verification Certificate at back of each hardbound

---

## PHASE 6 — Digital Submission

- [ ] Export dissertation as PDF
- [ ] File 1: 240031105151008_PraveenPalRawal_Abstract.pdf (Title Page + Certificate + Abstract)
- [ ] File 2: 240031105151008_PraveenPalRawal_Dissertation.pdf (full body)
- [ ] Prepare 1 CD/DVD with both files in correct folder structure
- [ ] Label CD/DVD with name, enrollment, degree
- [ ] Email PDF to School Head

---

## PHASE 7 — Hardbound Submission

- [ ] Submit 1 hardbound to School
- [ ] Submit 1 hardbound to Library
- [ ] Submit CD/DVD
- [ ] Get acknowledgement receipt from both

---

## PHASE 8 — Viva Preparation

- [ ] Prepare PowerPoint (15–20 slides)
- [ ] Revise all 6 chapters
- [ ] Know actual result numbers (Sharpe, VaR, convergence rounds)
- [ ] Bring all hardbound copies to viva hall

---

## Final Page Count Verification

| Section | Target Pages | Actual Pages |
|---------|-------------|--------------|
| Front Matter | 12–14 | ~13 |
| Chapter 1 | 8–10 | ~9 |
| Chapter 2 | 15–20 | ~13 (trimmed) |
| Chapter 3 | 28–35 | ~18 (condensed) |
| Chapter 4 | 25–30 | ~18 (condensed) |
| Chapter 5 | 15–18 | ~10 (condensed) |
| Chapter 6 | 5–7 | ~5 |
| References | 4–5 | ~5 (42 papers) |
| Appendices | 8–10 | ~6 (trimmed) |
| Abstract | 1 | 1 |
| **TOTAL** | | **~98 — UNDER 100 TARGET** |
| RRU LIMIT | 150 | |

---

*Last updated: 2026-04-30*
