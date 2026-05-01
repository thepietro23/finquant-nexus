# FRONT MATTER — All Templates
## Reference: DISSERTATION_FORMATTING.md → Sections 3, 4, 5, 6

> Status: [ ] Pending
> Fill blanks marked with [___] before printing

---

## DOCUMENT 1 — TITLE PAGE (Front Page + Cover Page)
*(Annexure-I — appears twice: outer cover and inner cover)*

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


             FINQUANT-NEXUS: AN AI-POWERED PORTFOLIO
            OPTIMIZATION SYSTEM FOR NIFTY 50


                    Dissertation submitted to

              RASHTRIYA RAKSHA UNIVERSITY
              (An Institution of National Importance)

         For the partial fulfilment for the award of the degree of

      Master of Technology in Data Science and Machine Learning


                         Submitted by

                      PRAVEEN PAL RAWAL
                      (240031105151008)


                    Under the Guidance of

                    DR. MAYUR MAKWANA
                      Assistant Professor,
    School of Information Technology Artificial Intelligence
                       and Cyber Security,
                            Gandhinagar


   SCHOOL OF INFORMATION TECHNOLOGY ARTIFICIAL INTELLIGENCE
                       AND CYBER SECURITY
                  RASHTRIYA RAKSHA UNIVERSITY,
          Lavad, Dehgam, Gandhinagar-382305, Gujarat, India

                           May 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Print notes:**
- Hardbound cover: Navy Blue
- All text on cover: Golden color font
- Same page printed inside as Cover Page

**All details confirmed — ready to print after final content approval.**

---

## DOCUMENT 2 — DECLARATION
*(Annexure-II)*

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                         DECLARATION

The work embodied in this Dissertation titled "FINQUANT-NEXUS:
An AI-Powered Portfolio Optimization System for NIFTY 50" submitted
for the partial fulfillment of the degree of Master of Technology in
Data Science and Machine Learning
is the original research work carried out by me. The research work
does not form the basis for the award of any degree, diploma,
associateship, fellowship or other titles in the Rashtriya Raksha
University or similar institutions of higher learning. All the ideas
and references have been duly acknowledged.




                              (Name & Signature of the Candidate)


Date:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## DOCUMENT 3 — CERTIFICATE
*(Annexure-III — to be signed by Dr. Makwana + School Director)*

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                         CERTIFICATE

This is to certify that the Dissertation titled "FINQUANT-NEXUS:
An AI-Powered Portfolio Optimization System for NIFTY 50" was
carried out by Mr. Praveen Pal Rawal (Enrollment No. 240031105151008)
studying at School of Information Technology Artificial Intelligence
and Cyber Security for partial fulfillment of Master of Technology
in Data Science and Machine Learning degree to be awarded by
Rashtriya Raksha University. This research work has been carried out
under my guidance and supervision and it is up to my satisfaction.
The Dissertation is fit to be considered for evaluation for the
degree of Master of Technology in Data Science and Machine Learning.


Date:
Place: Gandhinagar




Signature and Name of Supervisor      Signature and Name of School Director

    Dr. Mayur Makwana                         [School Director Name]
    Assistant Professor                        [Designation]
    School of IT AI and Cyber Security         School of IT AI and Cyber Security
    Rashtriya Raksha University                Rashtriya Raksha University


                         Round Seal of RRU

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Action needed:**
- [ ] Get this page signed physically by Dr. Makwana
- [ ] Get School Director signature
- [ ] Get RRU round seal affixed

---

## DOCUMENT 4 — DEDICATION
*(Optional — 1 page, center of page)*

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                         DEDICATION

              Dedicated to my family and mentors
              who supported me throughout this journey.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

*(Customize as per your preference — keep it short, 1–3 lines)*

---

## DOCUMENT 5 — ACKNOWLEDGEMENTS
*(Annexure-V — write personally, ~1 page)*

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                      ACKNOWLEDGEMENTS

I would like to thank my dissertation supervisor, Dr. Mayur Makwana,
Assistant Professor, School of Information Technology Artificial
Intelligence and Cyber Security, Rashtriya Raksha University, for
his guidance throughout this work. His feedback helped me keep the
scope realistic and the technical choices grounded. I am grateful
for the time he gave to this project.

I also thank the faculty and staff of the School of IT AI and Cyber
Security at Rashtriya Raksha University for the academic environment
they provide. The program gave me the space to try a project that
combined multiple areas I was curious about, and I am glad for that.

I owe a great deal to my family for their patience during the months
I spent on this. Finishing a dissertation while managing coursework
requires quiet support in the background, and they provided it.

Finally, I am grateful to the open-source community whose work made
this project possible — the teams behind PyTorch, Stable-Baselines3,
HuggingFace Transformers, PyTorch Geometric, Flower, FastAPI, and
React. Without these libraries, none of the implementation described
in this dissertation would have been buildable in a single semester.


                                            Praveen Pal Rawal
                                            240031105151008
                                            May 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## DOCUMENT 6 — TABLE OF CONTENTS
*(Annexure-VI — generate automatically in Word after all chapters done)*

```
TABLE OF CONTENTS

Declaration                                                    ii
Certificate                                                   iii
Dedication                                                     iv
Acknowledgements                                               v
Table of Contents                                              vi
List of Figures                                              viii
List of Tables                                                 ix
Abstract                                                       x

Chapter 1   Introduction                                        1
  1.1       Background of the Work                             1
  1.2       Motivation                                         4
  1.3       Problem Statement                                  5
  1.4       Objectives of the Work                             6
  1.5       Scope of the Work                                  7

Chapter 2   Literature Review                                  9
  2.1       Classical Portfolio Optimisation                   9
  2.2       Deep Reinforcement Learning in Finance            12
  2.3       Graph Neural Networks for Stock Market Modelling  16
  2.4       Financial Sentiment Analysis                      19
  2.5       Federated Learning in Financial Applications      22
  2.6       Monte Carlo Methods in Risk Management            25
  2.7       Research Gap and Contribution                     27

Chapter 3   System Design and Methodology                     29
  3.1       Overall System Architecture                       29
  3.2       Dataset Description                               31
  3.3       Data Preprocessing                                33
  3.4       Feature Engineering — Technical Indicators        35
  3.5       Sentiment Analysis Module                         38
  3.6       Stock Relationship Graph Construction             42
  3.7       Temporal Graph Attention Network (T-GAT)          45
  3.8       Reinforcement Learning Environment                49
  3.9       Reinforcement Learning Agents                     52
  3.10      Stress Testing Framework                          57
  3.11      Federated Learning System                         60
  3.12      REST API Design                                   64
  3.13      Dashboard Design                                  66

Chapter 4   Implementation and Results                        68
  4.1       Development Environment                           68
  4.2       Data Collection and Processing Results            69
  4.3       Portfolio Analytics and Benchmark Comparison      72
  4.4       Reinforcement Learning Training Results           75
  4.5       Sentiment Analysis Results                        80
  4.6       Graph Visualization Results                       83
  4.7       Stress Testing Results                            86
  4.8       Federated Learning Results                        89
  4.9       Pipeline Workflow Visualization                   93
  4.10      Future Prediction Dashboard                       94
  4.11      Testing and Validation                            95

Chapter 5   Analysis and Discussion                           96
  5.1       Portfolio Performance Against Benchmark           96
  5.2       Reinforcement Learning Comparative Analysis       99
  5.3       Sentiment Impact on Portfolio Decisions          103
  5.4       T-GAT Graph Embedding Quality                   106
  5.5       Stress Testing Interpretation                    109
  5.6       Federated Learning Analysis                      112
  5.7       Future Prediction Analysis                       115
  5.8       Limitations                                      116

Chapter 6   Conclusions and Future Work                      117
  6.1       Summary of Work Done                             117
  6.2       Key Contributions                                119
  6.3       Conclusions                                      120
  6.4       Future Work                                      122

Bibliography and References                                  124

Appendix A  System Architecture Diagram                      128
Appendix B  Configuration File (base.yaml)                   129
Appendix C  REST API Endpoint Reference                      133
Appendix D  Test Results Summary                             136
Appendix E  List of Abbreviations                            138
```

> NOTE: Page numbers above are estimates. Update actual page numbers after final formatting in Word.

---

## DOCUMENT 7 — LIST OF FIGURES

# LIST OF FIGURES

| Figure No. | Caption | Page |
|-----------|---------|------|
| Figure 3.1 | Overall System Architecture Diagram | * |
| Figure 3.4 | Sentiment Analysis Pipeline: News Sources to RL Observation Space | * |
| Figure 3.5 | Stock Relationship Graph with All Three Edge Types Enabled | * |
| Figure 3.6 | Temporal Graph Attention Network Architecture | * |
| Figure 3.7 | RL Environment State-Action-Reward Cycle | * |
| Figure 3.8 | Federated Learning System: FedProx vs FedAvg Convergence | * |
| Figure 4.1 | Sample NIFTY 50 Stock Price Chart (January 2015 to December 2025) | * |
| Figure 4.2 | Portfolio Analytics Tab Showing Five Performance Metric Cards | * |
| Figure 4.3 | Growth Chart: Portfolio vs NIFTY 50 Index vs Fixed Deposit | * |
| Figure 4.4 | RL Agent Tab with Ensemble Algorithm Selected | * |
| Figure 4.5 | RL Algorithm Comparison Table from Dashboard | * |
| Figure 4.6 | Sentiment Analysis Tab Showing Live FinBERT Scores | * |
| Figure 4.7 | Graph Visualization Tab with All Three Edge Types Enabled | * |
| Figure 4.8 | Stress Testing Tab Showing Monte Carlo Simulation Fan Chart | * |
| Figure 4.9 | Federated Learning Tab Showing Convergence and Privacy Budget | * |
| Figure 4.10 | Pipeline Workflow Tab Showing 15-Stage Animated Data Flow | * |
| Figure 4.11 | Future Prediction Tab Showing Black Bootstrap Simulation Paths | * |
| Figure 5.1 | Benchmark Growth Chart: Portfolio vs NIFTY 50 vs Fixed Deposit | * |
| Figure A.1 | Complete FINQUANT-NEXUS System Architecture Diagram | * |

> Note: Page numbers marked * are to be updated after final Word assembly and pagination.

---

## DOCUMENT 8 — LIST OF TABLES

# LIST OF TABLES

| Table No. | Title | Page |
|----------|-------|------|
| Table 2.1 | Comparison of Related Works in RL Portfolio Optimisation | * |
| Table 3.1 | Dataset Statistics: NIFTY 50 Constituent Stocks | * |
| Table 3.2 | 21 Technical Indicators: Name, Type, Window, and Purpose | * |
| Table 3.3 | RL Algorithm Hyperparameters | * |
| Table 3.4 | Federated Learning Client Sector Groups | * |
| Table 4.1 | Development Environment Specifications | * |
| Table 4.2 | Dataset Summary Statistics | * |
| Table 4.3 | RL Algorithm Performance Comparison (Test Period 2024 to 2025) | * |
| Table 4.4 | Stock Graph Statistics | * |
| Table 4.5 | Stress Testing Risk Metrics: All Eight Scenarios | * |
| Table 4.6 | Federated Learning Results Summary | * |
| Table 4.7 | Forward Simulation Results per Algorithm (1-Year Horizon) | * |
| Table 4.8 | Test Coverage Summary (12 Test Files) | * |
| Table 5.1 | Portfolio vs Benchmark Comparison (April 2025 to March 2026) | * |
| Table 5.2 | RL Algorithm Behavioral Analysis by Market Condition | * |
| Table 5.3 | Federated Learning Privacy-Utility Summary | * |

> Note: Page numbers marked * are to be updated after final Word assembly and pagination.
Table 5.3   Federated Learning Privacy-Utility Summary                    114
```

---

*Reference: DISSERTATION_FORMATTING.md*
*Last updated: 2026-04-29*
