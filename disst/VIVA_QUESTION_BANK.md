# FINQUANT-NEXUS v4 — Complete Viva / Interview Question Bank

> **Scope:** Dissertation defense + panel interview + live demo questions  
> **Organized:** Component-wise (Phase 0 → Phase 14 + Cross-cutting)  
> **Total Questions:** 300+  
> **Project:** Self-Optimizing Federated Portfolio Intelligence Platform  
> **Stack:** FastAPI · PyTorch · React/TypeScript · Stable-Baselines3 · FinRL · Flower FL · Qiskit · FinBERT  

---

## HOW TO USE THIS FILE

Har section ek **Phase / Component** ko cover karta hai. Pehle apne components ke concepts padho, fir uske questions dekho. Agar ek question ka jawab nahi aata, wahi section dobara padho.

**Difficulty legend:**
- `[B]` = Basic / Conceptual (definitely aayega)
- `[M]` = Medium / Implementation-level (expect karo)
- `[H]` = Hard / Research-level (panel ke liye)

---

## SECTION 0 — Project Vision & Problem Statement

> **"Kya bana rahe ho aur kyun?"** — Yahi se shuru hoga har interview.

1. `[B]` FINQUANT-NEXUS ka exact objective kya hai, aur existing portfolio tools (Zerodha, Groww, Bloomberg) se ye kaise different hai?
2. `[B]` "Self-optimizing" term use kiya hai — isme self-optimization kis layer par hoti hai? Kaun automatically optimize hota hai?
3. `[B]` Is project ka core research contribution kya hai — RL, GNN, FL, sentiment, DARTS, Quantum, ya inka combination?
4. `[B]` Agar simple Markowitz model already available hai, to is system ki need kyun padi? Markowitz ki limitations kya hain?
5. `[B]` Is platform ka real-world user kaun hai: retail investor, quant analyst, fund manager, ya research team?
6. `[B]` Is system ka primary success metric kya hai — return, risk, stability, ya adaptability?
7. `[M]` Aapne NIFTY 50 hi kyun choose kiya, S&P 500 ya BSE 500 kyun nahi?
8. `[M]` Is project me novelty exactly kya hai jo sirf "dashboard project" se alag banata hai?
9. `[B]` Agar ek line me batana ho ki project kya solve karta hai, to kya bolenge?
10. `[M]` Is system ka biggest practical use-case kya hai?
11. `[H]` Agar professor bole ki "yeh sab models together too complex hain for production," to aap kya justify karoge?
12. `[M]` Project me "v4" kya suggest karta hai — kya previous versions the? Unme kya evolve hua?
13. `[H]` What is the mathematical objective of the entire pipeline — ek equation me likho to kya hogi?
14. `[M]` Agar aapko sirf ek component rakhna ho to kaunsa rakhenge aur kyun?
15. `[H]` Kaunsa component sabse zyada incremental value add karta hai aur kaise prove karoge?

---

## SECTION 1 — Data Pipeline & Feature Engineering (Phase 1–2)

> **Files:** `src/data/download.py`, `features.py`, `quality.py`, `stocks.py`, `live.py`

### Data Download & Quality

16. `[B]` Yahoo Finance se data lene par data quality issues kaise handle kiye? Specifically SSL issues ka kya solution tha?
17. `[M]` Retry/backoff logic kyun zaroori hai data download me? Without it kya fail hota?
18. `[M]` Missing values, outliers, corporate actions, split/bonus adjustments ka kya treatment hai?
19. `[M]` `min_trading_days: 1000` ka logic kya hai — koi stock agar 999 days ka data ho to exclude ho jaata hai?
20. `[M]` `max_nan_percent: 5%` threshold kyun choose kiya?
21. `[B]` NIFTY 50 ki stocks ke liye data period kya use kiya — 2015 se 2025 kyun?
22. `[M]` CSV gap-fill mechanism kya karta hai exactly? Startup par background thread kaise kaam karta hai?
23. `[H]` Data survivorship bias kaise handle kiya? Jo stocks NIFTY 50 se bahar ho gayi 2015–2025 ke beech, unka kya?
24. `[M]` Live price fetching (`live.py`) aur historical data me consistency kaise maintain ki?
25. `[B]` `all_close_prices.csv` me exactly kya stored hai? Columns aur rows ka format kya hai?

### Feature Engineering (21 Technical Indicators)

26. `[B]` 21 technical indicators kyun choose kiye? Ye selection data-driven thi ya heuristic?
27. `[B]` RSI (Relative Strength Index) kya measure karta hai aur portfolio context me kaise useful hai?
28. `[B]` MACD (Moving Average Convergence Divergence) kya hai? Signal line aur histogram ka kya matlab?
29. `[B]` Bollinger Bands kya hain? Inka use squeeze detection me kaise hota hai?
30. `[M]` SMA aur EMA me difference kya hai? EMA kyun zyada responsive hoti hai?
31. `[M]` ATR (Average True Range) kya measure karta hai? Volatility position sizing me kaise use hota hai?
32. `[M]` Stochastic Oscillator kya hai — %K aur %D ka matlab kya hai?
33. `[M]` Volume ratio features kyun include kiye? Volume ka price prediction me kya role hai?
34. `[M]` Z-score normalization kyun choose ki (`normalize: zscore`)? Min-max scaling kyun nahi?
35. `[M]` `clip_range: 5.0` — z-score 5 se zyada hone par clip karna kya represent karta hai?
36. `[M]` `rolling_window: 252` (1 year) kyun choose kiya normalization ke liye?
37. `[H]` Kya feature leakage ka risk tha? Agar haan, usse kaise prevent kiya?
38. `[H]` Train-test split time-series style me kiya gaya ya random split? Random split kya problem create karta hai time-series data me?
39. `[H]` Different stocks ke price scales itne different hote hain (e.g., MRF ₹1.2L vs SUZLON ₹30) — unko comparable kaise banaya?
40. `[H]` Kya feature multicollinearity issue aaya? SMA-5 aur SMA-20 highly correlated hote hain — kya ye problem hai?
41. `[H]` Aapne stationarity check kiya ya directly indicators use kiye? Non-stationary features RL training ko kaise affect karte hain?
42. `[M]` Returns aur volatility features included hain — ye lagging indicators hain, leading indicators nahi. Is limitation ka kya impact hai?

---

## SECTION 2 — Sentiment Analysis / FinBERT (Phase 3)

> **Files:** `src/sentiment/finbert.py`, `news_fetcher.py`, `yfinance_news.py`, `indian_rss.py`

### FinBERT Model

43. `[B]` Financial sentiment ko portfolio me include karna zaroori kyun hai? Pure technical analysis se kya miss hota hai?
44. `[B]` FinBERT kya hai? General BERT se kyun better hai financial news ke liye?
45. `[B]` `ProsusAI/finbert` specifically kya hai — kaun sa data par fine-tune kiya gaya hai?
46. `[M]` FinBERT ka output kya hota hai — classes kya hain? (positive/negative/neutral)
47. `[M]` Sentiment score -1 to +1 ka calibration kaise hua — raw softmax output se [-1,+1] me mapping kaise ki?
48. `[M]` FP16 (half precision) use karne ka reason kya tha? Specifically 4GB VRAM constraint kyun mention ki?
49. `[H]` FP16 me numerical precision loss hota hai — sentiment scoring ke context me kya impact hai?
50. `[M]` FinBERT locally download kar ke `data/finbert_local/` me rakha — ye architectural decision kyun?

### News Sources & Aggregation

51. `[B]` News headlines ko stock-specific sentiment me convert kaise kiya? "Reliance" mention hone par kya Reliance ka score update hota hai?
52. `[M]` Multiple headlines same stock ke liye aaye to aggregation kaise hoti hai — average, max, ya weighted?
53. `[M]` `decay_factor: 0.95` ka matlab kya hai? Purani news ka sentiment kaise expire hota hai?
54. `[M]` `news_cache_ttl: 180` seconds kyun? 3 minute refresh logic ka kya rationale hai?
55. `[M]` Indian RSS feeds specifically kyun include kiye — Yahoo Finance international news kyun sufficient nahi?
56. `[H]` Source reliability ka kya treatment hai — CNBC headline aur random blog post ko same weight milti hai?
57. `[H]` Rumors aur duplicate headlines ka filter kaise kiya?
58. `[H]` News latency ka issue kaise handle kiya? Market already price karta hai news — kya lag analysis kiya?
59. `[H]` Agar sentiment signal aur price momentum conflict kare (positive news, falling price) — tab kya hota hai?
60. `[H]` Sentiment tabhi useful hota hai jab market react kare — aapne lag effect check kiya? t+1, t+2?

### Integration with Portfolio

61. `[M]` `sensitivity: 2.0` parameter kya control karta hai? Sentiment ka portfolio weights par exactly kitna impact hai?
62. `[M]` Sector mood kaise compute hoti hai — individual stock sentiments ka aggregate?
63. `[M]` Bullish/neutral/bearish thresholds ka basis kya hai — 0.1 se upar bullish, neeche bearish?
64. `[H]` Sentiment model news coverage bias se affected to nahi hota? Large-cap stocks ka news zyada hota hai.
65. `[H]` Sentiment score aur actual next-day returns ka correlation tune kiya? Out-of-sample validation hua?
66. `[M]` SQLite `sentiment.db` me kya store hota hai — schema kya hai?
67. `[B]` Dashboard ke Sentiment tab me "LIVE" badge kab bujhta hai aur kab jalata hai?

---

## SECTION 3 — Graph Construction (Phase 4)

> **Files:** `src/graph/builder.py`

68. `[B]` Stock graph banane ka purpose kya hai — prediction, clustering, ya representation learning?
69. `[B]` Graph me kitne types ke edges hain? Teeno ko explain karo.
70. `[M]` **Sector edges** — same-sector stocks ko connect karte ho. Ye static hain ya dynamic? Source kya hai?
71. `[M]` **Supply-chain edges** — business relationships ka source kya tha? Manually curate kiye ya automated?
72. `[M]` **Correlation edges** — rolling 60-day correlation `> 0.6` threshold kyun? 0.4 ya 0.8 kyun nahi?
73. `[H]` Correlation threshold 0.6 — Is ye aapke specific data par validated hai ya literature se liya?
74. `[M]` Graph me directed edges hain ya undirected? Supply chain ke liye directional sense banta hai?
75. `[M]` Node features kya hain — sirf stock identity ya technical features bhi?
76. `[M]` PyTorch Geometric `Data` object me kya store hota hai — `x`, `edge_index`, `edge_attr`?
77. `[H]` Stock relationships static nahi hote — correlation edges time ke saath change karte hain. Is temporal evolution ko kaise capture kiya?
78. `[H]` Sector clustering aur correlation clustering me conflict ho to — same sector me hone par bhi low correlation, to kaun sa signal dominate karta hai?
79. `[H]` Agar ek stock highly connected hai (high degree), to uska importance kaise interpret karte hain — influence ya noise?
80. `[H]` Graph me isolated nodes possible hain — koi stock sector me unique aur correlations bhi low? Inhe kaise handle kiya?

---

## SECTION 4 — T-GAT (Temporal Graph Attention Network) — Phase 5

> **Files:** `src/models/tgat.py`

81. `[B]` T-GAT ka full form kya hai aur intuition kya hai?
82. `[B]` Graph Attention Network (GAT) me "attention" kya karta hai — traditional GCN se kaise alag hai?
83. `[M]` Temporal component graph me kaise integrate hota hai T-GAT me?
84. `[M]` Default config: `hidden_dim: 64`, `output_dim: 64`, `2 layers`, `4 attention heads` — inme se koi ek change karo aur kya impact hoga?
85. `[M]` Multi-relational edges (3 types) T-GAT me kaise handle hoti hain — separate attention ya combined?
86. `[M]` Attention mechanism graph me kya learn kar raha hai exactly — kaunse neighbors important hain?
87. `[H]` Simple correlation matrix se T-GAT better kyun hai? Specifically quantify karo — ablation test tha?
88. `[H]` Graph embeddings RL agent ko kaise feed hote hain — concatenation, addition, ya gating?
89. `[H]` GNN me over-smoothing problem kya hai? 2 layers rakhne ka kya reason tha?
90. `[H]` Graph layer ka ablation test kiya — without graph vs with graph performance difference kya raha?
91. `[M]` DARTS (Phase 10) T-GAT architecture ko kaise optimize karta hai?
92. `[H]` T-GAT training me graph structure fixed thi ya each epoch me update hoti thi?

---

## SECTION 5 — Deep RL Agents (Phase 6–7)

> **Files:** `src/rl/agent.py`, `src/rl/environment.py`

### RL Formulation

93. `[B]` Portfolio optimization ko RL problem me formulate kaise kiya — State, Action, Reward define karo?
94. `[B]` State space kya hai exactly — kitne dimensions? (50 stocks × 21 features + graph embeddings + sentiment?)
95. `[B]` Action kya represent karta hai — weight vector [0,1]^50 ya buy/sell signals?
96. `[M]` Action space continuous hai ya discrete? Kyun?
97. `[B]` Reward function kya hai — exact formula batao.
98. `[M]` `reward = sharpe_weight(1.0) × Sharpe - drawdown_penalty(0.4) × drawdown - turnover_penalty(0.02) × turnover` — har term ka rationale kya hai?
99. `[M]` Sharpe ratio reward ke form me use kiya ya post-evaluation metric ke form me? Dono me difference kya hoga?
100. `[M]` Transaction cost `0.001` (0.1%) aur slippage `0.0005` reward me include hai? Kyun zaroori hai?
101. `[H]` Reward shaping ka risk kya hai? Suboptimal shaping kya problems create kar sakti hai?
102. `[M]` Episode length `252` (1 trading year) kyun? Shorter/longer episode ka kya impact hota?
103. `[H]` Exploitation vs exploration balance kaise achieve kiya — specifically har algorithm me?

### RL Environment Constraints

104. `[B]` `max_position: 0.12` (12%) — ek stock me maximum 12% allocation. Kyun ye constraint?
105. `[M]` `stop_loss: -0.03` (-3% per stock) — iska enforcement environment me kaise hota hai?
106. `[M]` `max_drawdown: -0.12` circuit breaker — episode reset hota hai ya penalty milti hai?
107. `[H]` Constraints ke saath RL environment non-trivial action masking require karta hai — kaise handle kiya?
108. `[M]` `trading_days_per_year: 248` India calendar ke liye — standard 252 se alag kyun?

### Individual RL Algorithms

109. `[B]` PPO (Proximal Policy Optimization) kya hai? "Proximal" ka matlab kya hai?
110. `[M]` PPO me clipping epsilon ka role kya hai? Ye kyu stability improve karta hai?
111. `[B]` SAC (Soft Actor-Critic) kya hai? "Soft" ka entropy se kya connection hai?
112. `[M]` SAC entropy regularization kya karta hai aur financial markets me ye useful kyun hai?
113. `[B]` TD3 (Twin Delayed Deep Deterministic Policy Gradient) kya hai?
114. `[M]` TD3 me "Twin" kya hai — twin Q-networks ka overestimation problem se kya connection?
115. `[M]` TD3 me "Delayed" kya hai — policy update delay kyun?
116. `[B]` A2C (Advantage Actor-Critic) kya hai? PPO se kaise alag hai?
117. `[M]` "Advantage" kya hota hai — baseline se subtract karna kyun important hai?
118. `[B]` DDPG (Deep Deterministic Policy Gradient) kya hai?
119. `[M]` DDPG aur TD3 me main difference kya hai?
120. `[H]` In 5 algorithms me se kaun sa market volatility me zyada stable tha aur kyun?
121. `[M]` 500 episodes training ka selection justification kya hai?
122. `[H]` RL me overfitting ka risk kaise control kiya — validation environment alag tha?

### Ensemble

123. `[M]` Ensemble banane ka weighting scheme kya hai — equal weight ya performance-based?
124. `[M]` "Top-3 models by recent Sharpe ratio" — recent matlab kitne episodes ka window?
125. `[H]` Ensemble ka improvement genuine hai ya simply overfitting ka artifact? Kaise distinguish karoge?
126. `[H]` RL stability aur interpretability me tradeoff kaise handle kiya?
127. `[H]` RL agent ka benchmark equal-weight portfolio se hi kyun compare kiya — Markowitz optimal frontier kyun nahi?
128. `[M]` RL model ka output directly portfolio weights deta hai — softmax apply hoti hai?
129. `[H]` Agar market regime suddenly change ho jaye (bull to bear) to RL agent ka behavior kya hoga?

---

## SECTION 6 — TimeGAN (Phase 8)

> **Files:** `src/gan/timegan.py`

130. `[B]` TimeGAN kya hai aur iska use is project me kyu kiya?
131. `[M]` GAN architecture me Generator aur Discriminator ka kya role hai?
132. `[M]` TimeGAN normal GAN se kaise different hai — temporal component kaise add kiya?
133. `[M]` TimeGAN config: `seq_len: 128`, `latent_dim: 64`, `hidden_dim: 128`, `3 layers`, `500 epochs`. Inme se koi parameter change karo aur effect batao.
134. `[M]` Gradient accumulation `× 4` (effective batch 128) kyun use kiya? Memory constraint?
135. `[H]` Synthetic time series data ka quality kaise validate kiya — FID score ya statistical tests?
136. `[H]` TimeGAN se generate kiya data stress testing me kaise use hota hai — actual historical data replace karta hai ya augment?
137. `[H]` TimeGAN overfitting ho sakta hai training data par — generated series training data ke patterns hi repeat kare. Kaise avoid kiya?
138. `[H]` Real financial returns fat-tailed distribution follow karte hain (non-Gaussian) — TimeGAN ye capture kar pata hai?

---

## SECTION 7 — Stress Testing & Risk Management (Phase 9)

> **Files:** `src/gan/stress.py`

139. `[B]` Stress testing ka need kyun hai, agar returns already high hain?
140. `[B]` VaR (Value at Risk) kya hai — exact definition do.
141. `[B]` CVaR (Conditional VaR / Expected Shortfall) kya hai aur VaR se kaise better hai?
142. `[M]` Historical VaR vs Parametric VaR — aapne kaun sa use kiya aur kyun?
143. `[M]` VaR 95% vs VaR 99% — dono me kya difference hai practical interpretation me?
144. `[B]` Survival rate ka exact definition kya hai — kab koi simulation "survive" maana jata hai?
145. `[M]` 4 scenarios — Normal, 2008 crash, COVID crash, Flash crash — ka parameterization kaise kiya? Numbers kahan se aaye?
146. `[M]` Monte Carlo simulation me return distribution kya assume kiya — Gaussian, t-distribution, or empirical?
147. `[M]` `monte_carlo_paths: 10000`, `stress_scenarios: 1000` — ye numbers sufficient kyun hain?
148. `[H]` Correlation spike stress me portfolio kyun fail ho sakta hai — diversification illusion kya hoti hai?
149. `[H]` Black swan events me model ka robustness kaise judge kiya?
150. `[H]` Max drawdown high ho to Sharpe achha hone ke baad bhi portfolio acceptable hoga?
151. `[M]` Stress testing ka output actual allocation me use hota hai ya sirf visualization hai?
152. `[H]` Risk-adjusted return aur raw return me kaun zyada important hai real portfolio management me?
153. `[H]` Monte Carlo me correlation structure preserve kiya ya independent returns assume kiye?

---

## SECTION 8 — DARTS Neural Architecture Search (Phase 10)

> **Files:** `src/nas/darts.py`, `src/nas/search_space.py`

154. `[B]` NAS (Neural Architecture Search) kya hai — kya problem solve karta hai?
155. `[B]` DARTS (Differentiable Architecture Search) kya hai? Random search se kaise better hai?
156. `[M]` DARTS search space kya hai — MixedOp operations kaunse hain? (linear, conv1d, attention, skip, none)
157. `[M]` "Bilevel optimization" kya hai DARTS me — alpha weights aur model weights alag kyun optimize hote hain?
158. `[M]` Alpha weights kya represent karte hain — operation importance ka measure?
159. `[M]` Search phase: `50 epochs` → top-3 architectures extract → retrain `100 epochs`. Ye 2-phase approach kyun?
160. `[H]` DARTS me discretization step — continuous alpha weights se discrete architecture kaise nikalte hain?
161. `[H]` DARTS is project me exactly kisko optimize karta hai — T-GAT architecture ko ya RL policy network ko?
162. `[H]` DARTS bilevel optimization convergence guarantee karta hai? Kya local optima me phase ho sakta hai?
163. `[H]` NAS computationally expensive hai — is project me kya tradeoffs the search time vs performance gain?
164. `[M]` Dashboard me NAS tab kya dikhata hai — alpha weights, convergence curves, discovered architectures?

---

## SECTION 9 — Federated Learning & Privacy (Phase 11)

> **Files:** `src/federated/server.py`, `client.py`, `privacy.py`

### FL Architecture

165. `[B]` Federated learning ki need kya thi? Centralized training se kya problem solve hoti hai?
166. `[B]` Is project me FL ka setup kya hai — kaun clients hain, server kaun hai?
167. `[M]` 4 clients sectors par kyun based hain — ek possible design sectors me nahi clients as stocks hote?
168. `[M]` `50 rounds`, `5 local epochs` — ye hyperparameters kaise tune kiye?
169. `[B]` FedAvg kya hai — exact aggregation formula batao.
170. `[M]` FedAvg me weighting scheme kya hai — dataset size proportional kyun?
171. `[B]` FedProx kya hai aur FedAvg se kaise alag hai?
172. `[M]` FedProx ka proximal term `μ=0.01` — exact formula aur intuition kya hai?
173. `[M]` Client drift kya hota hai aur FedProx usse kaise reduce karta hai?
174. `[H]` Har client ka data distribution alag ho (non-IID) — ye FL me kyun challenging hai?
175. `[H]` Non-IID data pe FedAvg kyun struggle karta hai — gradient direction conflict?
176. `[H]` Real fund houses is system me kaise integrate ho sakte hain — actual deployment scenario kya hoga?
177. `[M]` Gradient sharing se privacy kaise preserve hoti hai — raw data share nahi hota?
178. `[H]` Kya malicious client poison karke global model ko affect kar sakta hai? Byzantine fault tolerance hai?

### Differential Privacy

179. `[B]` Differential Privacy kya hai — intuitive explanation do.
180. `[B]` DP-SGD kya karta hai step by step?
181. `[M]` Epsilon `8.0` ka practical meaning kya hai — "privacy budget" kyun kehte hain?
182. `[M]` Delta `10^-5` ka interpretation kya hai?
183. `[M]` Gradient clipping (`max_grad_norm`) DP me kyun zaroori hai?
184. `[H]` Epsilon 8.0 — ye strong privacy hai ya weak? Literature me financial data ke liye kya recommended hai?
185. `[H]` Privacy aur utility ke beech tradeoff kaise balance kiya — epsilon kam karo to kya hota hai?
186. `[H]` Agar noise zyada ho jaye to model performance par kya impact padega specifically?
187. `[H]` Secure aggregation implement ki hai ya sirf DP-SGD hai? Dono me difference kya hai?
188. `[M]` Federated learning me communication overhead ka kya cost hai — kitna data per round transfer hota hai?
189. `[H]` Why should federated learning improve performance if data is already centralized in a research setup?

---

## SECTION 10 — Quantum QAOA (Phase 12)

> **Files:** `src/quantum/qaoa.py`, `src/quantum/portfolio.py`

190. `[B]` QAOA kya hai — Quantum Approximate Optimization Algorithm?
191. `[B]` Is project me quantum computing kyon use kiya — classical optimization se kya fayda?
192. `[M]` QUBO (Quadratic Unconstrained Binary Optimization) formulation kya hai — portfolio problem kaise QUBO me convert hota hai?
193. `[M]` QUBO formula: `maximize returns - λ × risk subject to cardinality` — cardinality constraint kya hai?
194. `[M]` Ising Hamiltonian me QUBO kaise convert hota hai?
195. `[M]` QAOA circuit me cost unitary aur mixer unitary kya karte hain?
196. `[M]` `qaoa_layers: 3` — deeper circuit better results deta hai kya?
197. `[M]` COBYLA classical optimizer kyun choose kiya — gradient-free kyun?
198. `[M]` `max_qubits: 12` — 12 qubits se maximum kitne stocks select ho sakte hain (8 stocks configured)?
199. `[H]` Qiskit simulator use kiya real quantum hardware nahi — simulation results real hardware par reproduce honge?
200. `[H]` QAOA approximate hai — approximation guarantee kya hai? p=1,2,3 layers me quality kaise improve hoti hai?
201. `[H]` Classical portfolio optimization (Markowitz) aur QAOA ka comparison — kab QAOA genuinely better hoga?
202. `[H]` Quantum advantage abhi NISQ era me theoretical hai — is project me quantum results actually better hai ya equivalent?
203. `[M]` Dashboard me QAOA output kaise display hota hai — selected stocks aur weights?

---

## SECTION 11 — FastAPI Backend (Phase 13)

> **Files:** `src/api/main.py` (2300+ lines), `schemas.py`

### Architecture & Design

204. `[B]` FastAPI kyun choose kiya Flask ke instead?
205. `[M]` 25+ endpoints me CSV caching with thread-safe locks kyun zaroori tha?
206. `[M]` News sentiment TTL cache (180s) — FinBERT har request par run karna kyun avoid kiya?
207. `[M]` CORS configuration me `cors_origins: ['http://localhost:3000']` — production me kya change hoga?
208. `[M]` Startup background thread (`gap-fill`) kya karta hai — blocking startup se better kyun?
209. `[H]` Thread-safe CSV caching me potential race conditions kya hain — lock granularity kya rakhi?
210. `[M]` Pydantic schemas (`schemas.py`) use karne ka benefit kya hai — validation aur serialization?
211. `[H]` 25 endpoints ek file (`main.py` 2300 lines) me — kya better architecture possible tha?

### Key Endpoints

212. `[M]` `/api/portfolio-summary` vs `/api/portfolio-smart` vs `/api/portfolio-optimized` — teeno me kya difference hai?
213. `[M]` `/api/rl-summary` endpoint 6 algorithms ka data return karta hai — ye data live compute hoti hai ya pre-computed?
214. `[M]` `/api/news-sentiment` me `sector mood` compute kaise hoti hai — individual sentiments aggregate?
215. `[M]` `/api/stress-test` POST request me kya body send hoti hai — portfolio weights ya config?
216. `[M]` `/api/qaoa` endpoint quantum circuit actually run karta hai per request — ya cached?
217. `[H]` `/api/future-prediction` me percentile bands kaise compute hoti hain — Monte Carlo ya parametric?
218. `[M]` `/api/portfolio-growth` GET vs POST — dono me kya difference hai?
219. `[B]` `/api/health` endpoint me kya return hota hai — sirf status ya detailed info?

### Performance & Reliability

220. `[M]` API latency critical tabs me kaise handle ki — async endpoints kahan use kiye?
221. `[H]` Agar Yahoo Finance ya RSS source down ho jaye to fallback kya hai?
222. `[M]` Swagger UI automatically `/docs` par milti hai — is feature ka benefit kya hai?
223. `[H]` 2300 lines main.py — startup time kitna hoga, aur RL models loading kab hoti hai?

---

## SECTION 12 — React Dashboard / Frontend (Phase 14)

> **Files:** `dashboard/src/pages/`, `components/`, `lib/`

### Architecture

224. `[B]` React + TypeScript choose karne ka reason kya tha?
225. `[M]` Vite bundler kyun choose kiya — Create React App ya Next.js kyun nahi?
226. `[M]` Tailwind CSS use karne ka advantage kya hai component-based styling ke liye?
227. `[M]` `App.tsx` me 8 routes defined hain — React Router v6 use kiya? NavLink vs Link difference?
228. `[M]` State management kaise handle kiya — local state, Context API, Redux, ya custom hooks?
229. `[M]` `lib/api.ts` — centralized API client kyun banaya? Fetch directly kyun nahi kiya?
230. `[H]` Auto-refresh Sentiment tab me (every 3 min) race condition ka risk hai — same request multiple times overlap ho?

### Individual Pages

231. `[B]` Portfolio page me exactly kaun se metrics dikhte hain — list karo.
232. `[M]` RL Agent page me 6 algorithm buttons hain — ek select karne par kya update hota hai?
233. `[M]` Stress Testing page me 4 scenarios me se ek select karne par chart kaise update hota hai?
234. `[M]` Federated page me FedAvg vs FedProx convergence chart kya dikhata hai?
235. `[B]` Sentiment page me "LIVE" badge kab show hoti hai aur kab nahi?
236. `[M]` Graph Visualization page me force-directed layout kyun choose kiya — alternative layouts kya the?
237. `[M]` `150-frame physics simulation` graph me — 150 frames ka settling logic kya hai?
238. `[H]` Node click on graph tab — kya information show hoti hai us stock ke liye?
239. `[M]` Pipeline/Workflow tab me ML pipeline visualization kya dikhata hai — actual live pipeline ya static diagram?
240. `[M]` Future Prediction tab me `bull/base/bear` scenarios ka parameterization kya hai?

### Components & UI

241. `[M]` `ErrorBoundary.tsx` — har async path ko cover kyun nahi karta? Kahan cover nahi hota?
242. `[M]` `MetricCard.tsx` me color-coded thresholds ka basis kya hai — Sharpe > 1 green, < 0.5 red?
243. `[M]` `MetricInfoPanel.tsx` aur `PageInfoPanel.tsx` — user education ke liye kyun important hain?
244. `[M]` `Skeleton.tsx` loading state — UX improvement ke alawa technical benefit kya hai?
245. `[M]` `Toast.tsx` notification system — kab trigger hota hai? API error, data refresh?
246. `[H]` Charts performance (`PerformanceChart.tsx` Recharts) — large dataset (10 years daily) render karne ka approach kya hai?
247. `[M]` `SectorDonut.tsx` — sectors ki allocation kaise compute hoti hai from weights?
248. `[H]` `formatters.ts` — Indian number system formatting (lakhs, crores) kaise handle kiya?

---

## SECTION 13 — Evaluation, Metrics & Benchmarking

249. `[B]` Sharpe Ratio 1.87 ka benchmark kya hai, aur ye exceptional kyun maana jata hai?
250. `[B]` Annual return 28.4% kaise compute hua — exact formula kya hai?
251. `[M]` Volatility 14.2% kis frequency par measured hai — daily volatility ko annualize kaise kiya?
252. `[M]` Max drawdown -11.3% ka context kya hai — kis period me tha?
253. `[M]` Sharpe Ratio formula: `(Annualized Return - Risk Free Rate) / Annualized Volatility`. Risk-free rate 5% kyun?
254. `[B]` Sortino Ratio aur Sharpe me difference kya hai?
255. `[M]` Calmar Ratio kya hai — `Annualized Return / Max Drawdown`? Kab useful hai?
256. `[M]` Portfolio Turnover kya measure karta hai aur high turnover kyun bad hai?
257. `[H]` Outperformance statistically significant hai ya just visual? t-test ya bootstrap confidence interval?
258. `[H]` Backtesting bias avoid kaise kiya — specifically walk-forward testing ya expanding window?
259. `[H]` Look-ahead bias kya hota hai aur aapne kaise prevent kiya?
260. `[H]` Survivorship bias kya hota hai is context me — handled hai ya nahi?
261. `[M]` Equal-weight baseline kyun choose kiya benchmark ke liye — Markowitz efficient frontier kyun nahi?
262. `[M]` Portfolio vs NIFTY 50 comparison me transaction cost include hai?
263. `[H]` Metrics per algorithm (PPO, SAC, etc.) — kaun sa most consistent tha across different market regimes?
264. `[H]` Different time windows (2015–18, 2019–22, 2023–25) pe results stable rahe ya regime-dependent?
265. `[H]` Ablation study kiya — RL only, RL+Graph, RL+Sentiment, RL+FL, full stack? Results kya the?

---

## SECTION 14 — Software Architecture & Engineering

266. `[B]` PyTorch kyun choose kiya TensorFlow ke instead?
267. `[M]` `configs/base.yaml` centralized config ka benefit kya hai — hardcoded values se better kyun?
268. `[M]` `src/utils/seed.py` — reproducibility ke liye exactly kya karta hai? NumPy, PyTorch, Python random sab seed karta hai?
269. `[M]` Logging framework (`logger.py`) me kya log hota hai — request logs, model metrics, errors?
270. `[M]` 246 tests (245 pass) — 1 `xfail` kya hai aur kyun acceptable maana?
271. `[M]` Docker + docker-compose — exactly kya containerize kiya hai?
272. `[H]` Frontend aur backend ke beech data contract kaise defined hai — TypeScript types aur Pydantic schemas me consistency?
273. `[H]` 20,000+ lines of code me biggest complexity kis module me thi?
274. `[M]` Module boundaries kaise maintain kiye — circular imports se kaise bacha?
275. `[H]` Caching strategy kya hai — CSV cache, sentiment cache, model cache teeno alag hain?
276. `[M]` `data/` directory me CSV stored hai aur `data/sentiment.db` SQLite — consistency ya inconsistency?

---

## SECTION 15 — Deployment & Production Readiness

277. `[M]` Is system ko production me deploy karne se pehle kaun se blockers hain?
278. `[M]` Real-time data updates me reliability kaise ensure karoge?
279. `[M]` Model versioning kaise handle hoga — RL agent retrain hone par purana model kab replace hoga?
280. `[M]` Reproducibility kaise ensure karoge — `seed: 42` sirf training time par kaam karta hai?
281. `[M]` Different users ke liye different portfolios store karne ka plan kya hai — multi-tenancy?
282. `[M]` Authentication layer kyun important hai? JWT use karne ka reason kya hai (planned feature)?
283. `[H]` Model drift detect kaise karoge — portfolio performance degrade ho to kaise pata chalega?
284. `[H]` Human override feature hoga ya system fully autonomous rahega? Regulatory perspective kya hai?
285. `[M]` Audit trail ka importance kya hai — financial systems me logging requirements?
286. `[H]` Monitoring aur alerting system ka kya plan hai — Prometheus, Grafana, ya custom?
287. `[M]` PostgreSQL persistence planned hai — CSV se migrate karne ka migration plan kya hai?

---

## SECTION 16 — Limitations & Critical Thinking

288. `[B]` Project ki sabse badi limitation kya hai — honestly batao.
289. `[B]` Kaunsi assumption sabse fragile hai?
290. `[M]` Aapke results real market conditions me kitne transferable hain?
291. `[M]` Agar market crash ke baad regime permanently change ho jaye (e.g., like Japan lost decade) to kya hoga?
292. `[H]` FL setup virtual clients hai — real-world deployment me (actual fund houses) kya gap hai?
293. `[H]` Graph relationships ka ground truth kaise verify karte hain — supply chain edges manually curate kiye, errors possible?
294. `[H]` RL ka biggest criticism — black box decision making. Financial regulators isko kaise accept karenge?
295. `[M]` Kya project live trading ke liye ready hai? Agar nahi, specifically kya missing hai?
296. `[H]` Backtesting always looks good — kaise prove karoge ki is system ka alpha real hai aur not data mining bias?
297. `[H]` NIFTY 50 universe me limited assets — market impact aur liquidity constraints real portfolio me matter karte hain?
298. `[H]` Quantum component (QAOA) is project me genuine value add karta hai ya marketing term hai?
299. `[H]` Is ensemble genuinely better ya just overfitting to the validation period?
300. `[M]` Sentiment signal noisy hai — specifically kitna noise, kitna signal?

---

## SECTION 17 — Very Tough Panel-Level Questions

301. `[H]` Agar aapko 1 month aur mile to sabse pehle kya improve karoge — specifically?
302. `[H]` Kaunsa metric aapke system ka false sense of success de sakta hai — Sharpe ratio kab mislead karta hai?
303. `[H]` Is project me kaun sa part truly "AI" hai aur kaun sa engineering/data processing?
304. `[H]` Kaise prove karoge ki each layer adds value independently — joint vs individual contribution?
305. `[H]` If market becomes highly efficient (EMH holds), will your system still have edge?
306. `[H]` What would fail first in live deployment: data pipeline, model, or infrastructure?
307. `[H]` How would you explain the system to a non-technical investor in 2 minutes?
308. `[H]` If output is unstable in production, how would you diagnose the root cause — systematic debugging approach?
309. `[H]` Agar aapko is project pe paper likhna ho, kaunsa novel contribution highlight karoge?
310. `[H]` Professor bole "yeh sirf libraries ka combination hai, research kahan hai?" — kya reply karoge?

---

## SECTION 18 — Demo-Specific Questions

> *Ye questions live demo ke dauran pooche jaa sakte hain.*

311. `[B]` Dashboard kaise start karte hain — exactly kaun se commands?
312. `[B]` Agar live demo me sentiment API slow ho jaye to aap kya bolenge?
313. `[M]` Agar graph tab me node positions change ho rahe hon to stability ka logic kya hai — force simulation?
314. `[B]` RL tab me kaun sa chart most important hai aur kyun?
315. `[M]` Portfolio tab me metrics ko color-coded thresholds me map karne ka basis kya hai?
316. `[B]` Stress Testing tab me 2008 crisis vs COVID crash results me kya difference dikhta hai?
317. `[M]` Federated tab me FedProx consistently better dikhta hai ya kabhi FedAvg better?
318. `[B]` Sentiment tab me auto-refresh button manually bhi trigger kar sakte hain?
319. `[M]` Graph tab me edge type filter (sector/supply-chain/correlation) toggle karne par kya change hota hai?
320. `[B]` Pipeline tab me visualization static hai ya animated?
321. `[M]` Future Prediction tab me `bull/base/bear` scenario select karne par exactly kya change hota hai?
322. `[B]` Settings page ka kya hai — config change karne ka UI hai ya sirf display?

---

## SECTION 19 — Rapid Fire / One-Liners

> *30 second me jawab dene wale questions*

323. Sharpe Ratio kya measure karta hai?
324. Sortino aur Sharpe me ek line me difference?
325. VaR (Value at Risk) kya hai?
326. CVaR kya hai — Expected Shortfall kyun bhi kehte hain?
327. FedAvg kya hai — ek sentence me?
328. FedProx FedAvg se kaise alag hai?
329. DP-SGD kya karta hai?
330. Epsilon (ε) privacy budget me kya represent karta hai?
331. FinBERT kya hai?
332. T-GAT full form aur use?
333. DARTS kya hai?
334. QAOA full form aur purpose?
335. TimeGAN kya generate karta hai?
336. Non-IID data kya hota hai?
337. Client drift kya hota hai?
338. Look-ahead bias kya hota hai?
339. Survivorship bias kya hota hai?
340. Max drawdown kya hai?
341. Calmar ratio kya hai?
342. DDPG continuous ya discrete action space?
343. PPO me clipping kyun?
344. SAC me entropy term kyun?
345. QUBO full form?
346. Ising Hamiltonian kya hai?
347. COBYLA kya hai?
348. MixedOp DARTS me kya hai?
349. Bilevel optimization kya hai?
350. GAT aur GCN me difference?
351. Correlation threshold 0.6 — above ya below edge banata hai?
352. `seed: 42` — kyun 42? (reproducibility ke liye koi bhi fixed seed chalta hai)

---

## SECTION 20 — Mathematical / Formulae Questions

> *Professor formula pooch sakta hai — ye ready rakho*

353. Sharpe Ratio formula likhiye — return, volatility, risk-free rate ke terms me?
354. Sortino Ratio formula — downside deviation kaise compute hoti hai?
355. VaR formula (historical method) — percentile kaise extract karte hain?
356. CVaR formula — conditional expectation kaise compute hota hai?
357. FedAvg aggregation formula — client weights kaise combine hote hain?
358. FedProx objective function — proximal term kya add hota hai?
359. Differential Privacy (ε, δ)-DP definition — formal definition kya hai?
360. Gaussian Mechanism — kitna noise add karna padta hai for (ε, δ)-DP?
361. QAOA cost Hamiltonian formula — portfolio context me kya hogi?
362. MACD formula — EMA(12) - EMA(26) aur signal line?
363. RSI formula — relative strength se RSI kaise compute hota hai?
364. Bollinger Bands formula — SMA ± 2σ?
365. PPO clipped objective — exact formula kya hai?
366. SAC objective — entropy term kaise add hota hai to actor loss?
367. Advantage function A(s,a) — Q(s,a) - V(s) kya hota hai?
368. Annualized return formula — daily returns se kaise compute karte hain?
369. Annualized volatility — daily std se kaise annualize karte hain (×√252)?
370. ATR formula — True Range ka average?

---

## SECTION 21 — Cross-Cutting & Integration Questions

> *Ye tab pooche jaate hain jab panel sirf ek component nahi poora system samajhna chahta hai*

371. `[H]` Data se RL action tak — ek complete data flow step-by-step batao.
372. `[H]` Sentiment signal RL environment me kab aur kaise integrate hota hai?
373. `[H]` T-GAT embeddings RL state me kaise jaate hain — concatenation ka dimension kya hota hai?
374. `[H]` FL training me locally trained RL models globally aggregate kaise hote hain?
375. `[H]` DARTS T-GAT architecture search karta hai — us optimized architecture ko RL phir use karta hai?
376. `[H]` TimeGAN data stress testing me kaise use hota hai — RL agent bhi TimeGAN data par test hota hai?
377. `[H]` QAOA ka output (selected stocks) RL agent ke universe ko restrict karta hai ya independent hai?
378. `[H]` Agar ek component fail ho jaye (e.g., FinBERT model load na ho) to system gracefully degrade karta hai?
379. `[H]` End-to-end training kiya gaya ya har phase independently trained?
380. `[H]` Sabse slow bottleneck component kaun sa hai — data download, FinBERT inference, RL training, ya graph construction?

---

## SECTION 22 — Future Scope Questions

381. `[B]` Future scope me kaunsa feature most impactful hoga?
382. `[M]` PostgreSQL replace karega CSV — migration strategy kya hogi?
383. `[M]` Options aur derivatives support add karna ho to RL action space kaise change hoga?
384. `[M]` Multi-asset classes (bonds, gold, crypto) add karne ka plan kya hai?
385. `[H]` Real-time streaming data (WebSocket) integrate karna ho to architecture kaise change karoge?
386. `[H]` Larger quantum hardware available hone par — QAOA ka benefit genuinely increase hoga?
387. `[M]` Mobile app/PWA banane ka plan hai?
388. `[H]` Agar international markets (NYSE, NASDAQ) include karo to — cross-market correlation kaise handle karoge?

---

## QUICK REFERENCE — Key Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| Sharpe Ratio | 1.87 | Ensemble portfolio |
| Annual Return | 28.4% | Backtest 2023–2025 |
| Volatility | 14.2% | Annualized |
| Max Drawdown | -11.3% | Backtest period |
| Total Stocks | 50 | NIFTY 50 |
| Technical Indicators | 21 | Feature engineering |
| RL Algorithms | 6 | PPO/SAC/TD3/A2C/DDPG/Ensemble |
| FL Clients | 4 | Sector-based |
| FL Rounds | 50 | 5 local epochs each |
| DP Epsilon | 8.0 | Privacy budget |
| DP Delta | 10⁻⁵ | Failure probability |
| GNN Hidden Dim | 64 | T-GAT |
| GNN Attention Heads | 4 | T-GAT |
| Correlation Threshold | 0.6 | Graph edges |
| Correlation Window | 60 days | Rolling |
| Monte Carlo Paths | 10,000 | Stress testing |
| Stress Scenarios | 1,000 | Per scenario |
| RL Max Position | 12% | Per stock constraint |
| RL Stop Loss | -3% | Per stock |
| RL Circuit Breaker | -12% | Portfolio drawdown |
| Sharpe Reward Weight | 1.0 | RL reward |
| Drawdown Penalty | 0.4 | RL reward |
| Turnover Penalty | 0.02 | RL reward |
| Transaction Cost | 0.1% (0.001) | Per trade |
| Slippage | 0.05% (0.0005) | Market impact |
| Data Start | 2015-01-01 | Yahoo Finance |
| Train End | 2021-12-31 | 70% split |
| Val End | 2023-12-31 | 15% split |
| Eval End | 2025-12-31 | 15% split |
| RL Episodes | 500 | Training |
| FinBERT Cache TTL | 180 seconds | News sentiment |
| Sentiment Decay | 0.95 | Per day without news |
| DARTS Search | 50 epochs | Architecture search |
| DARTS Retrain | 100 epochs | Final architecture |
| TimeGAN Epochs | 500 | GAN training |
| QAOA Layers | 3 | Circuit depth |
| Max Qubits | 12 | Simulator constraint |
| QAOA Assets | 8 | Selected per run |
| Risk Free Rate | 5% | Sharpe calculation |
| Starting Capital | ₹1 Crore | Portfolio simulation |
| Tests | 246 (245 pass) | pytest suite |
| API Endpoints | 25+ | FastAPI |
| Frontend Pages | 8 | React dashboard |
| Lines of Code | 20,000+ | Total project |

---

## PREPARATION TIPS

### Kya zaroor samjhna chahiye:
1. **RL reward function** — formula yaad karo aur har term ka rationale samjho
2. **Ensemble weighting** — Sharpe-based top-3 selection logic
3. **FedProx proximal term** — formula aur intuition
4. **T-GAT vs simple correlation** — ablation argument ready rakho
5. **VaR vs CVaR** — dono ki definitions aur limitations
6. **DP epsilon interpretation** — nahi toh FL section weak lagegi
7. **Data split** — 2015-2021 train, 2021-2023 val, 2023-2025 eval — kyun ye dates

### Common professor traps:
- "Sharpe 1.87 — transaction cost included?" → Yes, 0.1% per trade
- "FL me real privacy hai?" → Gradient sharing + DP-SGD, lekin secure aggregation nahi
- "QAOA genuinely better hai?" → Honest answer: simulator par, small scale, not proven advantage
- "Backtest me future data use nahi kiya?" → Time-based split, no lookahead
- "Sentiment actually helps?" → Need correlation study — honest gap

---

*Total Questions: 388 | Last Updated: 2026-05-01*
