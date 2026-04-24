# FINQUANT-NEXUS v4 — System Working (Q&A Style)

---

## Q: Mera project kya hai?

Tum ek **AI-based portfolio management system** bana rahe ho NIFTY 50 stocks ke liye.

Simple shabdon mein:

> "₹1 crore hai — inhe NIFTY 50 ke 45 stocks mein kaise invest karo ki maximum profit ho aur risk kam ho?"

---

## Q: Data kahan se aata hai?

```
Yahoo Finance → 2015 se 2025 tak ka price data download kiya → CSV files mein save hai
```

Yahi foundation hai. Sab kuch inhi CSV files se chalta hai. Live/real-time download sirf ek extra utility hai — core system ke liye zaroorat nahi.

---

## Q: System end-to-end kaise kaam karta hai?

```
CSV Data (raw prices)
      ↓
Feature Engineering  ← RSI, MACD, Bollinger Bands, volatility etc. calculate karo
      ↓
      ├── GNN          ← Stocks ke beech relationships samjho (correlation graph)
      ├── RL Agents    ← Portfolio weights seekho (kisme kitna invest karo)
      ├── Sentiment    ← News padho, market mood samjho
      └── Stress Test  ← Crash scenarios mein portfolio kaisa behave karta hai
      ↓
FastAPI Backend  ← Sab results ek jagah compile karo
      ↓
React Dashboard  ← Visually dikhao
```

---

## Q: Har component kya karta hai?

### RL Agent (6 algorithms) — Sabse important part

- Agent ek "trader" ki tarah seekhta hai
- Environment: stock market simulation
- Action: har stock mein kitna % invest karo
- Reward: Sharpe ratio (return/risk balance)

| Algorithm | Strategy |
|-----------|----------|
| PPO | Moderate momentum + equal-weight blend |
| SAC | Soft momentum, diversified |
| TD3 | Short-term reversal (contrarian) |
| A2C | Inverse-volatility weighting |
| DDPG | Concentrated top-K momentum |
| Ensemble | Sab 5 ko performance-weighted combine karo |

---

### GNN (Graph Neural Network)

- Stocks ek dusre se connected hain (HDFC Bank → HDFC Life — related)
- Graph banao: nodes = stocks, edges = correlation strength
- Isse portfolio better diversify hota hai (correlated stocks pe over-weight nahi karte)

---

### Sentiment Analysis

- News RSS feeds se headlines laao (Business Standard, LiveMint, etc.)
- FinBERT (finance-specific AI) se analyze karo — positive / negative / neutral
- Portfolio weights sentiment ke hisaab se adjust karo

---

### Stress Testing

- 2008 crash, COVID crash, flash crash — in scenarios mein portfolio simulate karo
- VaR (Value at Risk), CVaR calculate karo
- 1000 Monte Carlo paths run karo — survival rate dekho

---

### Federated Learning

- 4 "clients" (alag alag sectors) independently train karte hain
- Privacy-preserving — data share nahi hota, sirf model weights share hote hain
- Real-world distributed training simulate karta hai

---

## Q: Federated Learning mein weights/biases kaise share hote hain?

### 4 Clients (Sector-wise split)

| Client | Sectors | Stocks |
|--------|---------|--------|
| Client 1 | Banking + Finance | ~15 stocks |
| Client 2 | IT + Telecom | ~6 stocks |
| Client 3 | Pharma + FMCG | ~8 stocks |
| Client 4 | Energy + Auto + Others | ~15 stocks |

### Ek FL Round ka lifecycle (50 baar repeat hota hai):

```
Server → Global model weights broadcast karo → sabhi 4 clients ko

  Client 1 (Banking/Finance):
    1. Apne 15 banking stocks ke returns pe 5 epochs train karo
    2. Gradient compute karo (weight update)
    3. DP-SGD step:
         a. Har gradient ko max-norm 1.0 pe clip karo
         b. Gaussian noise add karo: noise ~ N(0, σ²·I)
         c. σ = √(2·ln(1.25/δ)) / ε  ← privacy formula
    4. Sirf YE NOISY weight delta bhejo server ko
    5. Raw price data kabhi nahi bheja — kabhi bhi

  Client 2, 3, 4 — same process parallel mein

Server receives 4 noisy weight deltas:
  FedProx aggregate:
    w_global = Σ(client_i_weight × n_i/N) + μ||w - w_global||²
    ↑ proximal term = clients ko global se zyada door nahi jaane deta

Server → Updated global model weights broadcast → next round
```

---

## Q: Privacy kaise maintain hoti hai? DP-SGD kya hai?

### Intuition (simple)

> "Agar tum kisi se kuch batana chahte ho lekin poora nahi batana, toh thoda jhooth mila do. Itna jhooth ki asli information recover na ho sake — lekin model phir bhi kuch seekh sake."

Yahi hai Differential Privacy.

### Technical

**DP-SGD = Differentially Private Stochastic Gradient Descent**

```
Normal SGD:   gradient = ∂Loss/∂weights  ← exact, data leak possible
DP-SGD:       gradient = clip(∂Loss/∂weights, C) + N(0, σ²C²I)
                                                    ↑ controlled noise
```

**Parameters:**
- `ε (epsilon) = 8.0` — privacy budget. Lower = more private. ε=8.0 is "moderate" for finance
- `δ = 10⁻⁵` — probability that guarantee fails (very small)
- `C = 1.0` — gradient clipping norm (max gradient size)
- `σ` — noise scale, calibrated so total privacy loss after 50 rounds = (ε=8.0, δ=10⁻⁵)

**Mathematical guarantee:**
> Agar koi adversary saare 50 rounds ke saare weight updates dekhe bhi, toh kisi bhi individual stock ke price returns ko reverse-engineer karne ki probability (ε, δ) se bound hai — matlab mathematically impossible hai poori information recover karna.

---

## Q: FL ka Portfolio se kya connection hai?

```
Smart Optimize (Portfolio tab) =
  RL momentum weights    × 40%   ← RL Agent tab se
+ News Sentiment weights × 40%   ← Sentiment tab se
+ FL sector weights      × 20%   ← Federated tab se (YE YAHAN SE AATA HAI)
  ↓
SLSQP Max Sharpe optimization
  ↓
Final "Smart Portfolio" weights
```

FL ka Global Sharpe jitna better hoga, utna better sector allocation signal Smart Portfolio mein jaayega.

**Practically:** FL tab pe agar Banking sector ka model achha converge hua hai, toh Smart Portfolio automatically Banking stocks ko thoda zyada weight dega — without any manual input.

---

## Q: FedProx vs FedAvg — kya fark hai?

```
FedAvg:   w_global = (1/4) × (w1 + w2 + w3 + w4)
           Simple average — Banking client (high vol) dominate kar sakta hai

FedProx:  w_global = Σ(w_i × n_i/N) + μ||w_i - w_global||²
           Proximal term = clients ko global se zyada door nahi jaane deta
           Better for Non-IID data (Banking ≠ IT — alag volatility profiles)
```

NIFTY 50 mein sectors bahut alag hain (Banking = high correlation, IT = high growth), isliye FedProx zyada suitable hai — convergence curve pe clearly dikhta hai FedProx faster converge karta hai.

---

## Q: Final output kya hai?

```
₹1 crore lao → Smart Portfolio batao → kitna profit, kitna risk
```

Dashboard dikhata hai:
- Portfolio performance vs NIFTY index vs Fixed Deposit
- RL agent ne kaunsa allocation recommend kiya
- News sentiment kaisi hai
- Stock correlation graph
- Crash scenarios mein survival rate

---

## Q: Yfinance se 45 stocks rate limit kyun hota hai?

Yahoo Finance ke saath 45 tickers ek saath request karo → wo "Too Many Requests" deta hai.

**Causes:**
1. Batch mein 45 parallel requests — Yahoo ka bot detection trigger hota hai
2. `gap_days == 1` — aaj ka data maang rahe ho jo market close ke pehle exist hi nahi karta

**Fix (dissertation ke liye):**
- CSV data 2015–2025 already hai — gap-fill ki zaroorat nahi demo mein
- `gap_days <= 1` → skip, CSV use karo
- Agar genuine gap (2+ din) → per-ticker download, 5s delay, 30s wait after batch fail

---

## Q: Ye "live" system hai ya simulation?

**Simulation hai** — research/academic project.

- Historical data (2015–2025) pe trained aur tested
- RL agents simulate karte hain trading decisions
- Real money involve nahi, real orders place nahi hote
- Demo ke liye CSV data kaafi hai — real-time feed optional hai

---

## Q: Metrics ka matlab kya hai?

| Metric | Matlab |
|--------|--------|
| Sharpe Ratio | Return / Risk — higher is better (>1 = good) |
| Sortino Ratio | Sharpe jaisa but sirf downside risk count karta hai |
| Annualized Return | Yearly average return % |
| Max Drawdown | Peak se maximum girna — e.g., -12% = 12% neeche gaya |
| Annualized Volatility | Return kitna fluctuate karta hai |

**Risk-free rate = 5%** (India historical average for 2015–2021 backtest period) — metrics calculate karne mein use hota hai.

---

## Q: Dashboard ke 6 tabs kya karte hain?

| Tab | Kya dikhata hai |
|-----|----------------|
| Portfolio | Core output — holdings, returns, sector allocation |
| RL Agent | 6 algorithms comparison, weights, reward curves |
| Stress Testing | Crash scenarios, VaR, Monte Carlo |
| Federated | FL training convergence, client fairness |
| Graph Viz | Stock correlation graph (GNN) |
| Sentiment | Live news sentiment, market mood |

---

## Q: Sentiment tab mein kya hota hai? FinBERT kaise kaam karta hai?

### Pipeline

```
Indian RSS Feeds (ET, BusinessStandard, LiveMint) ← PRIMARY
      ↓ (agar <10 articles mila)
yFinance news ← SECONDARY
      ↓ (agar abhi bhi <10)
Google News RSS ← FALLBACK
      ↓
Deduplication (same headline filter karo)
      ↓
FinBERT batch inference
      ↓
score = P(positive) − P(negative)   ← range: -1.0 to +1.0
      ↓
Sector aggregation + Portfolio weight adjustment
      ↓
Frontend: 3 tabs (News / Portfolio Impact / Sectors)
```

### FinBERT kya hai?

- BERT (Google ka NLP model) ko financial text pe fine-tune kiya gaya
- Input: ek headline string
- Output: 3 probabilities = [P(positive), P(negative), P(neutral)] — sum = 100%
- Score = P(positive) − P(negative)

**Kyun FinBERT aur General BERT nahi?**

| Headline | General BERT | FinBERT |
|----------|-------------|---------|
| "Markets turn bearish" | Neutral (word not understood) | Negative ✓ |
| "Profit booking seen" | Positive (profit = good) | Neutral ✓ |
| "HDFC beats estimates" | Neutral | Positive ✓ |

### Sentiment se Portfolio weights kaise bante hain?

```
For har stock t in NIFTY 50:
  agar t ke liye specific news hai:
    sent_t = average FinBERT score of t's headlines
  else:
    sent_t = t's sector ka average score

  adj_weight_t = base_weight × (1 + sent_t × sensitivity)
                                          ↑ sensitivity = 2.0

Sabhi adj_weights normalize karo → sum = 100%
```

**Example:**
- HDFC Bank base weight = 2.13% (equal weight, 47 stocks)
- FinBERT score = +0.30 (positive news)
- adj_weight = 2.13 × (1 + 0.30 × 2.0) = 2.13 × 1.60 = **3.41%** (+1.28% overweight)

**Example (negative):**
- WIPRO base weight = 2.13%
- FinBERT score = -0.25 (negative news)
- adj_weight = 2.13 × (1 − 0.25 × 2.0) = 2.13 × 0.50 = **1.07%** (−1.06% underweight)

### Smart Portfolio mein Sentiment ka role

```
Smart Portfolio =
  RL Agent weights     × 40%
+ Sentiment weights    × 40%   ← YE YAHAN SE AATA HAI
+ FL sector weights    × 20%
  ↓
SLSQP Max Sharpe optimization
  ↓
Final "Smart Portfolio" weights
```

**Practically:** Agar market mood "Bullish" hai (avg score > 0.08), toh positive-news stocks automatically overweight ho jaate hain. "Bearish" market mein negative-news stocks trim ho jaate hain — news-driven risk management.

### Score Distribution buckets

| Range | Bucket |
|-------|--------|
| score < -0.30 | Very Negative |
| -0.30 ≤ score < -0.10 | Negative |
| -0.10 ≤ score ≤ +0.10 | Neutral |
| +0.10 < score ≤ +0.30 | Positive |
| score > +0.30 | Very Positive |

Ek healthy market mein mostly Neutral headlines hote hain. Extreme values (> ±0.3) = "High Impact Alerts" section mein dikhate hain.

### Caching

- TTL = 15 minutes (default) — same request mein fresh data maangne se cache return hota hai
- "Force Refresh" button TTL bypass karta hai aur fresh data fetch karta hai
- Auto-refresh: har 3 minute mein (sirf tab visible ho tab)
- Session trend history: localStorage mein 48 data points tak save hota hai

---

## Q: Graph Visualization tab mein kya hota hai? GNN kaise kaam karta hai?

### Intuition

> "Stock market mein har stock sirf akela nahi hota — HDFC Bank aur HDFC Life correlated hain, TCS aur INFY compete karte hain, ONGC → RELIANCE supply chain hai. Agar agent yeh relationships samjhe toh better diversification kar sakta hai."

Yahi hai GNN ka kaam.

### 3 Edge Types (3 tarah ke relationships)

| Edge Type | Color | Matlab |
|-----------|-------|--------|
| Sector | Orange | Same industry ke stocks — Banking stocks sab ek dusre se connected |
| Supply Chain | Purple | Business relationships — ONGC crude supply karta hai RELIANCE ko |
| Correlation | Teal | 60-day rolling price co-movement > threshold (0.6) |

### Graph Structure

```
Nodes = NIFTY 50 stocks
  Size = Degree (kitne connections hain — bigger = more connected hub)
  Color = Sector (Banking=orange, IT=purple, etc.)

Edges = Relationships
  Weight = Correlation strength (0 to 1)

Force Simulation = Physics engine
  - Repulsion: nodes dusron se door bhaagte hain
  - Attraction: connected nodes ek dusre ke paas aate hain
  - Result: Banking cluster apne aap banta hai, IT cluster banta hai
```

### GNN (Graph Neural Network) Pipeline

```
Input: Stock price features (returns, RSI, MACD, etc.)
          ↓
T-GAT (Temporal Graph Attention Network):
  1. Har node apne neighbors ki features collect karta hai
  2. Attention mechanism: "HDFC Bank ki features mein HDFC Life
     ka contribution zyada important hai, CIPLA ka kam"
  3. Aggregation: weighted average → 32-dim embedding per stock
          ↓
Output: Node embeddings [n_stocks × 32]
  = "Har stock ka fingerprint jo uske neighbors ko encode karta hai"
```

### RL Agent ko Kya Milta Hai?

```
RL Agent ka input =
  Price features (returns, volatility)    ← CSV se
+ Sentiment scores                          ← FinBERT se
+ GNN node embeddings [n_stocks × 32]      ← YE YAHAN SE AATA HAI
  ↓
Agent action: allocation weights for each stock
```

**Practically:** HDFC Bank ka embedding encode karta hai ki wo Banking sector mein hai, HDFC Life se connected hai, ICICI Bank se correlated hai. Jab RL agent HDFC Bank mein invest karna chahta hai, embedding se samajhta hai ki HDFC Life mein bhi zyada invest karna = correlated bet = Sharpe ratio kamm = reward kam.

### GNN Diversification Mechanism

```
Agar 2 stocks highly connected hain (high edge weight):
  GNN embedding → dono ka vector similar hoga
  RL agent reward (Sharpe) = return / risk
  Correlated bets = same-direction moves = low diversification = high variance
  High variance = Sharpe down = reward down
  Agent learns: "inhe dono overweight mat karo"
```

**Example:**
- HDFC Bank + HDFC Life: correlation = 0.85 (strong sector + business edge)
- Agent: HDFC Bank ko 5% diya → HDFC Life ko 5% dena = 10% concentrated bet in correlated pair
- Better: 5% HDFC Bank + 2% HDFC Life + 3% diversify elsewhere → Sharpe 12% better

### Dashboard Tab Features

| Feature | Description |
|---------|-------------|
| Force Graph | Physics simulation — 150 frames stabilize karta hai |
| Node Click | Stock ka detail dekho: degree, weight, daily return, neighbors |
| Edge Toggle | Sector/Supply/Correlation edges on/off karo |
| Top Correlations | Strongest 8 pairs ranked by correlation weight |
| Sector Legend | Kitne stocks per sector hain |
| Graph Stats | Total nodes, edges, density, avg degree |

### Degree Distribution (not shown in UI but computed)

- High degree stock = hub = zyada connections
- Low degree = peripheral = good diversifier
- Banking stocks typically have highest degree (most correlated with each other)
- Pharma stocks typically low degree (less correlated with other sectors)
