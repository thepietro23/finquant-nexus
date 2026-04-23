# Federated Learning Tab
### Route: `/fl`

---

## 1. Purpose

Federated Learning tab **privacy-preserving collaborative model training** demonstrate karta hai. Yeh batata hai ki **multiple fund houses (clients) apne private portfolio data ko share kiye bina ek shared model train kar sakte hain** — aur phir bhi individually better performance milti hai.

**Project mein role:**
- Yeh tab FINQUANT-NEXUS ka **privacy aur fairness layer** hai
- Real-world problem solve karta hai: SEBI regulations ke karan fund houses raw trade data share nahi kar sakte
- Differential Privacy (DP-SGD) ek **mathematical privacy guarantee** deta hai
- Federated approach se **har client ko individually better Sharpe ratio milta hai** compared to training alone

---

## 2. Target Users & Usage

**Target Users:**
- Financial regulators / compliance officers
- ML researchers studying privacy-preserving learning
- Fund managers wanting collaborative model improvement without data leakage

**Real Usage Flow:**

```
User opens Federated Learning tab
        ↓
4 Metric Cards load: FL Rounds, Privacy ε, Global Sharpe, Clients
        ↓
User reads 4 Client Info Cards (Banking, Finance, IT, Others)
        ↓
Looks at Convergence Chart: FedProx vs FedAvg over 50 rounds
        ↓
Observes: FedProx converges faster and smoother than FedAvg
        ↓
Scrolls to Fairness Comparison BarChart
        ↓
Sees: "With FL" Sharpe > "Without FL" Sharpe for all 4 clients
        ↓
Clicks Privacy ε card → Explanation of DP-SGD opens
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component UI |
| Recharts | LineChart (convergence — 6 lines), BarChart (fairness comparison) |
| Framer Motion | Card hover animations, metric card expansion |
| Tailwind CSS v4 | Per-client color coding (4 distinct colors) |

### 3.2 Backend API Called

| Endpoint | Method | Returns |
|---|---|---|
| `/api/fl-summary` | GET | n_rounds, n_clients, epsilon, clients[], convergence[], fairness[] |

**Backend files:** `src/federated/server.py`, `src/federated/client.py`, `src/federated/privacy.py`

### 3.3 Federated Learning Architecture

**Setup:**
```
1 Global Server (FLServer)
        ↑↓ gradients only (never raw data)
4 Sector Clients (FLClient)
├── Client 0: Banking (15 stocks)
├── Client 1: Finance (10 stocks)
├── Client 2: IT (6 stocks)
└── Client 3: Others (19 stocks)
```

**Training Round (1 round of 50):**
```
Server broadcasts global model weights
        ↓
Each client trains locally on its own sector's data
        ↓
Each client computes gradient update
        ↓ DP-SGD applied here (clip + noise)
Clients send gradients to server (NOT raw data)
        ↓
Server aggregates using FedProx / FedAvg
        ↓
Global model updated
        ↓ repeat for 50 rounds
```

### 3.4 Federated Strategies Compared

**FedAvg (McMahan et al., 2017):**
```
global_weights = Σ (n_client_i / n_total) × local_weights_i
```
- Simple weighted average by client dataset size
- Prone to "client drift" when data distributions differ across clients (non-IID)

**FedProx (Li et al., 2020):**
```
Local objective = loss + (μ/2) × ||weights - global_weights||²
```
- Adds proximal term (μ) to prevent local model from drifting too far from global
- More stable on non-IID data (different sectors = non-IID by design)
- **FedProx wins** in this setting — banking data ≠ IT data

### 3.5 Differential Privacy — DP-SGD

**Why it's needed:**
- Even sharing gradients can leak information about individual training samples
- DP-SGD adds **calibrated noise** to guarantee that no adversary can infer any single data point

**How DP-SGD works:**
```
Step 1: Compute per-sample gradients
Step 2: Clip gradients: g̃ = g / max(1, ||g||/C)    (C = clip norm)
Step 3: Add Gaussian noise: g_noisy = g̃ + N(0, σ²C²I)
Step 4: Update model with noisy gradients
```

**Privacy Parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| **ε (epsilon)** | 8.0 | Privacy budget — lower is more private |
| **δ (delta)** | 10⁻⁵ | Probability of privacy failure |
| **Clip Norm (C)** | 1.0 | Max gradient norm before clipping |
| **Noise Multiplier (σ)** | Calibrated per (ε, δ) | Added noise scale |

**Interpretation:**
- ε = 8.0 is a **meaningful academic-level guarantee**
- (ε, δ) = (8.0, 10⁻⁵) is comparable to published FL papers in finance domain

### 3.6 Non-IID Data Split (Sector-Based)

| Client | Sectors | n_stocks | Data Characteristic |
|--------|---------|----------|---------------------|
| Banking | HDFC Bank, SBI, ICICI, etc. | 15 | High correlation, macro-sensitive |
| Finance | Bajaj Finance, HDFC, etc. | 10 | Moderate correlation |
| IT | TCS, Infosys, Wipro, etc. | 6 | USD-sensitive, different cycle |
| Others | Pharma, FMCG, Auto, Metals | 19 | Mixed, low inter-sector correlation |

This is **non-IID by design** — exactly the challenging setting where FedProx shines.

---

## 4. UI Components Breakdown

### 4.1 Metric Cards (4)
| Metric | Value | Badge |
|--------|-------|-------|
| FL Rounds | 50 | CONVERGED |
| Privacy ε | 8.0 | STRONG / MODERATE / WEAK PRIVACY |
| Global Sharpe | Ensemble FL Sharpe | EXCELLENT / GOOD / POOR |
| Clients | 4 | ACTIVE |

### 4.2 Client Info Cards (4 cards)
- Each card: left border in client color, gradient background
- Shows: client name, sectors (comma-separated), n_stocks
- Hover effect: slight Y-translation + shadow increase
- Grid layout (2×2 on desktop)

### 4.3 Convergence Chart (LineChart, 50 rounds)
| Line | Style | Represents |
|------|-------|-----------|
| FedProx | Solid, thick | Global model (FedProx strategy) |
| FedAvg | Dashed, medium | Global model (FedAvg strategy) |
| Client 0 | Dashed, thin, blue | Banking client loss |
| Client 1 | Dashed, thin, orange | Finance client loss |
| Client 2 | Dashed, thin, purple | IT client loss |
| Client 3 | Dashed, thin, teal | Others client loss |

- X-axis: FL round (1–50)
- Y-axis: Training loss
- **Key insight visible:** FedProx global line converges smoother and lower

### 4.4 Fairness Comparison (BarChart)
- Grouped bars per client: **"With FL" (orange) vs "Without FL" (gray)**
- X-axis: Client names
- Y-axis: Sharpe Ratio
- **Key insight visible:** All 4 clients improve with FL — collaborative learning is beneficial even with privacy

---

## 5. Data Flow

```
NIFTY 50 Data (all_close_prices.csv)
        ↓
Split by sector → 4 client datasets (non-IID)
        ↓
Each client: Local training (FLClient.train())
        ↓
DP-SGD: Gradient clipping + noise injection (privacy.py)
        ↓
Gradients sent to FLServer
        ↓
FLServer.aggregate() → FedProx / FedAvg weighted merge
        ↓
Repeat 50 rounds
        ↓
Record: per-round loss (6 curves), final Sharpe per client (with/without FL)
        ↓
/api/fl-summary → JSON
        ↓
Federated.tsx → Convergence Chart + Fairness BarChart
```

---

## 6. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| Client has very few stocks (IT = 6) | Weighted less in FedAvg aggregation |
| Privacy ε < 1 | "STRONG PRIVACY" badge (very high noise) |
| Privacy ε > 10 | "WEAK PRIVACY" badge (less protection) |
| API fails | Error state with AlertTriangle |
| Client loss doesn't converge | Still shown — professor can see divergence case |

---

## 7. Key Academic Concepts (Professor Questions)

**Q: Why not just train one central model on all data?**
> In real-world: fund houses cannot legally share raw trade data. FL solves this — model improves without data leaving each client.

**Q: What does ε=8.0 actually mean?**
> For any two neighboring datasets (differing by 1 record), the model's output distributions are at most e^8 ≈ 2981 times different. At ε=8, any adversary trying to reconstruct training data faces this bound. It's a weaker guarantee than ε=1 but stronger than no privacy.

**Q: Why FedProx over FedAvg?**
> Because our 4 clients have non-IID data (banking stocks ≠ IT stocks). FedAvg assumes IID — its global model drifts toward whichever client has most data. FedProx's proximal term keeps all clients anchored to the global model even when local data distributions differ.

---

## 8. What Makes This Tab Impressive

- **Dual strategy comparison** (FedProx vs FedAvg) — not just "we used FL", but we compared strategies
- **DP-SGD implementation from scratch** in PyTorch (not using Flower library)
- **Non-IID sector split** — realistic, not artificial IID assumption
- **Fairness chart** visually proves FL benefit — all 4 clients improve
- Connects to **real regulatory constraints** (SEBI, GDPR) — not just academic exercise

---

*Tab: Federated Learning | Route: `/fl` | File: `dashboard/src/pages/Federated.tsx`*
