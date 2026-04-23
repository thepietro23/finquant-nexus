# Stress Testing Tab
### Route: `/stress`

---

## 1. Purpose

Stress Testing tab yeh batata hai ki **FINQUANT-NEXUS ka portfolio extreme market conditions mein kaise behave karta hai**. Yeh tab Monte Carlo simulation aur 4 historical crash scenarios ke through portfolio ki **resilience aur downside risk** measure karta hai.

**Project mein role:**
- Portfolio optimization sirf returns ke baare mein nahi hoti — **risk management equally important hai**
- Yeh tab professors ko dikhata hai ki system **real-world black swan events** (2008 crash, COVID) ke liye tested hai
- VaR aur CVaR — industry-standard risk metrics hain jo real hedge funds use karte hain

---

## 2. Target Users & Usage

**Target Users:**
- Risk managers / quantitative analysts
- Professors evaluating financial robustness of the model
- Investors who want to understand downside scenarios

**Real Usage Flow:**

```
User opens Stress Testing tab
        ↓
Sees empty state with CTA (Configure & Generate)
        ↓
Adjusts n_stocks (2–47) and n_simulations (100–50,000)
        ↓
Clicks "Generate Stress Test" → API call with loading spinner
        ↓
Results appear: 3 metric cards (VaR, CVaR, Survival Rate)
        ↓
Monte Carlo Fan Chart shows 30 simulated portfolio paths
        ↓
Scenario Table shows 4 crash scenarios with color-coded rows
        ↓
User clicks on VaR card → Explanation panel opens
        ↓
Reads per-scenario: Mean Return, VaR 95%, CVaR 95%, Survival %
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component UI |
| Recharts | LineChart (Monte Carlo fan chart — 30 simulated paths) |
| Framer Motion | Animated scenario table rows, metric card expansion |
| Tailwind CSS v4 | Color-coded risk severity (green/amber/red/dark red) |

### 3.2 Backend API Called

| Endpoint | Method | Payload | Returns |
|---|---|---|---|
| `/api/stress-test` | POST | `{n_stocks, n_simulations}` | 4 scenario results (VaR, CVaR, Survival, Mean Return) |

**Backend file:** `src/gan/stress.py`

### 3.3 Monte Carlo Simulation

**Algorithm:**
```
For each scenario (Normal / 2008 / COVID / Flash):
    1. Set scenario-specific parameters:
       - daily_vol (volatility)
       - correlation_spike (how correlated stocks become in crisis)
       - n_days (simulation horizon)
    
    2. For i in range(n_simulations):
       - Generate daily_returns using multivariate normal distribution
       - Apply correlation matrix (baseline + spike)
       - Compute portfolio path: ∏(1 + r_t)
       - Record final portfolio value
    
    3. Compute risk metrics:
       - VaR_95 = 5th percentile of all final values
       - CVaR_95 = Mean of values below VaR threshold
       - Survival Rate = % of simulations above minimum threshold (0.85)
       - Mean Return = Average final portfolio value
```

### 3.4 Four Crash Scenarios

| Scenario | Daily Volatility | Correlation Spike | Represents |
|----------|-----------------|-------------------|------------|
| **Normal** | ~18% annualized | Baseline historical | Regular market conditions |
| **2008 Crisis** | 3.5% daily | +30% | Lehman Brothers collapse |
| **COVID Crash** | 5.0% daily | +40% | March 2020 pandemic crash |
| **Flash Crash** | 8.0% daily | Extreme (5-day event) | Sudden liquidity crisis |

### 3.5 Risk Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **VaR 95%** | 5th percentile loss | "In 95% of cases, loss won't exceed X%" |
| **CVaR 95%** | E[loss \| loss > VaR] | "If we ARE in the worst 5%, average loss is X%" |
| **Survival Rate** | % paths > 0.85 threshold | "How many simulations stay above minimum acceptable value" |

**Why CVaR > VaR:**
- VaR tells us the threshold — CVaR tells us what happens beyond it
- CVaR is a **coherent risk measure** (VaR is not) — standard in Basel III banking regulation

### 3.6 Client-Side Monte Carlo (Illustrative)

Before user runs the API, a **client-side preview** is shown:
```javascript
// 50 random paths × 60 days
// Gaussian random returns (σ = 0.015 per day)
// Rendered as LineChart to show what fan chart looks like
```
This is replaced by real API results after generation.

---

## 4. UI Components Breakdown

### 4.1 Control Card
- **n_stocks input** (2–47): How many stocks to include in simulation
- **n_simulations input** (100–50,000): More = more accurate, slower
- Recommended defaults: n_stocks=20, n_simulations=1000
- "Generate Stress Test" button → disabled during loading

### 4.2 Metric Cards (3, appear after run)
| Metric | Good Range | Badge |
|--------|-----------|-------|
| VaR (95%) | < 15% | LOW RISK / MODERATE / HIGH RISK |
| CVaR (95%) | < 20% | THIN TAILS / FAT TAILS / SEVERE |
| Survival Rate | > 80% | — (raw percentage) |

Each card expandable → MetricInfoPanel with definition + formula

### 4.3 Monte Carlo Fan Chart
- **30 simulated paths** displayed as lines (LineChart)
- X-axis: Trading days (60-day horizon)
- Y-axis: Portfolio value (1.0 = starting value)
- One thick red dashed line highlighted as "typical worst path"
- Faded thin lines = other simulation paths

### 4.4 Scenario Results Table
- 4 rows: Normal / 2008 Crisis / COVID Crash / Flash Crash
- **Color coding:**
  - Normal = green border
  - 2008 = amber border
  - COVID = red border
  - Flash = dark red border
- **Survival bar**: Animated fill, width = survival_rate%, color by risk level
- Row background tinted by danger level

---

## 5. Data Flow

```
User sets n_stocks=20, n_simulations=1000
        ↓
POST /api/stress-test → src/gan/stress.py
        ↓
For each of 4 scenarios:
  Monte Carlo simulation (1000 paths × 60 days)
        ↓
VaR, CVaR, Survival Rate computed per scenario
        ↓
JSON response → StressTesting.tsx
        ↓
Metric cards updated + Scenario table rendered
        ↓
Fan chart redrawn with actual simulation paths
```

---

## 6. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| n_stocks < 2 or > 47 | Toast warning, clamped to valid range |
| n_simulations < 100 | Toast: "Too few simulations for reliable results" |
| n_simulations > 50,000 | Toast: "May take very long to compute" |
| API timeout | Error state with retry button |
| All simulations survive | Survival Rate = 100%, green badge |
| Portfolio collapses in all runs | Survival Rate = 0%, dark red badge |

---

## 7. Performance Notes

- **n_simulations = 1000**: Fast (~2–3 seconds) — recommended for demo
- **n_simulations = 10,000**: More accurate (~15–20 seconds)
- **n_simulations = 50,000**: Production-grade accuracy (~2 minutes)
- 4 scenarios run **sequentially** in backend (parallelization is future improvement)

---

## 8. What Makes This Tab Impressive

- **Industry-standard metrics** — VaR and CVaR are used by actual banks and hedge funds
- **4 real historical crash calibrations** — not synthetic — based on actual 2008 and COVID volatility data
- **User-configurable** — number of stocks and simulations adjustable live
- **Survival rate** concept is intuitive — even non-technical professors instantly understand "82% of paths survived COVID"
- Shows the portfolio is **not just return-optimized** but **risk-aware**

---

*Tab: Stress Testing | Route: `/stress` | File: `dashboard/src/pages/StressTesting.tsx`*
