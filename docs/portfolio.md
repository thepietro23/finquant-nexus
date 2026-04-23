# Portfolio Overview Tab
### Route: `/` (Home/Index)

---

## 1. Purpose

Portfolio Overview tab **project ka final output** hai — yeh tab dikhata hai ki FINQUANT-NEXUS v4 ne NIFTY 50 stocks par apni intelligence (RL + GNN + Federated + Sentiment) apply karke **kaisa optimized portfolio banaya**.

**Project mein role:**
- Yeh tab **entry point** hai — professor ya user sabse pehle yahi dekhta hai
- Poore system ka **summary dashboard** — metrics, holdings, sector allocation, aur benchmark comparison sab ek jagah
- Ek **interactive investment simulator** bhi hai jo real returns calculate karta hai

---

## 2. Target Users & Usage

**Target Users:**
- Portfolio managers / financial analysts
- Professors / evaluators reviewing the project
- Retail investors looking at NIFTY 50 allocation

**Real Usage Flow:**

```
User lands on Portfolio tab
        ↓
Sees 5 key metrics (Sharpe, Sortino, Return, Volatility, Drawdown)
        ↓
Clicks any metric card → Expandable explanation panel opens
        ↓
Scrolls to Holdings Table → Clicks a stock (e.g., HDFCBANK)
        ↓
Stock detail panel expands → Price, 52W range, Sharpe, history chart
        ↓
Opens Investment Simulator → Enters ₹1,00,000 + start date
        ↓
Clicks Calculate → Sees per-stock profit/loss breakdown
        ↓
Clicks "Growth Chart" → Portfolio vs NIFTY 50 vs FD comparison
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component-based UI |
| Recharts | BarChart (sectors), AreaChart (stock price, growth), LineChart (portfolio growth) |
| Framer Motion | Expand/collapse animations on metric cards, stock panels |
| TanStack React Query | API state management, caching |
| Tailwind CSS v4 | Styling, color-coded badges |

### 3.2 Backend APIs Called

| API Endpoint | Method | What It Returns |
|---|---|---|
| `/api/portfolio-summary` | GET | Sharpe, Sortino, returns, volatility, drawdown, holdings[], sector_weights |
| `/api/refresh-data` | GET | Downloads latest prices from Yahoo Finance (yfinance) |
| `/api/stock/{ticker}` | GET | Price history (60-day), 52W high/low, daily change, Sharpe, drawdown |
| `/api/portfolio-growth` | POST | Portfolio vs NIFTY vs FD growth series over time |

### 3.3 Key Techniques Used

**Financial Metrics Computation (backend: `src/utils/metrics.py`):**
- **Sharpe Ratio** = (Annual Return − 7% risk-free) / Annual Volatility
- **Sortino Ratio** = (Return − Risk-free) / Downside Deviation only
- **Max Drawdown** = Largest peak-to-trough decline in portfolio value
- **Volatility** = Annualized standard deviation (252 trading days/year)

**Holdings & Weights:**
- Equal-weight baseline: 1/N per stock
- Sector weights: sum of stock weights per sector
- Sorted by cumulative return (descending)

**Investment Simulator (client-side logic):**
```
User Input: ₹1,00,000, Start Date: 2023-01-01
        ↓
Softmax normalization applied to portfolio weights
        ↓
Per-stock invested amount = weight × total amount
        ↓
1-year return applied per stock from historical data
        ↓
Output: Final value, Profit/Loss, Return % per stock
```

**Portfolio Growth Chart:**
- 3 comparison series over selected time period:
  - Our RL Portfolio (FINQUANT-NEXUS weights)
  - NIFTY 50 Index (equal market-cap tracking)
  - Fixed Deposit @ 7% (risk-free benchmark)

---

## 4. UI Components Breakdown

### 4.1 Metric Cards (5 cards)
| Metric | What it Means | Badge Thresholds |
|--------|--------------|-----------------|
| Sharpe Ratio | Risk-adjusted return | >1.5=EXCEPTIONAL, 1-1.5=EXCELLENT, 0.5-1=GOOD |
| Sortino Ratio | Downside risk adjusted | >2=EXCEPTIONAL, 1-2=EXCELLENT |
| Annual Return % | Yearly portfolio gain | >20%=EXCELLENT, 10-20%=GOOD |
| Volatility % | Price fluctuation risk | <15%=LOW, 15-25%=MODERATE, >25%=HIGH |
| Max Drawdown % | Worst peak-to-trough drop | <10%=EXCELLENT, 10-20%=ACCEPTABLE |

- Each card **clickable** → MetricInfoPanel slides in with definition, formula, and interpretation

### 4.2 Holdings Table
- **44 NIFTY stocks** (44 of 50 with sufficient data)
- Columns: Rank · Ticker · Sector (color dot) · Weight % · Return %
- Return column: animated green/red bar fills behind percentage
- **Clicking any row** → Stock detail panel expands below
  - Current price, daily change, previous close
  - 52-week high/low bar with current price marker
  - 60-day price history (AreaChart)
  - Key risk metrics grid

### 4.3 Sector Weights Chart
- Horizontal BarChart (Recharts)
- 11 sectors, each with unique color
- Sorted descending by weight percentage

### 4.4 Investment Simulator
- Preset amount buttons: ₹10K / ₹50K / ₹1L / ₹5L / ₹10L / ₹1M
- Custom amount input + date picker
- **Calculate** → Shows 4 summary cards + sector breakdown + per-stock table
  - Table sortable: Top Gainers / Top Losers / By Weight
- **Growth Chart** → Full time-series comparison with 3 series

---

## 5. Data Flow

```
Yahoo Finance (yfinance)
        ↓
all_close_prices.csv (cached data, NIFTY 50, 2015–2025)
        ↓
src/utils/metrics.py → Sharpe, Sortino, Volatility, Drawdown
        ↓
/api/portfolio-summary (FastAPI)
        ↓
Portfolio.tsx (React)
        ↓
Recharts Charts + Metric Cards + Holdings Table
        ↓
User clicks stock → /api/stock/{ticker} → Detail Panel
        ↓
User runs simulator → /api/portfolio-growth → Growth Chart
```

---

## 6. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| Stock has no data | Excluded from holdings (44 of 50 shown) |
| Negative daily return | Red color in table, downward arrow |
| Refresh button while loading | Disabled with spinner, status badge shows rows added |
| Invalid simulator amount | Clamped to valid range, toast warning |
| API timeout | Error state displayed with backend URL hint |
| Stock detail fetch fails | Error shown inline within expanded panel |

---

## 7. Performance Notes

- **CSV cache** (`all_close_prices.csv`) prevents repeated Yahoo Finance API calls
- Portfolio summary is computed once on startup, cached in memory
- Stock detail fetched **on-demand** (only when user clicks a stock)
- Portfolio growth uses 120s timeout (heavy computation for long date ranges)
- Holdings table renders 44 rows — no virtualization needed at this scale

---

## 8. What Makes This Tab Impressive

- **Real NIFTY 50 data** (2015–2025, ~10 years of price history)
- **3-way benchmark comparison** (RL Portfolio vs Market vs Fixed Deposit)
- **Interactive simulator** — user can input their own ₹ amount and see results
- Metrics are not just numbers — they have **threshold-based interpretations** with explanations
- Every stock has a **clickable detail panel** with 60-day chart and risk metrics

---

*Tab: Portfolio Overview | Route: `/` | File: `dashboard/src/pages/Portfolio.tsx`*
