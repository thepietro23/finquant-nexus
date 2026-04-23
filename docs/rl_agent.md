# RL Agent Tab
### Route: `/rl`

---

## 1. Purpose

RL Agent tab **project ka intelligence core** dikhata hai — yahan 5 reinforcement learning algorithms (PPO, SAC, TD3, A2C, DDPG) aur unka combined Ensemble show hota hai. Yeh tab batata hai ki **kaise RL agents NIFTY 50 portfolio allocation seekhte hain** aur ek doosre se kaise compare karte hain.

**Project mein role:**
- RL is the **primary decision-making engine** of FINQUANT-NEXUS
- Yeh tab **training performance, validation results, aur final allocations** ek saath visualize karta hai
- Professor ke liye yeh tab **system ki ML depth** demonstrate karta hai

---

## 2. Target Users & Usage

**Target Users:**
- ML researchers / professors evaluating algorithm design choices
- Financial analysts comparing strategy behavior
- Students understanding RL in finance

**Real Usage Flow:**

```
User opens RL Agent tab
        ↓
Sees default view: PPO algorithm selected
        ↓
Clicks "Ensemble" button → Switches to Ensemble view (best performer)
        ↓
Looks at Comparison Table → Sees all 6 algos side-by-side
        ↓
Clicks "Rewards" chart tab → Sees training Sharpe curves per algo
        ↓
Clicks "Cumulative Returns" tab → Validates against 2022–2023 data
        ↓
Clicks "Weights" tab → Sees final stock allocation for selected algo
        ↓
Scrolls to Sector Donut → Sees how algo distributes across sectors
        ↓
Scrolls to Return Contributions → Top 15 stock-level returns
        ↓
Clicks metric card (e.g., Avg Reward) → Explanation panel opens
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component UI |
| Recharts | AreaChart (rewards), LineChart (cumulative returns), BarChart (weights), PieChart (sector donut) |
| Framer Motion | Animated tab pill, stagger list animation on contributions |
| Tailwind CSS v4 | Algorithm-specific color coding |

### 3.2 Backend API Called

| Endpoint | Method | Returns |
|---|---|---|
| `/api/rl-summary` | GET | All 6 algo metrics, reward curves, weights, sector allocation, stock contributions |

### 3.3 RL Algorithms Implemented

| Algorithm | Type | Key Characteristic |
|-----------|------|-------------------|
| **PPO** | On-policy | Clipped policy gradient — stable, conservative |
| **SAC** | Off-policy | Entropy regularization — explores aggressively |
| **TD3** | Off-policy | Twin critics + delayed actor updates — reduces overestimation |
| **A2C** | On-policy | Advantage Actor-Critic — fast convergence |
| **DDPG** | Off-policy | Deterministic policy — precise continuous actions |
| **Ensemble** | Aggregated | Weighted average of all 5 — best stability & returns |

**Backend implementation:** `src/rl/agent.py` (Stable-Baselines3 wrappers)

### 3.4 RL Environment Design (`src/rl/environment.py`)

```
State Space:
  - 21 technical indicators per stock × 44 stocks
  - Portfolio weights (current allocation)
  - Account balance

Action Space:
  - Continuous: weight adjustment per stock (44-dimensional)
  - Constraints: max position 20%, stop-loss -5%

Reward Function:
  - Sharpe Ratio of portfolio at each step
  - Penalized for: excessive drawdown, high turnover

Training Data:
  - Historical NIFTY 50 prices: 2019–2022 (3 years)
  - Validation: 2022–2023 (unseen during training)

Episodes: 500 per algorithm
```

### 3.5 Portfolio Constraints System
| Constraint | Value | Purpose |
|------------|-------|---------|
| Max Position | 20% | No single stock dominates |
| Stop Loss | -5% | Auto-exit losing position |
| Circuit Breaker | -15% | Halt trading if portfolio drops 15% |
| Transaction Cost | 0.1% | Realistic brokerage simulation |
| Slippage | 0.05% | Market impact simulation |

---

## 4. UI Components Breakdown

### 4.1 Algorithm Selector (6 Buttons)
- Each button: algorithm name + short description
- Selected: colored highlight matching algorithm color
- **PPO** = blue · **SAC** = orange · **TD3** = green · **A2C** = purple · **DDPG** = red · **Ensemble** = emerald
- Clicking → toast notification shows algorithm description

### 4.2 Metric Cards (4, dynamic per algorithm)
| Metric | Meaning | Badges |
|--------|---------|--------|
| Episodes | Training episodes run | raw count |
| Avg Reward | Mean Sharpe during training | EXCELLENT/GOOD/LEARNING/EARLY |
| Sharpe (Val) | Validation-set Sharpe ratio | EXCELLENT/GOOD/POOR |
| Max Drawdown | Worst portfolio drop | EXCELLENT/ACCEPTABLE/HIGH RISK |

### 4.3 6-Algorithm Comparison Table
- All 6 algorithms as rows
- Columns: **Sharpe · Sortino · Annual Return % · Volatility % · Max Drawdown %**
- Horizontal Sharpe bar fills proportionally (best algo = 100% width)
- **"Best" badge** on highest Sharpe row
- **"Recommended" badge** always on Ensemble
- Row highlights when that algo is selected

### 4.4 Performance Chart Tabs (3 tabs)

**Tab 1: Training Rewards**
- AreaChart — 50 episodes on X-axis, Sharpe on Y-axis
- 6 colored area lines, Ensemble = thick solid line
- Shows convergence — all agents improve over training

**Tab 2: Cumulative Returns**
- LineChart — validation period 2022–2023
- 6 algorithm lines + gray dashed Equal-Weight baseline
- Ensemble sits above most individual agents

**Tab 3: Portfolio Weights**
- BarChart — top 15 stock allocations for selected algorithm
- Vertical bars with stock tickers on X-axis
- Switch algorithms → chart re-renders with new weights

### 4.5 Sector Allocation Donut
- PieChart (inner radius 55px, outer 100px)
- 10 sectors with unique colors
- Switches per selected algorithm — shows different sector preferences

### 4.6 Return Contribution List
- Top 15 stocks by contribution to portfolio return
- Animated slide-in (stagger effect per item)
- Each item: rank · sector dot · ticker · sector · weight% · return% · contribution%
- Green up-arrow / Red down-arrow based on direction
- Horizontal bar = abs(contribution) / max_contribution × 100%

---

## 5. Data Flow

```
Historical NIFTY 50 Prices (2019–2022)
        ↓
RL Environment (src/rl/environment.py)
        ↓
5 Agents Train in Parallel (PPO, SAC, TD3, A2C, DDPG)
        ↓ 500 episodes each
Validation on 2022–2023 data
        ↓
Ensemble = Weighted average of 5 agent outputs
        ↓
/api/rl-summary → all metrics, curves, weights, contributions
        ↓
RlAgent.tsx → 6 buttons + 3 chart tabs + comparison table
```

---

## 6. Why Ensemble Outperforms Individual Agents

| Reason | Explanation |
|--------|------------|
| **Variance Reduction** | Different agents make different errors; averaging cancels them out |
| **Diverse Exploration** | SAC explores aggressively, PPO stays conservative — ensemble balances both |
| **Sector Diversity** | Each algo bets on different sectors; ensemble captures best of all |
| **No Single Point of Failure** | If one algo underperforms in a market regime, others compensate |

---

## 7. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| API fetch fails | Error state with AlertTriangle icon + backend URL |
| Algorithm still loading | Skeleton cards + spinner |
| No data for selected algo | Empty state fallback |
| Chart tab with no data | Placeholder message shown |
| Ensemble metrics shown | Avg of 5 agents (not a separate training run) |

---

## 8. What Makes This Tab Impressive

- **5 independent algorithms** train in parallel — not just one model
- **Real validation split** (2022–2023 data never seen during training)
- **Constraint system** makes it realistically deployable (stop-loss, slippage, etc.)
- **Sector heatmap** shows each algo has different "personality" — proves diversity
- **Ensemble design** — standard technique in financial ML, reduces variance significantly

---

*Tab: RL Agent | Route: `/rl` | File: `dashboard/src/pages/RlAgent.tsx`*
