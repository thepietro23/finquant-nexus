# Graph Visualization Tab
### Route: `/graph`

---

## 1. Purpose

Graph Visualization tab **NIFTY 50 stocks ke beech relationships ko ek interactive network graph ke roop mein dikhata hai**. Yeh tab demonstrate karta hai ki **T-GAT (Temporal Graph Attention Network)** stock correlations, sector memberships, aur supply-chain connections ko kaise ek graph structure mein encode karta hai.

**Project mein role:**
- Most portfolio systems stocks ko **independent treat karte hain** — yeh tab uss assumption ko galat prove karta hai
- HDFC Bank girne par doosre banking stocks bhi girte hain — yeh correlation graph mein **visible** hai
- T-GAT is graph ko learn karta hai aur **richer stock embeddings** banata hai jo RL agents use karte hain
- Yeh tab project ka **most visually impressive** component hai — interactive force-directed physics simulation

---

## 2. Target Users & Usage

**Target Users:**
- Financial analysts studying stock interdependencies
- ML researchers understanding GNN input structure
- Professors evaluating graph-based approach to portfolio management

**Real Usage Flow:**

```
User opens Graph Visualization tab
        ↓
Force-directed simulation runs (150 frames) → graph stabilizes
        ↓
User sees 44 nodes arranged in sector clusters
        ↓
Toggles edge type buttons: Sector / Supply Chain / Correlation
        ↓
Hovers over a node (e.g., RELIANCE) → connected nodes highlight
        ↓
Clicks RELIANCE node → Right panel shows details
        ↓
Reads: sector, connections count, portfolio weight, neighbor list
        ↓
Toggles off Sector edges → Only supply-chain/correlation edges visible
        ↓
Scrolls to Graph Stats card → Reads density, avg degree, edge counts
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component UI |
| SVG (native browser) | Entire graph rendered as SVG (900×600 viewBox) |
| Custom Physics Engine | Force-directed simulation (150 frames, no D3 dependency) |
| Framer Motion | Node detail panel slide-in animation |
| CSS Animations | `flowDash` keyframe — animated dashed edge lines |
| Tailwind CSS v4 | Sector color legend, panel styling |

> **No D3.js used** — physics simulation written entirely in TypeScript

### 3.2 Backend API Called

| Endpoint | Method | Returns |
|---|---|---|
| `/api/gnn-summary` | GET | nodes[], edges[], graph stats, sector connectivity, degree distribution |

**Backend files:** `src/graph/builder.py`, `src/models/tgat.py`

### 3.3 Graph Construction (`src/graph/builder.py`)

**3 Types of Edges:**

**1. Sector Edges (Static)**
```
All stocks in the same NIFTY sector → fully connected within sector
Example: HDFCBANK ↔ SBIN ↔ ICICIBANK ↔ KOTAKBANK (all Banking)
~160 edges total
Weight = 1.0 (uniform sector relationship)
```

**2. Supply-Chain Edges (Manual, Static)**
```
30 curated business relationships:
  TATASTEEL → MARUTI (steel supplier to auto manufacturer)
  RELIANCE → ASIAN PAINTS (petrochemical supplier to paint co.)
  NTPC → HINDALCO (power supplier to aluminum smelter)
  ...
~54 edges total
Weight = 0.5–1.0 (relationship strength)
```

**3. Correlation Edges (Dynamic)**
```
For each pair (i, j) of stocks:
  Compute 60-day rolling Pearson correlation
  If |correlation| > 0.4 → add edge
  Weight = |correlation value|
Dynamic: changes as market conditions shift
```

### 3.4 T-GAT — Temporal Graph Attention Network (`src/models/tgat.py`)

**Architecture:**
```
Input: 44 stock nodes × 21 features (technical indicators)
        ↓
RelationalGATLayer — Separate GATConv per edge type
  - GATConv for sector edges    (attention on sector neighbors)
  - GATConv for supply edges    (attention on supply-chain neighbors)
  - GATConv for correlation edges (attention on correlated neighbors)
        ↓
Concatenate outputs + linear transform → 64-dim embedding per stock
        ↓
GRU layer — Encodes temporal sequence (t-1, t-2, ... t-k) of embeddings
        ↓
Final: 64-dim temporal embedding per stock (captures dynamics)
        ↓
Fed into RL environment as enriched state representation
```

**Why Graph Attention (not simple GCN):**
- GAT learns **which neighbors matter more** (attention weights)
- HDFC Bank's correlation with ICICI Bank may be more informative than with Infosys
- Attention weights are learnable — model decides relationship importance

**Key benefit:**
- Without T-GAT: RL sees 21 indicators per stock (independent)
- With T-GAT: RL sees 64-dim embedding that encodes neighborhood context
- Better embeddings → better RL allocation decisions

### 3.5 Custom Physics Simulation

**Parameters:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| Repulsion | 800 | Pushes nodes apart |
| Attraction | 0.005 × edge_weight | Pulls connected nodes together |
| Center gravity | 0.01 | Prevents nodes from flying off screen |
| Damping | 0.8 per frame | Slows velocities → convergence |
| Frames | 150 | Simulation runs then freezes |
| Bounds | 40px margin | Nodes stay within SVG boundaries |

**Initial Layout:**
- Nodes grouped by sector in circular arc segments
- Per-ticker random jitter added → prevents overlap
- **Deterministic** — same layout on every load (seeded jitter)

---

## 4. UI Components Breakdown

### 4.1 Edge Type Toggle Buttons (3)
| Button | Color | Edge Count | Effect |
|--------|-------|-----------|--------|
| Sector | Orange | ~160 | Sector cluster groupings |
| Supply Chain | Purple | ~54 | B2B business relationships |
| Correlation | Teal | Dynamic | |r| > 0.4 pairs |

- Each toggle: includes/excludes that edge type from the graph
- Toast notification on toggle

### 4.2 SVG Graph (Main Visualization)

**Nodes:**
- Circle, size = degree (more connections = bigger node)
- Color = sector (10 unique colors)
- High-degree nodes show ticker label always
- Low-degree nodes show label only on hover
- **Selected:** Dashed border ring + glow filter
- **Inactive (when another hovered):** 0.2 opacity

**Edges:**
- Bezier curves (not straight lines) → cleaner visual
- **Active edge:** color (sector=orange/supply=purple/corr=teal), dashed animation, opacity 0.8
- **Inactive edge (not connected to hover/selected):** opacity 0.08
- **Moving data packets:** White circles animate along active edges (animateMotion, 1.6s cycle)

### 4.3 Node Detail Panel (Right Side, Animated)
When a node is clicked:
- Slide-in animation from right (0.22s)
- **Header:** Ticker + colored sector dot
- **Sector badge**
- **Stats grid:** Connections count · Portfolio Weight % · Neighbors count
- **Connected stocks list:** Up to 15 neighbors, each as colored pill
  - Color = edge type (orange/purple/teal left border)
  - Shows ticker + edge weight

**Empty state (no selection):** Icon + "Click any node to explore" hint

### 4.4 Legend Card
- All 10 sectors with color dots and stock counts
- Banking(15) / Finance(10) / IT(6) / Telecom(2) / Pharma(4) / FMCG(3) / Energy(3) / Auto(2) / Metals(3) / Infrastructure(2)

### 4.5 Graph Stats Card
| Stat | Description |
|------|-------------|
| Total Nodes | All NIFTY stocks in graph |
| Total Edges | All edge types combined |
| Visible Edges | Currently displayed (per toggle state) |
| Avg Degree | Average connections per node |
| Density | Edges / Max possible edges |

---

## 5. Data Flow

```
NIFTY 50 Historical Prices (all_close_prices.csv)
        ↓
graph/builder.py:
  - Sector edges: from stocks.py sector registry
  - Supply-chain: hardcoded 30 relationships
  - Correlation: 60-day rolling Pearson on price returns
        ↓
Graph object: nodes (44) + edges (200+)
        ↓
tgat.py: T-GAT encodes graph → 64-dim embeddings per stock
        ↓
/api/gnn-summary → nodes[], edges[], top_connections[], stats
        ↓
GraphVisualization.tsx
        ↓
Physics simulation → SVG rendering
        ↓
User interaction: click/hover/toggle
```

---

## 6. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| All edge types toggled off | Empty graph (nodes visible, no edges) |
| Stock with no correlation edges | Node shown, no teal edges |
| API load fails | Error state, spinner |
| Node clicked twice | Deselects (toggle behavior) |
| Very small screen | SVG viewBox scales proportionally |
| High-degree node hover | All its edges highlight, others fade to 0.08 opacity |
| Node outside SVG bounds | Clamped to 40px margin by physics |

---

## 7. Performance Notes

- **150 physics frames** run synchronously on component mount
- After 150 frames, simulation freezes — **no ongoing CPU usage**
- All 44 nodes + 200+ edges rendered as SVG elements
- Edge animations (CSS keyframes) are GPU-accelerated
- Data packet animations use `animateMotion` (SVG native) — minimal JS overhead
- **No D3.js** — custom physics is lighter and more controllable

---

## 8. What Makes This Tab Impressive

- **3 edge types** — most graph visualizations show only one relationship type
- **Interactive physics** — nodes move realistically, then settle
- **Moving data packets** — animated white circles traveling along edges make it visually dynamic
- **Supply-chain edges** — manually curated 30 real business relationships (not just statistical)
- **Click-to-explore** — clicking any stock reveals its complete neighborhood
- **Edge type toggles** — professor can isolate sector vs correlation view instantly
- **No external graph library (D3)** — physics engine written in TypeScript from scratch

---

*Tab: Graph Visualization | Route: `/graph` | File: `dashboard/src/pages/GraphVisualization.tsx`*
