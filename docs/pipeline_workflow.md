# Pipeline Workflow Tab
### Route: `/workflow`

---

## 1. Purpose

Pipeline Workflow tab **FINQUANT-NEXUS v4 ka poora end-to-end system ek animated visualization mein dikhata hai**. Yeh tab batata hai ki data kaise enter hota hai, kaise process hota hai, kaise model train hote hain, aur kaise final portfolio output aata hai — **step-by-step, stage-by-stage**.

**Project mein role:**
- Yeh tab **system architecture ka visual explanation** hai — code samjhe bina bhi koi poora flow samajh sakta hai
- Professor ke liye yeh tab ideal hai — woh ek baar play button dabata hai aur poora system explain ho jaata hai
- **No API calls** — purely client-side animation, always available regardless of backend status
- Integration tab hai — **sab components (RL, GNN, FL, Sentiment) ek diagram mein dikhe**

---

## 2. Target Users & Usage

**Target Users:**
- Professors / evaluators getting a high-level system overview
- Stakeholders / non-technical audience
- Students understanding how all components connect

**Real Usage Flow:**

```
User opens Pipeline Workflow tab
        ↓
Sees complete system diagram (15 nodes, 19 connections)
        ↓
Clicks "Play" button → Animation starts
        ↓
Stage 1: NIFTY 50 Data + Google News nodes activate (blue glow)
        ↓
Stage 2: Feature Engineering + FinBERT activate
        ↓
Stage 3: Graph Builder activates
        ↓
Stage 4: T-GAT Model activates
        ↓
Stage 5: RL Environment activates
        ↓
Stages 6-10: All 5 RL Agents (PPO/SAC/TD3/A2C/DDPG) activate
        ↓
Stage 11: Ensemble node activates (combines all agents)
        ↓
Stage 12: Federated Learning + Portfolio Output activate
        ↓
Animation complete → "done" message shown
        ↓
User clicks any node → Right panel shows technical details
        ↓
User clicks stage buttons in sidebar → Jumps to that stage
```

---

## 3. Tools & Techniques

### 3.1 Frontend Stack
| Tool | Usage |
|------|-------|
| React 19 + TypeScript | Component UI |
| SVG (native browser) | Entire pipeline diagram (900×1080 viewBox) |
| CSS Animations | Dashed edge flow animation (`flowDash` keyframe) |
| SVG `animateMotion` | White data packets traveling along edges |
| Framer Motion | Stage label transitions, detail panel slide-in |
| Tailwind CSS v4 | Section color bands, node styling |

> **No external diagram library** — SVG + CSS + custom React state

### 3.2 No Backend API
- This tab is **100% client-side**
- All node data, edge paths, stage sequences are hardcoded in the component
- Always loads instantly, no dependency on backend server status

### 3.3 Pipeline Architecture — 15 Nodes

| # | Node | Icon | Color | Layer |
|---|------|------|-------|-------|
| 1 | NIFTY 50 Data | 📊 | Blue | Data Sources |
| 2 | Google News | 📰 | Purple | Data Sources |
| 3 | Feature Engineering | ⚙️ | Cyan | Processing |
| 4 | FinBERT Sentiment | 💬 | Pink | Processing |
| 5 | Graph Builder | 🕸️ | Amber | Graph Layer |
| 6 | T-GAT Model | 🧠 | Teal | Graph Layer |
| 7 | RL Environment | 🎮 | Purple | RL Setup |
| 8 | PPO Agent | 🤖 | Blue | RL Agents |
| 9 | SAC Agent | 🤖 | Orange | RL Agents |
| 10 | TD3 Agent | 🤖 | Green | RL Agents |
| 11 | A2C Agent | 🤖 | Red | RL Agents |
| 12 | DDPG Agent | 🤖 | Indigo | RL Agents |
| 13 | Ensemble | ⭐ | Emerald | Aggregation |
| 14 | Federated Learning | 🔒 | Blue | Privacy Layer |
| 15 | Portfolio Output | 💼 | Green | Final Output |

### 3.4 Pipeline Connections — 19 Edges

| From | To | Meaning |
|------|-----|---------|
| NIFTY 50 Data | Feature Engineering | Price data → technical indicators |
| Google News | FinBERT | Raw news → sentiment scoring |
| Feature Engineering | Graph Builder | Indicators used to compute correlations |
| Feature Engineering | RL Environment | 21 features as state input |
| FinBERT | Graph Builder | Sentiment as node features |
| Graph Builder | T-GAT Model | Graph structure fed to GNN |
| T-GAT Model | RL Environment | 64-dim embeddings added to state |
| RL Environment | PPO Agent | Environment interactions |
| RL Environment | SAC Agent | Environment interactions |
| RL Environment | TD3 Agent | Environment interactions |
| RL Environment | A2C Agent | Environment interactions |
| RL Environment | DDPG Agent | Environment interactions |
| PPO Agent | Ensemble | PPO weights fed to aggregator |
| SAC Agent | Ensemble | SAC weights fed to aggregator |
| TD3 Agent | Ensemble | TD3 weights fed to aggregator |
| A2C Agent | Ensemble | A2C weights fed to aggregator |
| DDPG Agent | Ensemble | DDPG weights fed to aggregator |
| Ensemble | Federated Learning | Combined weights enter FL privacy layer |
| Federated Learning | Portfolio Output | Privacy-preserved final allocation |

### 3.5 Animation System — 9 Stages

| Stage | Label | Nodes Activated |
|-------|-------|----------------|
| ① | Data Ingestion | NIFTY 50 Data, Google News |
| ② | Data Processing | Feature Engineering, FinBERT |
| ③ | Graph Construction | Graph Builder |
| ④ | Graph Learning | T-GAT Model |
| ⑤ | RL Environment Setup | RL Environment |
| ⑥ | Agent Training (1/2) | PPO, SAC, TD3 |
| ⑦ | Agent Training (2/2) | A2C, DDPG |
| ⑧ | Ensemble Aggregation | Ensemble |
| ⑨ | Portfolio Output | Federated Learning, Portfolio |

**Timing:** 1.8 seconds per stage → total animation ~16 seconds

**Edge animation on active edges:**
- Dashed line animation (`flowDash` CSS keyframe)
- White data packet circles travel from source → target (SVG animateMotion, 1.6s cycle)

---

## 4. UI Components Breakdown

### 4.1 Header Controls
- **Title:** "Pipeline Visualization"
- **Subtitle:** "End-to-end FINQUANT-NEXUS v4 workflow"
- **Reset button:** Clears all active nodes, resets to initial state (rotating icon during play)
- **Play / Pause / Replay button:**
  - Play → starts sequential animation
  - During play → shows Pause (click to freeze current stage)
  - After completion → shows Replay

### 4.2 Stage Label Banner (Animated)
- Shows current stage label (e.g., "⑤ RL Environment Setup")
- **Framer Motion fade + slide** transition between stage labels
- Initial: hint text "Click Play to start"
- Complete: "✓ Pipeline complete"

### 4.3 SVG Pipeline Diagram (900×1080)

**Section backgrounds (7 colored bands):**
- Each layer has a soft background band with label (e.g., "Data Sources", "RL Agents")
- Helps professor visually group components by function

**Node rendering:**
- Inactive: light gray fill, gray border
- Active: colored fill + glow filter (`feGaussianBlur` SVG filter)
- Selected: dashed selection ring (animated rotation)
- Color transitions: CSS transition on fill (0.3s)

**Edge rendering:**
- Bezier curve paths with arrowhead markers
- Inactive: solid, low opacity (0.25)
- Active: dashed animation + high opacity (0.8) + color (source node color)
- Data packets: white circle, 3px radius, infinite animateMotion loop

### 4.4 Progress Stepper (Right edge of SVG)
- 9 dots (one per stage), vertically arranged
- Completed: green fill
- Current: primary color
- Pending: gray
- Vertical line between dots: green segment fills downward as stages complete

### 4.5 Right Panel — Node Detail Card (Animated Slide-in)
When user clicks any node:
- **Header:** Colored bar + icon + node name
- **Tech Stack badge:** (e.g., "PyTorch + SB3" for RL agents)
- **Input box:** What data this node receives
- **Output box:** What this node produces
- **Key Details list:** 5–7 bullet points
  - Example for T-GAT: "3 relational GAT layers", "64-dim stock embeddings", "GRU temporal encoding"

### 4.6 Pipeline Stages List (Right Panel)
- 9 clickable stage buttons (① to ⑨)
- Status indicator:
  - Pending → gray circle
  - Current → highlighted (primary color)
  - Completed → green circle with checkmark
- Clicking a stage button jumps animation to that stage

### 4.7 System Stats Grid (Right Panel, 2×4)
| Stat | Value |
|------|-------|
| Components | 15 |
| Connections | 19 |
| RL Algorithms | 5 + Ensemble |
| FL Clients | 4 |
| Features | 21 price + 64 GNN + 1 sentiment |
| Tests | 246 / 246 passing |
| NIFTY Stocks | 44+ |
| Privacy | ε=8, δ=10⁻⁵ |

---

## 5. Data Flow (Client-Side Only)

```
Component mounts → No API call
        ↓
Hardcoded: 15 nodes, 19 edges, 9 stage sequences
        ↓
Physics: initial positions computed (deterministic)
        ↓
User clicks Play → setInterval(1800ms)
        ↓
Each tick: step++ → activeNodes = all nodes up through current stage
        ↓
Active edges computed: both endpoints in activeNodes
        ↓
SVG re-renders: active nodes glow, active edges animate
        ↓
Stage label updates with Framer Motion
        ↓
Progress dots update
        ↓
At step 9: animation complete → Replay button shown
        ↓
User clicks any node → selected node set → detail panel slides in
```

---

## 6. Edge Cases & Validations

| Scenario | Handling |
|----------|----------|
| User clicks Play mid-animation | Pauses at current stage |
| User clicks Reset during play | Stops interval, clears all active nodes, step = -1 |
| User clicks stage 5 directly | Jumps to stage 5, activates all nodes up through stage 5 |
| User clicks node while paused | Detail panel opens; animation stays paused |
| User clicks already-selected node | Deselects (toggle behavior) |
| Backend down | No effect — tab has no API dependency |
| Small screen | SVG viewBox scales (proportional) |

---

## 7. Performance Notes

- **No API calls** — instant load, always available
- SVG with 15 nodes + 19 edges: extremely lightweight DOM
- CSS animations (flowDash) are GPU-accelerated
- animateMotion (SVG native) — no JavaScript per-frame cost
- setInterval(1800ms) — minimal CPU overhead
- Framer Motion transitions: hardware-accelerated via CSS transforms

---

## 8. What Makes This Tab Impressive

- **Complete system in one view** — professor sees all 15 components and how they connect
- **Sequential animation** — each stage lights up in order, mimicking real data flow
- **Moving data packets** — white circles travel along edges, making data flow tangible
- **Click-to-explore any node** — technical depth available on demand
- **Stage jump buttons** — professor can skip to any stage for focused Q&A
- **System stats panel** — 246 tests, ε=8 privacy budget — concrete implementation proof
- **No backend dependency** — works even if Python server is off during presentation
- Effectively replaces a system architecture slide with an **interactive, animated version**

---

*Tab: Pipeline Workflow | Route: `/workflow` | File: `dashboard/src/pages/WorkflowViz.tsx`*
