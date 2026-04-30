# Dissertation Diagrams — Mermaid Source Code
## How to use: Go to https://mermaid.live, paste code, screenshot

---

## DIAGRAM 1 — fig_3_1_architecture.png
## System Architecture (Chapter 3.1)

```mermaid
flowchart TD
    subgraph SRC["DATA SOURCES"]
        direction LR
        A1["Yahoo Finance\nNIFTY 50 · 44 Stocks\n2015–2025"]
        A2["News Sources\nGoogle News · Yahoo Finance\nIndian RSS Feeds"]
    end

    subgraph PROC["PROCESSING"]
        direction LR
        B1["Feature Engineering\n21 Technical Indicators\nRSI · MACD · Bollinger · ATR"]
        B2["FinBERT NLP\nSentiment Score\n−1 to +1 per stock"]
    end

    subgraph GNN["GRAPH + T-GAT"]
        C1["Graph Builder\nSector Edges\nSupply Chain Edges\nCorrelation Edges"]
        C2["T-GAT Model\nRelational Attention x 3\nGRU Temporal Encoder\n32-dim Embeddings"]
    end

    subgraph RLENV["RL ENVIRONMENT"]
        D1["Portfolio Env\nGymnasium Compatible\nState · Action · Reward"]
    end

    subgraph AGENTS["RL AGENTS"]
        direction LR
        E1["PPO"]
        E2["SAC"]
        E3["TD3"]
        E4["A2C"]
        E5["DDPG"]
    end

    F1["Ensemble Agent\nAverage of 5 Algorithms"]

    subgraph FL["FEDERATED LEARNING"]
        G1["FedProx Server\n4 Sector Clients\nDP-SGD · epsilon = 8.0"]
    end

    subgraph OUT["OUTPUT"]
        direction LR
        H1["FastAPI Backend\n50+ REST Endpoints"]
        H2["React Dashboard\n8 Interactive Pages"]
    end

    A1 --> B1
    A2 --> B2
    B1 --> C1
    B2 --> D1
    C1 --> C2
    C2 --> D1
    D1 --> E1 & E2 & E3 & E4 & E5
    E1 & E2 & E3 & E4 & E5 --> F1
    F1 --> G1
    G1 --> H1
    H1 --> H2

    style SRC fill:#dbeafe,stroke:#3b82f6
    style PROC fill:#dcfce7,stroke:#22c55e
    style GNN fill:#fef9c3,stroke:#eab308
    style RLENV fill:#fce7f3,stroke:#ec4899
    style AGENTS fill:#ede9fe,stroke:#8b5cf6
    style FL fill:#ffedd5,stroke:#f97316
    style OUT fill:#f0fdf4,stroke:#16a34a
```

---

## DIAGRAM 2 — fig_3_4_sentiment_pipeline.png
## FinBERT Sentiment Pipeline (Chapter 3.5)

```mermaid
flowchart TD
    subgraph SRC["News Sources"]
        direction LR
        N1["Google News\n(RSS Feed)"]
        N2["Yahoo Finance\nNews API"]
        N3["Indian RSS\nMoneycontrol · ET"]
    end

    F["News Fetcher\nThread-safe · 5s Timeout\nDeduplication"]

    T["FinBERT Tokenizer\nMax 512 tokens\nWordPiece encoding"]

    M["FinBERT Model\n(ProsusAI/finbert)\nLocally Cached"]

    S["Score Extraction\nP(positive) minus P(negative)\nRange: −1 to +1"]

    A["Daily Aggregation\nWeighted Average per Stock\n44 stocks x 1 score/day"]

    subgraph OUT["Output"]
        direction LR
        C["SQLite Cache\nTTL: 3 minutes"]
        R["RL Observation Space\nSentiment Feature Vector"]
        D["Dashboard\nMarket Mood · Live Feed\nPortfolio Impact"]
    end

    N1 & N2 & N3 --> F
    F --> T
    T --> M
    M --> S
    S --> A
    A --> C
    A --> R
    A --> D

    style SRC fill:#dbeafe,stroke:#3b82f6
    style OUT fill:#dcfce7,stroke:#22c55e
    style M fill:#fef9c3,stroke:#eab308,stroke-width:2px
    style F fill:#ede9fe,stroke:#8b5cf6
```

---

## DIAGRAM 3 — fig_3_6_tgat.png
## T-GAT Architecture (Chapter 3.7)

```mermaid
flowchart TD
    NF["Node Features\n21 Technical Indicators\n+ Sentiment Score\nper stock per day"]

    subgraph RGAT["RelationalGATLayer — 3 Parallel Attention Heads"]
        direction LR
        G1["Sector\nAttention\nW_sector"]
        G2["Supply Chain\nAttention\nW_supply"]
        G3["Correlation\nAttention\nW_corr"]
    end

    MH["Multi-Head Attention\n8 heads per edge type\nLeakyReLU activation"]

    CAT["Concatenate\nAll edge-type outputs\nfused representation"]

    GRU["GRU Temporal Encoder\n2-layer GRU\nHidden size: 128\nCaptures sequential dynamics"]

    EMB["32-dim Stock Embedding\nOne vector per stock\nper time step"]

    subgraph USE["Used By"]
        direction LR
        RL["RL Observation\nSpace"]
        VIZ["Graph\nVisualization\nDashboard"]
    end

    NF --> RGAT
    G1 & G2 & G3 --> MH
    MH --> CAT
    CAT --> GRU
    GRU --> EMB
    EMB --> RL
    EMB --> VIZ

    style RGAT fill:#ede9fe,stroke:#8b5cf6
    style GRU fill:#fef9c3,stroke:#eab308,stroke-width:2px
    style EMB fill:#dcfce7,stroke:#22c55e,stroke-width:2px
    style USE fill:#dbeafe,stroke:#3b82f6
```

---

## DIAGRAM 4 — fig_3_7_rl_env.png
## RL Environment Cycle (Chapter 3.8)

```mermaid
flowchart LR
    subgraph STATE["Observation Space"]
        S1["21 Technical\nIndicators\n44 stocks"]
        S2["Current Portfolio\nWeights\n44 values"]
        S3["T-GAT\nEmbeddings\n44 x 32-dim"]
        S4["Sentiment\nScores\n44 values"]
    end

    AGT["RL Agent\n(PPO / SAC / TD3\nA2C / DDPG)"]

    subgraph ACT["Action Space"]
        A1["Raw Weight\nVector\n44 dimensions"]
        A2["Softmax\nNormalization\nSum = 1.0"]
    end

    subgraph ENV["Portfolio Environment"]
        E1["Apply\nWeights"]
        E2["Compute\nReturns"]
        E3["Reward\nSharpe minus Penalties"]
    end

    subgraph CON["Constraints Enforced"]
        C1["Max 12%\nper stock"]
        C2["Stop-loss\n-3%"]
        C3["Max Drawdown\n-12%"]
    end

    STATE --> AGT
    AGT --> A1 --> A2
    A2 --> E1
    E1 --> E2
    E2 --> E3
    E3 -->|"New State + Reward"| AGT
    CON -.->|"Hard constraints"| E1

    style STATE fill:#dbeafe,stroke:#3b82f6
    style AGT fill:#fef9c3,stroke:#eab308,stroke-width:2px
    style ACT fill:#ede9fe,stroke:#8b5cf6
    style ENV fill:#dcfce7,stroke:#22c55e
    style CON fill:#fee2e2,stroke:#ef4444
```

---

## DIAGRAM 5 — fig_3_8_fl.png
## Federated Learning System (Chapter 3.11)

```mermaid
flowchart TB
    subgraph SERVER["FedProx Aggregation Server"]
        SRV["Global Model\nFedProx Aggregation\nProximal Term mu = 0.01\n50 Communication Rounds"]
    end

    subgraph C1["Client 1"]
        CL1["Banking and Finance\n10 Stocks\nHDFCBANK · ICICIBANK\nKOTAKBANK · SBIN · AXISBANK"]
    end

    subgraph C2["Client 2"]
        CL2["IT and Telecom\n6 Stocks\nTCS · INFOSYS · WIPRO\nHCLTECH · TECHM · BHARTIARTL"]
    end

    subgraph C3["Client 3"]
        CL3["Pharma and FMCG\n8 Stocks\nSUNPHARMA · DRREDDY\nHINDUNILVR · ITC · NESTLEIND"]
    end

    subgraph C4["Client 4"]
        CL4["Energy · Auto · Metals\n28 Stocks\nRELIANCE · ONGC · MARUTI\nTATASTEEL · HINDALCO"]
    end

    DP["DP-SGD Noise\nGaussian Noise\nClip Norm = 1.0\nPrivacy epsilon = 8.0"]

    SRV -->|"Broadcast Global Weights"| CL1 & CL2 & CL3 & CL4
    CL1 & CL2 & CL3 & CL4 -->|"Local Training 5 epochs"| DP
    DP -->|"Noisy Gradient Updates"| SRV

    style SERVER fill:#fef9c3,stroke:#eab308,stroke-width:2px
    style C1 fill:#dbeafe,stroke:#3b82f6
    style C2 fill:#dcfce7,stroke:#22c55e
    style C3 fill:#fce7f3,stroke:#ec4899
    style C4 fill:#ffedd5,stroke:#f97316
    style DP fill:#fee2e2,stroke:#ef4444,stroke-width:2px
```

---

## File Names to Save As
- Diagram 1: fqn1/disst/imgs/fig_3_1_architecture.png
- Diagram 2: fqn1/disst/imgs/fig_3_4_sentiment_pipeline.png
- Diagram 3: fqn1/disst/imgs/fig_3_6_tgat.png
- Diagram 4: fqn1/disst/imgs/fig_3_7_rl_env.png
- Diagram 5: fqn1/disst/imgs/fig_3_8_fl.png
