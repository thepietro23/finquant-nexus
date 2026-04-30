# FINQUANT-NEXUS v4 — Personal PC Setup Guide

> **For:** Praveen Pal Rawal, MTech DS&ML (RRU)
> **Project:** FINQUANT-NEXUS v4 Dissertation
> **Purpose:** Complete local setup — including everything that is NOT on GitHub

---

## What GitHub Does NOT Include (Must Do Manually)

| Item | Why Gitignored | Action Needed |
|------|---------------|---------------|
| `.env` | Contains passwords/keys | Create manually (template below) |
| `venv/` | Machine-specific binaries | Re-create with pip |
| `data/finbert_local/` | 417 MB model — too large | Auto-downloads on first run |
| `data/*.csv` | Large data files | Auto-downloads via yfinance |
| `models/*.pt / *.pth` | Trained checkpoints — huge | Re-train or copy from backup |
| `experiments/` | Run artifacts | Re-run experiments |
| `logs/` | Runtime logs | Auto-created on startup |
| `wandb/` | W&B experiment cache | Re-sync or ignore |
| `pgdata/` | Docker Postgres volume | Docker re-creates automatically |
| `node_modules/` | npm cache | Run `npm install` |

---

## Prerequisites — Install These First

### 1. Python 3.11 (exact version required)
Download from: https://www.python.org/downloads/release/python-3119/

- During install: check **"Add Python to PATH"**
- Verify: `python --version` → should show `Python 3.11.x`

### 2. Node.js 18+ (for React dashboard)
Download from: https://nodejs.org/en/download (LTS version)

- Verify: `node --version` → `v18.x.x` or higher
- Verify: `npm --version` → `9.x.x` or higher

### 3. Git
Download from: https://git-scm.com/download/win

### 4. PostgreSQL 16
Download from: https://www.postgresql.org/download/windows/

- **During install set:**
  - Superuser password: anything you remember (e.g., `postgres123`)
  - Port: `5432` (default)
- After install, open **pgAdmin** or **psql** and run:

```sql
CREATE USER finquant WITH PASSWORD 'finquant123';
CREATE DATABASE finquant OWNER finquant;
GRANT ALL PRIVILEGES ON DATABASE finquant TO finquant;
```

### 5. Visual C++ Build Tools (required for some Python packages)
Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Select: **"Desktop development with C++"**

---

## Step 1 — Get the Code

### Option A: Copy from existing machine
Copy the entire `fqn1/` folder to your PC (USB/cloud), but **skip these folders** as they are too large and machine-specific:
```
fqn1/venv/           ← skip (re-create below)
fqn1/node_modules/   ← skip (re-create below)
fqn1/data/finbert_local/  ← skip (auto-downloads)
```

### Option B: Clone from GitHub
```bash
git clone https://github.com/<your-username>/finquant-nexus.git fqn1
cd fqn1
```

---

## Step 2 — Create Python Virtual Environment

Open terminal/PowerShell inside the `fqn1/` folder:

```bash
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Git Bash)
source venv/Scripts/activate

# Verify you're inside venv
python --version   # Should show Python 3.11.x
pip --version
```

> **Note:** Every time you open a new terminal, run the activate command before doing anything else.

---

## Step 3 — Install PyTorch

**IMPORTANT:** Install PyTorch BEFORE running `pip install -r requirements.txt`.

### If you have NVIDIA GPU (CUDA):
First check your CUDA version: `nvidia-smi`

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### If CPU only (no GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```python
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs all 39+ packages including:
- FastAPI, Uvicorn (backend API)
- Hugging Face Transformers (FinBERT)
- Stable-Baselines3 (RL agents: PPO, SAC, TD3, A2C, DDPG)
- PyTorch Geometric (Graph Neural Network)
- Flower (Federated Learning)
- Qiskit (Quantum Computing)
- yfinance, pandas, scikit-learn
- SQLAlchemy, psycopg2 (database)
- plotly, matplotlib, seaborn

> **If torch-geometric install fails:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```
Replace `cpu` with `cu121` or `cu118` if you have GPU.

---

## Step 5 — Create the `.env` File

This file is **NOT on GitHub** (it contains credentials). Create it manually at `fqn1/.env`:

```bash
# In fqn1/ folder, create .env file with this exact content:
```

**File content — `fqn1/.env`:**
```
DATABASE_URL=postgresql://finquant:finquant123@localhost:5432/finquant
MODEL_DIR=./models
LOG_LEVEL=INFO
SEED=42
WANDB_PROJECT=finquant-nexus
# WANDB_API_KEY=your_key_here
```

> **Optional:** If you want W&B experiment tracking, get a free API key from https://wandb.ai and replace `your_key_here`.

---

## Step 6 — Create Required Directories

These folders are gitignored but the app needs them to exist:

```bash
mkdir -p models
mkdir -p data
mkdir -p logs
mkdir -p experiments
```

On Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path models, data, logs, experiments
```

---

## Step 7 — Set Up the Database

Make sure PostgreSQL is running, then:

```bash
# With venv activated, run database setup
python -c "
from src.utils.config import load_config
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    print('Database connection: OK')
    conn.execute(text('SELECT 1'))
print('Setup complete')
"
```

If this fails, verify PostgreSQL is running and the user/database was created in Step 0.

---

## Step 8 — Download Stock Data (First Time Only)

This downloads NIFTY 50 historical data (2015–2025) via yfinance:

```bash
python -c "from src.data.download import download_all; download_all()"
```

> This creates CSV files in `data/` — takes about 2–5 minutes depending on internet.
> These CSVs are gitignored (large files), so run this once on every new machine.

---

## Step 9 — FinBERT Model (Auto-Downloads on First Use)

The FinBERT model (`data/finbert_local/`, ~417MB) is gitignored.
It **downloads automatically** the first time sentiment analysis is called.

To pre-download it manually:
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = 'ProsusAI/finbert'
AutoTokenizer.from_pretrained(model_name, cache_dir='data/finbert_local')
AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='data/finbert_local')
print('FinBERT downloaded successfully')
"
```

> Takes ~5 minutes on first run. Subsequent runs use cached version.

---

## Step 10 — Install Frontend Dependencies

```bash
cd dashboard
npm install
cd ..
```

This downloads all React/Vite/Tailwind packages into `dashboard/node_modules/` (~200MB).

---

## Step 11 — Run the Project

### Terminal 1: Start Backend (FastAPI)

```bash
# In fqn1/ folder, with venv activated
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs (Swagger UI)

### Terminal 2: Start Frontend (React Dashboard)

```bash
# In fqn1/dashboard/ folder
cd dashboard
npm run dev
```

Dashboard runs at: http://localhost:3000

### Verify Everything Works

Open http://localhost:3000 in your browser. You should see the FINQUANT-NEXUS dashboard with 6 tabs:
- Portfolio
- RL Agent (PPO/SAC/TD3/A2C/DDPG/Ensemble)
- Stress Testing
- Sentiment Analysis
- Federated Learning
- Graph Visualization

---

## Step 12 — Run Tests (Optional but Recommended)

```bash
# Run all 246 tests
python -m pytest tests/ -v --tb=short

# Run specific phase tests
python -m pytest tests/test_phase0.py -v    # Config & utils
python -m pytest tests/test_data.py -v      # Data pipeline
python -m pytest tests/test_agent.py -v     # RL agents
python -m pytest tests/test_api.py -v       # API endpoints
```

Expected: ~245 passed, 1 xfail (GPU test skipped on CPU-only machine — this is normal).

---

## Alternative: Docker Setup (Easiest Method)

If you have Docker Desktop installed, you can skip Steps 2–9 entirely:

### Install Docker Desktop
Download: https://www.docker.com/products/docker-desktop/

### Run with Docker Compose
```bash
# In fqn1/ folder
docker-compose up --build
```

This automatically:
- Creates PostgreSQL 16 database
- Installs all Python dependencies
- Starts the FastAPI backend on port 8000

> **Note:** You still need to run `npm install && npm run dev` in `dashboard/` for the frontend.

---

## Troubleshooting

### `psycopg2` install error on Windows
```bash
pip install psycopg2-binary
```

### `torch-geometric` import error
```bash
pip uninstall torch-geometric torch-scatter torch-sparse -y
pip install torch-geometric
```

### `yfinance` SSL error / rate limiting
```bash
pip install --upgrade yfinance curl_cffi
```

### `uvicorn` command not found
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Dashboard shows "Network Error" / API not connecting
- Make sure backend (port 8000) is running first
- Check that vite.config.ts has proxy to `http://localhost:8000`
- Disable Windows Firewall for local connections

### `ModuleNotFoundError: No module named 'src'`
- Make sure you are running commands from inside the `fqn1/` directory
- Make sure venv is activated

### Port 5432 already in use
```powershell
# Find and stop conflicting process
netstat -ano | findstr :5432
taskkill /PID <pid_number> /F
```

---

## Quick Reference — Daily Workflow

```bash
# 1. Open terminal in fqn1/

# 2. Activate venv
source venv/Scripts/activate   # Git Bash
# OR
.\venv\Scripts\Activate.ps1    # PowerShell

# 3. Start backend (Terminal 1)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start frontend (Terminal 2)
cd dashboard && npm run dev
```

---

## Project Structure Summary

```
fqn1/
├── .env                    ← CREATE THIS (not on GitHub)
├── configs/base.yaml       ← All hyperparameters (on GitHub)
├── requirements.txt        ← Python deps (on GitHub)
├── src/
│   ├── api/               ← FastAPI backend
│   ├── data/              ← NIFTY 50 data pipeline
│   ├── sentiment/         ← FinBERT sentiment
│   ├── graph/             ← GNN stock correlations
│   ├── models/            ← T-GAT model
│   ├── rl/                ← PPO/SAC/TD3/A2C/DDPG/Ensemble
│   ├── gan/               ← TimeGAN + stress testing
│   ├── nas/               ← DARTS neural architecture search
│   ├── federated/         ← Flower federated learning
│   ├── quantum/           ← Qiskit QAOA
│   └── utils/             ← Config, logger, metrics
├── dashboard/             ← React + Vite + Tailwind frontend
├── tests/                 ← 246 pytest tests
├── data/                  ← CSVs + FinBERT cache (NOT on GitHub)
├── models/                ← Trained .pt checkpoints (NOT on GitHub)
├── logs/                  ← Runtime logs (NOT on GitHub)
└── experiments/           ← W&B / experiment outputs (NOT on GitHub)
```

---

*Last updated: April 2026 | FINQUANT-NEXUS v4 | MTech DS&ML Dissertation — RRU*
