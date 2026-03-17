"""Phase 13: Pydantic schemas for API request/response models.

Every endpoint has typed request and response schemas.
React dashboard (Phase 14) will consume these.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# HEALTH / CONFIG
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "4.0.0"
    project: str = "FINQUANT-NEXUS v4"
    phases_complete: int = 13


class ConfigResponse(BaseModel):
    seed: int
    device: str
    fp16: bool
    data: dict
    rl: dict
    quantum: dict
    fl: dict


# ============================================================
# STOCKS
# ============================================================

class StockInfo(BaseModel):
    ticker: str
    sector: str


class StockListResponse(BaseModel):
    count: int
    stocks: list[StockInfo]


# ============================================================
# FEATURES
# ============================================================

class FeatureResponse(BaseModel):
    ticker: str
    n_days: int
    n_features: int
    feature_names: list[str]
    latest: dict  # last day's features


# ============================================================
# SENTIMENT
# ============================================================

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class SentimentResponse(BaseModel):
    text: str
    score: float = Field(..., ge=-1.0, le=1.0)
    positive: float
    negative: float
    neutral: float
    label: str


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class BatchSentimentResponse(BaseModel):
    count: int
    results: list[SentimentResponse]


# ============================================================
# STRESS TEST
# ============================================================

class StressTestRequest(BaseModel):
    n_stocks: int = Field(default=10, ge=2, le=47)
    n_simulations: int = Field(default=1000, ge=100, le=50000)
    scenarios: list[str] = Field(
        default=["normal", "crash_2008", "crash_covid", "flash_crash"]
    )


class ScenarioResult(BaseModel):
    scenario: str
    mean_return: str
    var_95: str
    cvar_95: str
    survival_rate: str


class StressTestResponse(BaseModel):
    n_stocks: int
    n_simulations: int
    scenarios: list[ScenarioResult]


# ============================================================
# QUANTUM / QAOA
# ============================================================

class QAOARequest(BaseModel):
    n_assets: int = Field(default=6, ge=2, le=12)
    k_select: int = Field(default=3, ge=1, le=12)
    qaoa_layers: int = Field(default=2, ge=1, le=5)
    shots: int = Field(default=512, ge=64, le=4096)
    risk_aversion: float = Field(default=0.5, ge=0.0, le=5.0)


class QAOAResponse(BaseModel):
    quantum_assets: list[int]
    quantum_sharpe: float
    quantum_weights: list[float]
    classical_assets: list[int]
    classical_sharpe: float
    classical_weights: list[float]
    n_qubits: int
    best_bitstring: str
    n_function_evals: int


# ============================================================
# METRICS
# ============================================================

class MetricsRequest(BaseModel):
    returns: list[float] = Field(..., min_length=10)


class MetricsResponse(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    n_days: int
