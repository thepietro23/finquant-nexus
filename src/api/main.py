"""Phase 13: FastAPI Application.

REST API for FINQUANT-NEXUS v4.
Endpoints serve the React dashboard (Phase 14) and external consumers.

Run:  uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs (Swagger UI)
"""

import traceback

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.api.schemas import (
    HealthResponse, ConfigResponse,
    StockInfo, StockListResponse,
    SentimentRequest, SentimentResponse,
    BatchSentimentRequest, BatchSentimentResponse,
    StressTestRequest, StressTestResponse, ScenarioResult,
    QAOARequest, QAOAResponse,
    MetricsRequest, MetricsResponse,
)

logger = get_logger('api')


# ============================================================
# APP SETUP
# ============================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = get_config('api')

    app = FastAPI(
        title="FINQUANT-NEXUS v4 API",
        description="Self-Optimizing Federated Portfolio Intelligence Platform",
        version="4.0.0",
    )

    # CORS for React dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.get('cors_origins', ['http://localhost:3000']),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ============================================================
# HEALTH + CONFIG
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/api/config", response_model=ConfigResponse)
def get_app_config():
    """Return current configuration (non-sensitive)."""
    cfg = get_config()
    return ConfigResponse(
        seed=cfg.get('seed', 42),
        device=cfg.get('device', 'cpu'),
        fp16=cfg.get('fp16', False),
        data=cfg.get('data', {}),
        rl=cfg.get('rl', {}),
        quantum=cfg.get('quantum', {}),
        fl=cfg.get('fl', {}),
    )


# ============================================================
# STOCKS
# ============================================================

@app.get("/api/stocks", response_model=StockListResponse)
def list_stocks():
    """Return NIFTY 50 stock list with sectors."""
    from src.data.stocks import NIFTY50, get_all_tickers, get_sector

    tickers = get_all_tickers()
    stocks = []
    for t in tickers:
        sector = get_sector(t) or 'Unknown'
        stocks.append(StockInfo(ticker=t, sector=sector))

    return StockListResponse(count=len(stocks), stocks=stocks)


# ============================================================
# SENTIMENT
# ============================================================

@app.post("/api/sentiment", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    """Predict sentiment for a single text using FinBERT."""
    try:
        from src.sentiment.finbert import predict_sentiment as _predict
        result = _predict(req.text)

        label = 'positive' if result['score'] > 0.1 else (
            'negative' if result['score'] < -0.1 else 'neutral')

        return SentimentResponse(
            text=req.text,
            score=round(result['score'], 4),
            positive=round(result['positive'], 4),
            negative=round(result['negative'], 4),
            neutral=round(result['neutral'], 4),
            label=label,
        )
    except Exception as e:
        logger.error(f'Sentiment error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sentiment/batch", response_model=BatchSentimentResponse)
def predict_sentiment_batch(req: BatchSentimentRequest):
    """Batch sentiment prediction."""
    try:
        from src.sentiment.finbert import predict_batch

        results = predict_batch(req.texts)
        responses = []
        for text, r in zip(req.texts, results):
            label = 'positive' if r['score'] > 0.1 else (
                'negative' if r['score'] < -0.1 else 'neutral')
            responses.append(SentimentResponse(
                text=text,
                score=round(r['score'], 4),
                positive=round(r['positive'], 4),
                negative=round(r['negative'], 4),
                neutral=round(r['neutral'], 4),
                label=label,
            ))

        return BatchSentimentResponse(count=len(responses), results=responses)
    except Exception as e:
        logger.error(f'Batch sentiment error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# STRESS TESTING
# ============================================================

@app.post("/api/stress-test", response_model=StressTestResponse)
def run_stress_test(req: StressTestRequest):
    """Run portfolio stress test with multiple crash scenarios."""
    try:
        from src.gan.stress import run_all_stress_tests, stress_test_summary

        n = req.n_stocks
        np.random.seed(42)
        weights = np.ones(n) / n
        mean_ret = np.random.normal(0.0005, 0.001, n)
        cov = np.eye(n) * 0.0001 + np.ones((n, n)) * 0.00002

        results = run_all_stress_tests(
            weights, mean_ret, cov,
            n_simulations=req.n_simulations,
        )
        summary = stress_test_summary(results)

        scenarios = []
        for name, data in summary.items():
            scenarios.append(ScenarioResult(
                scenario=name,
                mean_return=str(data.get('mean_return', 'N/A')),
                var_95=str(data.get('var_95', 'N/A')),
                cvar_95=str(data.get('cvar_95', 'N/A')),
                survival_rate=str(data.get('survival_rate', 'N/A')),
            ))

        return StressTestResponse(
            n_stocks=n,
            n_simulations=req.n_simulations,
            scenarios=scenarios,
        )
    except Exception as e:
        logger.error(f'Stress test error: {traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# QUANTUM / QAOA
# ============================================================

@app.post("/api/qaoa", response_model=QAOAResponse)
def run_qaoa_optimization(req: QAOARequest):
    """Run QAOA quantum portfolio optimization."""
    try:
        from src.quantum.portfolio import quantum_portfolio_optimize

        np.random.seed(42)
        # Synthetic returns for demo (real data would come from data pipeline)
        rng = np.random.RandomState(42)
        returns = rng.randn(500, req.n_assets) * 0.01 + 0.0005

        result = quantum_portfolio_optimize(
            returns,
            n_assets=req.n_assets,
            k_select=req.k_select,
            risk_aversion=req.risk_aversion,
            qaoa_layers=req.qaoa_layers,
            shots=req.shots,
            seed=42,
        )

        return QAOAResponse(
            quantum_assets=result.quantum_assets,
            quantum_sharpe=round(result.quantum_sharpe, 4),
            quantum_weights=result.quantum_weights.round(4).tolist(),
            classical_assets=result.classical_assets,
            classical_sharpe=round(result.classical_sharpe, 4),
            classical_weights=result.classical_weights.round(4).tolist(),
            n_qubits=result.qaoa_result.n_qubits,
            best_bitstring=result.qaoa_result.best_bitstring,
            n_function_evals=result.qaoa_result.n_function_evals,
        )
    except Exception as e:
        logger.error(f'QAOA error: {traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# METRICS
# ============================================================

@app.post("/api/metrics", response_model=MetricsResponse)
def compute_metrics(req: MetricsRequest):
    """Compute financial metrics from daily returns."""
    from src.utils.metrics import (
        sharpe_ratio, sortino_ratio, annualized_return,
        annualized_volatility, max_drawdown,
    )

    returns = np.array(req.returns, dtype=np.float64)
    values = 100 * np.cumprod(1 + returns)

    return MetricsResponse(
        sharpe_ratio=round(sharpe_ratio(returns), 4),
        sortino_ratio=round(sortino_ratio(returns), 4),
        annualized_return=round(annualized_return(returns), 4),
        annualized_volatility=round(annualized_volatility(returns), 4),
        max_drawdown=round(max_drawdown(values), 4),
        n_days=len(returns),
    )
