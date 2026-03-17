"""Phase 13: FastAPI REST API Tests.

10 unit tests + 5 edge cases = 15 total.

Unit tests:
  T13.1:  Health endpoint returns 200 + correct structure
  T13.2:  Config endpoint returns seed, device, fp16
  T13.3:  Stock list returns NIFTY 50 tickers with sectors
  T13.4:  Sentiment endpoint — positive text scores > 0
  T13.5:  Sentiment endpoint — negative text scores < 0
  T13.6:  Batch sentiment — processes multiple texts
  T13.7:  Stress test endpoint — returns scenario results
  T13.8:  QAOA endpoint — returns quantum + classical results
  T13.9:  Metrics endpoint — returns valid financial metrics
  T13.10: CORS headers present in response

Edge cases:
  E13.1:  Sentiment with empty text → 422 validation error
  E13.2:  Sentiment with very long text → 422 validation error
  E13.3:  Batch sentiment with empty list → 422 validation error
  E13.4:  Stress test with n_stocks=1 → 422 (min is 2)
  E13.5:  Metrics with too few returns → 422 (min 10)
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.main import app

client = TestClient(app)


# ============================================================
# UNIT TESTS
# ============================================================

class TestHealthConfig:
    """T13.1–T13.2: Health and config endpoints."""

    def test_health_check(self):
        """T13.1: Health endpoint returns 200 with correct fields."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "4.0.0"
        assert "FINQUANT" in data["project"]

    def test_config_endpoint(self):
        """T13.2: Config returns expected keys."""
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "seed" in data
        assert "device" in data
        assert "fp16" in data
        assert isinstance(data["seed"], int)


class TestStocks:
    """T13.3: Stock list endpoint."""

    def test_stock_list(self):
        """T13.3: Returns NIFTY 50 tickers with sectors."""
        resp = client.get("/api/stocks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["stocks"]) == data["count"]
        # Each stock has ticker + sector
        stock = data["stocks"][0]
        assert "ticker" in stock
        assert "sector" in stock


class TestSentiment:
    """T13.4–T13.6: Sentiment analysis endpoints."""

    def test_positive_sentiment(self):
        """T13.4: Positive text returns score > 0."""
        resp = client.post("/api/sentiment", json={
            "text": "Company profits surged 50% with record revenue growth"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] > 0
        assert data["label"] == "positive"
        assert "positive" in data
        assert "negative" in data
        assert "neutral" in data

    def test_negative_sentiment(self):
        """T13.5: Negative text returns score < 0."""
        resp = client.post("/api/sentiment", json={
            "text": "Company reports massive losses and bankruptcy risk"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] < 0
        assert data["label"] == "negative"

    def test_batch_sentiment(self):
        """T13.6: Batch sentiment processes multiple texts."""
        texts = [
            "Strong quarterly earnings beat expectations",
            "Stock crashed after fraud allegations",
            "Market remained flat today",
        ]
        resp = client.post("/api/sentiment/batch", json={"texts": texts})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3
        for r in data["results"]:
            assert "score" in r
            assert "label" in r


class TestStressTest:
    """T13.7: Stress test endpoint."""

    def test_stress_test(self):
        """T13.7: Returns scenario results with expected structure."""
        resp = client.post("/api/stress-test", json={
            "n_stocks": 5,
            "n_simulations": 200,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_stocks"] == 5
        assert data["n_simulations"] == 200
        assert len(data["scenarios"]) > 0
        scenario = data["scenarios"][0]
        assert "scenario" in scenario
        assert "mean_return" in scenario
        assert "var_95" in scenario


class TestQAOA:
    """T13.8: QAOA quantum optimization endpoint."""

    def test_qaoa_optimization(self):
        """T13.8: Returns quantum + classical portfolio results."""
        resp = client.post("/api/qaoa", json={
            "n_assets": 4,
            "k_select": 2,
            "qaoa_layers": 1,
            "shots": 64,
            "risk_aversion": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["quantum_assets"]) > 0
        assert isinstance(data["quantum_sharpe"], float)
        assert len(data["quantum_weights"]) > 0
        assert len(data["classical_assets"]) > 0
        assert isinstance(data["classical_sharpe"], float)
        assert data["n_qubits"] >= 4
        assert len(data["best_bitstring"]) == 4


class TestMetrics:
    """T13.9: Financial metrics endpoint."""

    def test_compute_metrics(self):
        """T13.9: Returns valid Sharpe, Sortino, etc."""
        import numpy as np
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100).tolist()

        resp = client.post("/api/metrics", json={"returns": returns})
        assert resp.status_code == 200
        data = resp.json()
        assert "sharpe_ratio" in data
        assert "sortino_ratio" in data
        assert "annualized_return" in data
        assert "annualized_volatility" in data
        assert "max_drawdown" in data
        assert data["n_days"] == 100


class TestCORS:
    """T13.10: CORS middleware configured."""

    def test_cors_headers(self):
        """T13.10: OPTIONS request returns CORS headers."""
        resp = client.options("/api/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        # FastAPI CORS middleware should respond
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    """E13.1–E13.5: Validation and error handling."""

    def test_sentiment_empty_text(self):
        """E13.1: Empty text → 422."""
        resp = client.post("/api/sentiment", json={"text": ""})
        assert resp.status_code == 422

    def test_sentiment_too_long(self):
        """E13.2: Text > 1000 chars → 422."""
        resp = client.post("/api/sentiment", json={"text": "x" * 1001})
        assert resp.status_code == 422

    def test_batch_empty_list(self):
        """E13.3: Empty text list → 422."""
        resp = client.post("/api/sentiment/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_stress_test_invalid_n_stocks(self):
        """E13.4: n_stocks=1 → 422 (minimum is 2)."""
        resp = client.post("/api/stress-test", json={"n_stocks": 1})
        assert resp.status_code == 422

    def test_metrics_too_few_returns(self):
        """E13.5: Fewer than 10 returns → 422."""
        resp = client.post("/api/metrics", json={"returns": [0.01, 0.02]})
        assert resp.status_code == 422
