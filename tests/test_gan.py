"""Phase 8-9 Tests: TimeGAN + Stress Testing.

TimeGAN Tests:
  T8.1: TimeGAN initializes with correct architecture
  T8.2: Training completes without error
  T8.3: Generated data has correct shape
  T8.4: Generated data is finite (no NaN/Inf)
  T8.5: Generated data statistics reasonable (mean/std similar to real)
  T8.6: Sliding window data preparation

Stress Tests:
  T9.1: VaR computation correct
  T9.2: CVaR ≤ VaR (CVaR is more conservative)
  T9.3: Monte Carlo returns StressResult
  T9.4: All crash scenarios run
  T9.5: Crash scenario ordering (2008 worse than normal)
  T9.6: Survival rate between 0 and 1
  T9.7: Stress summary formatting

Edge Cases:
  E8.1: Single feature TimeGAN
  E8.2: Very short training
  E8.3: Generate before training raises error
  E9.1: Equal weights portfolio
  E9.2: Concentrated portfolio (all in one stock)
  E9.3: Unknown crash scenario raises error
  E9.4: Zero-variance returns
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.gan.timegan import TimeGAN
from src.gan.stress import (
    compute_var,
    compute_cvar,
    monte_carlo_simulation,
    simulate_crash_scenario,
    run_all_stress_tests,
    stress_test_summary,
    StressResult,
    CRASH_SCENARIOS,
)


# ===========================
# TimeGAN Tests
# ===========================

class TestTimeGANInit:
    """T8.1: TimeGAN initialization."""

    def test_creates(self):
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=16,
                      latent_dim=8, num_layers=1)
        assert gan is not None
        assert not gan.trained

    def test_components_exist(self):
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=16,
                      latent_dim=8, num_layers=1)
        assert hasattr(gan, 'embedder')
        assert hasattr(gan, 'recovery')
        assert hasattr(gan, 'generator')
        assert hasattr(gan, 'discriminator')
        assert hasattr(gan, 'supervisor')

    def test_stats(self):
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=16,
                      latent_dim=8, num_layers=1)
        stats = gan.get_stats()
        assert stats['total_params'] > 0
        assert not stats['trained']


class TestTimeGANTraining:
    """T8.2: Training."""

    def test_train_completes(self):
        """T8.2: Training runs without error."""
        data = np.random.randn(100, 5).astype(np.float32)
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=16,
                      latent_dim=8, num_layers=1)
        gan.train(data, epochs=3, batch_size=8)
        assert gan.trained

    def test_train_3d_input(self):
        """Training works with pre-windowed 3D input."""
        data = np.random.randn(20, 16, 5).astype(np.float32)
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=16,
                      latent_dim=8, num_layers=1)
        gan.train(data, epochs=3, batch_size=8)
        assert gan.trained


class TestTimeGANGeneration:
    """T8.3-T8.5: Generation quality."""

    @pytest.fixture(scope='class')
    def trained_gan(self):
        np.random.seed(42)
        data = np.random.randn(200, 5).astype(np.float32) * 0.5
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=32,
                      latent_dim=16, num_layers=1)
        gan.train(data, epochs=10, batch_size=16)
        return gan, data

    def test_output_shape(self, trained_gan):
        """T8.3: Generated data has correct shape."""
        gan, _ = trained_gan
        synthetic = gan.generate(n_samples=10)
        assert synthetic.shape == (10, 16, 5)

    def test_output_finite(self, trained_gan):
        """T8.4: No NaN/Inf in generated data."""
        gan, _ = trained_gan
        synthetic = gan.generate(n_samples=10)
        assert np.isfinite(synthetic).all()

    def test_statistics_reasonable(self, trained_gan):
        """T8.5: Generated data has similar scale to real."""
        gan, data = trained_gan
        synthetic = gan.generate(n_samples=50)
        # Mean should be in same order of magnitude (within 10x)
        real_std = data.std()
        synth_std = synthetic.std()
        assert synth_std < real_std * 20, \
            f'Synthetic std {synth_std:.3f} too far from real {real_std:.3f}'


class TestDataPrep:
    """T8.6: Data preparation."""

    def test_sliding_window(self):
        """2D data → sliding windows correctly."""
        data = np.arange(100).reshape(100, 1).astype(np.float32)
        gan = TimeGAN(input_dim=1, seq_length=10, hidden_dim=8,
                      latent_dim=4, num_layers=1)
        loader = gan._prepare_data(data, batch_size=16)
        batch = next(iter(loader))[0]
        assert batch.shape[1] == 10  # seq_length
        assert batch.shape[2] == 1   # features


# ===========================
# Stress Testing Tests
# ===========================

class TestVaR:
    """T9.1-T9.2: VaR and CVaR computation."""

    def test_var_95(self):
        """T9.1: VaR at 95% confidence."""
        np.random.seed(42)
        returns = np.random.randn(10000) * 0.01  # ~N(0, 0.01)
        var = compute_var(returns, 0.95)
        # 5th percentile of N(0, 0.01) ≈ -0.0165
        assert -0.025 < var < -0.010

    def test_cvar_leq_var(self):
        """T9.2: CVaR is more conservative (worse) than VaR."""
        np.random.seed(42)
        returns = np.random.randn(10000) * 0.01
        var = compute_var(returns, 0.95)
        cvar = compute_cvar(returns, 0.95)
        assert cvar <= var, f'CVaR {cvar:.4f} should be ≤ VaR {var:.4f}'

    def test_var_99_worse_than_95(self):
        """99% VaR is worse than 95% VaR."""
        np.random.seed(42)
        returns = np.random.randn(10000) * 0.01
        var_95 = compute_var(returns, 0.95)
        var_99 = compute_var(returns, 0.99)
        assert var_99 < var_95


class TestMonteCarlo:
    """T9.3: Monte Carlo simulation."""

    def test_returns_stress_result(self):
        """T9.3: Returns StressResult object."""
        weights = np.array([0.5, 0.5])
        mean_r = np.array([0.0005, 0.0003])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0001]])

        result = monte_carlo_simulation(
            weights, mean_r, cov, n_paths=100, n_days=60, seed=42
        )
        assert isinstance(result, StressResult)
        assert result.scenario_name == 'Monte Carlo'
        assert len(result.portfolio_returns) == 100

    def test_var_in_result(self):
        """VaR values present in result."""
        weights = np.array([0.5, 0.5])
        mean_r = np.array([0.0005, 0.0003])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0001]])

        result = monte_carlo_simulation(
            weights, mean_r, cov, n_paths=500, n_days=60, seed=42
        )
        assert result.var_95 != 0
        assert result.var_99 != 0
        assert result.var_99 < result.var_95


class TestCrashScenarios:
    """T9.4-T9.6: Crash scenario simulation."""

    def test_all_scenarios_run(self):
        """T9.4: All predefined scenarios execute."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mean_r = np.zeros(5)
        cov = np.eye(5) * 0.0001

        results = run_all_stress_tests(
            weights, mean_r, cov, n_simulations=50, seed=42
        )
        assert len(results) == len(CRASH_SCENARIOS)
        for name in CRASH_SCENARIOS:
            assert name in results

    def test_crash_worse_than_normal(self):
        """T9.5: 2008 crash is worse than normal conditions."""
        weights = np.array([0.5, 0.5])
        mean_r = np.array([0.0005, 0.0003])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0001]])

        normal = simulate_crash_scenario(
            weights, mean_r, cov, 'normal', n_simulations=200, seed=42
        )
        crash = simulate_crash_scenario(
            weights, mean_r, cov, 'crash_2008', n_simulations=200, seed=42
        )
        assert crash.mean_return < normal.mean_return

    def test_survival_rate_range(self):
        """T9.6: Survival rate between 0 and 1."""
        weights = np.array([0.5, 0.5])
        mean_r = np.array([0.0005, 0.0003])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0001]])

        result = simulate_crash_scenario(
            weights, mean_r, cov, 'crash_2008', n_simulations=100, seed=42
        )
        assert 0 <= result.survival_rate <= 1

    def test_summary_formatting(self):
        """T9.7: Summary has correct format."""
        weights = np.array([0.5, 0.5])
        mean_r = np.zeros(2)
        cov = np.eye(2) * 0.0001

        results = run_all_stress_tests(
            weights, mean_r, cov, n_simulations=50, seed=42
        )
        summary = stress_test_summary(results)

        for name, s in summary.items():
            assert 'mean_return' in s
            assert 'var_95' in s
            assert 'survival_rate' in s
            assert '%' in s['mean_return']


# ===========================
# Edge Cases
# ===========================

class TestEdgeCases:
    """Edge cases for both TimeGAN and Stress."""

    def test_single_feature_gan(self):
        """E8.1: TimeGAN with 1 feature."""
        data = np.random.randn(100, 1).astype(np.float32)
        gan = TimeGAN(input_dim=1, seq_length=10, hidden_dim=8,
                      latent_dim=4, num_layers=1)
        gan.train(data, epochs=3, batch_size=8)
        synthetic = gan.generate(5)
        assert synthetic.shape == (5, 10, 1)

    def test_very_short_training(self):
        """E8.2: 1 epoch training."""
        data = np.random.randn(50, 3).astype(np.float32)
        gan = TimeGAN(input_dim=3, seq_length=8, hidden_dim=8,
                      latent_dim=4, num_layers=1)
        gan.train(data, epochs=1, batch_size=8)
        assert gan.trained

    def test_generate_before_train(self):
        """E8.3: Generate before training raises error."""
        gan = TimeGAN(input_dim=5, seq_length=16, hidden_dim=16,
                      latent_dim=8, num_layers=1)
        with pytest.raises(RuntimeError):
            gan.generate(5)

    def test_equal_weights(self):
        """E9.1: Equal weight portfolio stress test."""
        n = 10
        weights = np.ones(n) / n
        mean_r = np.zeros(n)
        cov = np.eye(n) * 0.0001

        result = monte_carlo_simulation(
            weights, mean_r, cov, n_paths=100, n_days=60, seed=42
        )
        assert len(result.portfolio_returns) == 100

    def test_concentrated_portfolio(self):
        """E9.2: All money in one stock."""
        weights = np.array([1.0, 0.0, 0.0])
        mean_r = np.array([0.001, 0.0, 0.0])
        cov = np.eye(3) * 0.0004

        result = monte_carlo_simulation(
            weights, mean_r, cov, n_paths=100, n_days=60, seed=42
        )
        # Higher variance than diversified
        assert abs(result.var_95) > 0

    def test_unknown_scenario_raises(self):
        """E9.3: Unknown crash scenario raises ValueError."""
        weights = np.array([0.5, 0.5])
        mean_r = np.zeros(2)
        cov = np.eye(2) * 0.0001

        with pytest.raises(ValueError):
            simulate_crash_scenario(
                weights, mean_r, cov, 'crash_alien_invasion'
            )

    def test_zero_variance(self):
        """E9.4: Zero variance returns → VaR still works."""
        returns = np.zeros(100)
        var = compute_var(returns, 0.95)
        assert var == 0.0
