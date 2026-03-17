"""Phase 9: Stress Testing — VaR, Monte Carlo, Scenario Analysis.

Evaluates portfolio robustness under extreme market conditions:
  1. Value at Risk (VaR): How much could we lose at 95%/99% confidence?
  2. Monte Carlo simulation: Random future paths
  3. Historical scenarios: What if 2008/COVID crash happened again?
  4. Custom crash scenarios: Flash crash, sector rotation

All methods return standardized StressResult objects for comparison.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('stress')


@dataclass
class StressResult:
    """Standardized stress test result."""
    scenario_name: str
    portfolio_returns: np.ndarray  # Array of scenario returns
    var_95: float = 0.0           # 5% VaR
    var_99: float = 0.0           # 1% VaR
    cvar_95: float = 0.0          # Conditional VaR (Expected Shortfall)
    max_loss: float = 0.0         # Worst case
    mean_return: float = 0.0
    survival_rate: float = 1.0    # % of scenarios not hitting max drawdown
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

def compute_var(returns, confidence=0.95):
    """Compute Value at Risk (Historical method).

    VaR = the loss threshold that is exceeded only (1 - confidence)% of the time.

    Args:
        returns: array of portfolio returns
        confidence: confidence level (0.95 or 0.99)

    Returns:
        VaR value (negative number = loss)
    """
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns, confidence=0.95):
    """Compute Conditional VaR (Expected Shortfall).

    CVaR = average loss in the worst (1 - confidence)% of cases.
    More conservative than VaR — captures tail risk.

    Args:
        returns: array of portfolio returns
        confidence: confidence level

    Returns:
        CVaR value (negative number)
    """
    var = compute_var(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def monte_carlo_simulation(weights, mean_returns, cov_matrix,
                           n_paths=10000, n_days=252, seed=None):
    """Monte Carlo portfolio simulation.

    Generates random future return paths based on historical statistics.

    Args:
        weights: portfolio weights (n_stocks,)
        mean_returns: daily mean returns per stock (n_stocks,)
        cov_matrix: covariance matrix (n_stocks, n_stocks)
        n_paths: number of simulation paths
        n_days: days to simulate per path
        seed: random seed

    Returns:
        StressResult with simulated portfolio returns
    """
    if seed is not None:
        np.random.seed(seed)

    n_stocks = len(weights)

    # Generate correlated random returns
    # Cholesky decomposition for correlated samples
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Not positive definite — add small regularization
        cov_reg = cov_matrix + np.eye(n_stocks) * 1e-6
        L = np.linalg.cholesky(cov_reg)

    # Simulate paths
    portfolio_total_returns = []
    for _ in range(n_paths):
        # Random daily returns for n_days
        z = np.random.randn(n_days, n_stocks)
        daily_returns = mean_returns + z @ L.T

        # Portfolio daily returns
        port_daily = daily_returns @ weights
        # Cumulative return over n_days
        total_return = np.prod(1 + port_daily) - 1
        portfolio_total_returns.append(total_return)

    returns_arr = np.array(portfolio_total_returns)

    result = StressResult(
        scenario_name='Monte Carlo',
        portfolio_returns=returns_arr,
        var_95=compute_var(returns_arr, 0.95),
        var_99=compute_var(returns_arr, 0.99),
        cvar_95=compute_cvar(returns_arr, 0.95),
        max_loss=float(returns_arr.min()),
        mean_return=float(returns_arr.mean()),
        metadata={'n_paths': n_paths, 'n_days': n_days},
    )

    logger.info(f'Monte Carlo: {n_paths} paths, {n_days} days, '
                f'VaR95={result.var_95:.2%}, VaR99={result.var_99:.2%}')
    return result


# ---------------------------------------------------------------------------
# Historical crash scenarios
# ---------------------------------------------------------------------------

# Pre-defined crash parameters (daily return shocks)
CRASH_SCENARIOS = {
    'normal': {
        'description': 'Normal market conditions',
        'daily_shock_mean': 0.0,
        'daily_shock_std': 0.01,
        'duration_days': 252,
        'correlation_boost': 0.0,
    },
    'crash_2008': {
        'description': '2008 Global Financial Crisis',
        'daily_shock_mean': -0.003,
        'daily_shock_std': 0.035,
        'duration_days': 120,
        'correlation_boost': 0.3,  # Correlations increase in crisis
    },
    'crash_covid': {
        'description': 'COVID-19 March 2020 crash',
        'daily_shock_mean': -0.005,
        'daily_shock_std': 0.05,
        'duration_days': 30,
        'correlation_boost': 0.4,
    },
    'flash_crash': {
        'description': 'Flash crash (rapid intraday)',
        'daily_shock_mean': -0.02,
        'daily_shock_std': 0.08,
        'duration_days': 5,
        'correlation_boost': 0.5,
    },
}


def simulate_crash_scenario(weights, mean_returns, cov_matrix,
                            scenario_name='crash_2008',
                            n_simulations=1000, seed=None):
    """Simulate portfolio under a specific crash scenario.

    Args:
        weights: portfolio weights
        mean_returns: base daily mean returns
        cov_matrix: base covariance matrix
        scenario_name: key from CRASH_SCENARIOS
        n_simulations: number of simulations
        seed: random seed

    Returns:
        StressResult
    """
    if seed is not None:
        np.random.seed(seed)

    scenario = CRASH_SCENARIOS.get(scenario_name)
    if scenario is None:
        raise ValueError(f'Unknown scenario: {scenario_name}. '
                         f'Available: {list(CRASH_SCENARIOS.keys())}')

    n_stocks = len(weights)
    duration = scenario['duration_days']
    shock_mean = scenario['daily_shock_mean']
    shock_std = scenario['daily_shock_std']
    corr_boost = scenario['correlation_boost']

    # Modify covariance for crisis (higher correlations)
    stressed_cov = cov_matrix.copy()
    if corr_boost > 0:
        # Increase off-diagonal correlations
        stds = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(stds, stds)
        corr = np.clip(corr + corr_boost, -1, 1)
        np.fill_diagonal(corr, 1.0)
        stressed_cov = corr * np.outer(stds * (1 + shock_std * 10), stds * (1 + shock_std * 10))

    # Ensure positive definite
    try:
        L = np.linalg.cholesky(stressed_cov)
    except np.linalg.LinAlgError:
        stressed_cov += np.eye(n_stocks) * 1e-5
        L = np.linalg.cholesky(stressed_cov)

    # Stressed mean returns
    stressed_mean = mean_returns + shock_mean

    portfolio_returns = []
    max_dd_threshold = get_config('rl').get('max_drawdown', -0.15)
    survived = 0

    for _ in range(n_simulations):
        z = np.random.randn(duration, n_stocks)
        daily_returns = stressed_mean + z @ L.T
        port_daily = daily_returns @ weights

        # Track drawdown
        cumulative = np.cumprod(1 + port_daily)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_dd = drawdowns.min()

        total_return = cumulative[-1] - 1
        portfolio_returns.append(total_return)

        if max_dd > max_dd_threshold:
            survived += 1

    returns_arr = np.array(portfolio_returns)

    result = StressResult(
        scenario_name=scenario_name,
        portfolio_returns=returns_arr,
        var_95=compute_var(returns_arr, 0.95),
        var_99=compute_var(returns_arr, 0.99),
        cvar_95=compute_cvar(returns_arr, 0.95),
        max_loss=float(returns_arr.min()),
        mean_return=float(returns_arr.mean()),
        survival_rate=survived / n_simulations,
        metadata={
            'description': scenario['description'],
            'n_simulations': n_simulations,
            'duration_days': duration,
        },
    )

    logger.info(f'{scenario_name}: VaR95={result.var_95:.2%}, '
                f'survival={result.survival_rate:.0%}')
    return result


def run_all_stress_tests(weights, mean_returns, cov_matrix,
                         n_simulations=1000, seed=42):
    """Run all predefined stress scenarios.

    Args:
        weights: portfolio weights
        mean_returns: daily mean returns
        cov_matrix: covariance matrix
        n_simulations: simulations per scenario
        seed: random seed

    Returns:
        dict of {scenario_name: StressResult}
    """
    results = {}
    for name in CRASH_SCENARIOS:
        results[name] = simulate_crash_scenario(
            weights, mean_returns, cov_matrix,
            scenario_name=name,
            n_simulations=n_simulations,
            seed=seed,
        )
    return results


def stress_test_summary(results):
    """Create summary table from stress test results.

    Args:
        results: dict of {name: StressResult}

    Returns:
        dict with formatted summary
    """
    summary = {}
    for name, r in results.items():
        summary[name] = {
            'description': r.metadata.get('description', name),
            'mean_return': f'{r.mean_return:.2%}',
            'var_95': f'{r.var_95:.2%}',
            'var_99': f'{r.var_99:.2%}',
            'cvar_95': f'{r.cvar_95:.2%}',
            'max_loss': f'{r.max_loss:.2%}',
            'survival_rate': f'{r.survival_rate:.0%}',
        }
    return summary
