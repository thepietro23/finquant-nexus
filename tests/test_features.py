"""Phase 2 Tests: Feature Engineering — indicators, normalization, feature matrix.

Tests:
  T2.1: All 21 indicator columns present after compute_technical_indicators
  T2.2: No NaN in final output of engineer_stock_features
  T2.3: Z-score values clipped within [-5, +5]
  T2.4: Feature tensor shape is (n_stocks, n_time, n_features)
  T2.5: Feature column list matches config
  T2.6: Rolling normalization doesn't leak future data (look-ahead test)

Edge Cases:
  E2.1: Stock with short history (< min_periods) — should still work (fewer rows)
  E2.2: Stock with zero volume days — volume_ratio handles div-by-zero
  E2.3: Constant price series — z-score handles zero std
  E2.4: Feature tensor with single stock
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

from src.data.features import (
    compute_technical_indicators,
    normalize_features,
    engineer_stock_features,
    build_feature_tensor,
    get_feature_columns,
    FEATURE_COLUMNS,
)
from src.utils.config import get_config


def _load_reliance():
    """Load RELIANCE CSV for testing."""
    path = os.path.join(DATA_DIR, 'RELIANCE_NS.csv')
    if not os.path.exists(path):
        pytest.skip('RELIANCE data not downloaded yet')
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _make_synthetic(n_days=500, seed=42):
    """Create synthetic OHLCV data for deterministic testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    close = 100 + np.cumsum(rng.randn(n_days) * 0.5)
    close = np.maximum(close, 1.0)  # no negative prices
    high = close + rng.rand(n_days) * 2
    low = close - rng.rand(n_days) * 2
    low = np.maximum(low, 0.5)
    opn = close + rng.randn(n_days) * 0.5
    volume = rng.randint(100000, 10000000, n_days).astype(float)

    return pd.DataFrame({
        'Open': opn, 'High': high, 'Low': low,
        'Close': close, 'Adj Close': close, 'Volume': volume,
    }, index=dates)


# ===========================
# Unit Tests
# ===========================

class TestTechnicalIndicators:
    """T2.1: All indicator columns present."""

    def test_all_columns_present(self):
        """All 21 feature columns exist after compute_technical_indicators."""
        df = _make_synthetic(500)
        result = compute_technical_indicators(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f'Missing column: {col}'

    def test_indicator_count(self):
        """At least 21 feature columns defined."""
        assert len(FEATURE_COLUMNS) >= 21, f'Expected 21+ features, got {len(FEATURE_COLUMNS)}'

    def test_on_real_data(self):
        """Indicators compute without error on real RELIANCE data."""
        df = _load_reliance()
        result = compute_technical_indicators(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f'Missing column: {col}'
        # Real data should have substantial non-NaN values
        non_nan = result[FEATURE_COLUMNS].notna().sum().min()
        assert non_nan > 1000, f'Too many NaN in real data indicators'


class TestNormalization:
    """T2.3: Z-score normalization works correctly."""

    def test_zscore_clipped(self):
        """Z-scores clipped within [-5, +5] range."""
        df = _make_synthetic(500)
        featured = compute_technical_indicators(df)
        normalized = normalize_features(featured, clip_range=5.0, min_periods=60)
        feature_cols = [c for c in FEATURE_COLUMNS if c in normalized.columns]
        for col in feature_cols:
            valid = normalized[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= -5.0, f'{col} has value below -5: {valid.min()}'
                assert valid.max() <= 5.0, f'{col} has value above +5: {valid.max()}'

    def test_no_lookahead(self):
        """T2.6: Rolling normalization doesn't use future data.

        Check that z-score at time t only depends on data up to time t.
        We do this by computing z-score on full data vs truncated data
        and verifying the value at truncation point is the same.
        """
        df = _make_synthetic(400)
        featured = compute_technical_indicators(df)

        # Normalize full series
        full_norm = normalize_features(featured, window=100, min_periods=60)

        # Normalize only first 300 rows
        partial = featured.iloc[:300].copy()
        partial_norm = normalize_features(partial, window=100, min_periods=60)

        # Value at row 299 should be identical
        col = 'rsi'
        full_val = full_norm[col].iloc[299]
        partial_val = partial_norm[col].iloc[299]
        if not (np.isnan(full_val) and np.isnan(partial_val)):
            assert abs(full_val - partial_val) < 1e-10, \
                f'Look-ahead bias detected in {col}: full={full_val}, partial={partial_val}'


class TestEngineerStockFeatures:
    """T2.2: Full pipeline produces clean output."""

    def test_no_nan_in_output(self):
        """No NaN values in engineer_stock_features output."""
        df = _make_synthetic(500)
        result = engineer_stock_features(df)
        feature_cols = [c for c in FEATURE_COLUMNS if c in result.columns]
        nan_count = result[feature_cols].isna().sum().sum()
        assert nan_count == 0, f'Found {nan_count} NaN values in output'

    def test_output_has_all_features(self):
        """Output has all feature columns."""
        df = _make_synthetic(500)
        result = engineer_stock_features(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f'Missing column: {col}'

    def test_rows_reduced(self):
        """NaN rows dropped — output should be shorter than input."""
        df = _make_synthetic(500)
        result = engineer_stock_features(df)
        # Rolling windows need warm-up, so some rows dropped
        assert len(result) < len(df), 'Expected some rows to be dropped'
        assert len(result) > 100, f'Too many rows dropped: {len(result)} remaining'

    def test_real_data_pipeline(self):
        """Full pipeline on real RELIANCE data."""
        df = _load_reliance()
        from src.data.quality import DataQualityChecker
        qc = DataQualityChecker()
        clean = qc.clean_stock(df)
        result = engineer_stock_features(clean)
        feature_cols = [c for c in FEATURE_COLUMNS if c in result.columns]
        nan_count = result[feature_cols].isna().sum().sum()
        assert nan_count == 0, f'Found {nan_count} NaN in real data output'
        assert len(result) > 1000, f'Too few rows after feature eng: {len(result)}'


class TestFeatureTensor:
    """T2.4: Feature tensor shape and properties."""

    def test_tensor_shape(self):
        """Tensor shape is (n_stocks, n_time, n_features)."""
        df1 = _make_synthetic(500, seed=42)
        df2 = _make_synthetic(500, seed=43)
        feat1 = engineer_stock_features(df1)
        feat2 = engineer_stock_features(df2)
        features_dict = {'STOCK_A': feat1, 'STOCK_B': feat2}
        tensor, dates, tickers = build_feature_tensor(features_dict)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 2  # 2 stocks
        assert tensor.shape[2] == len(FEATURE_COLUMNS)
        assert len(dates) == tensor.shape[1]
        assert len(tickers) == 2

    def test_tensor_no_nan(self):
        """No NaN in final tensor."""
        df1 = _make_synthetic(500, seed=42)
        df2 = _make_synthetic(500, seed=43)
        feat1 = engineer_stock_features(df1)
        feat2 = engineer_stock_features(df2)
        features_dict = {'STOCK_A': feat1, 'STOCK_B': feat2}
        tensor, _, _ = build_feature_tensor(features_dict)
        assert not np.isnan(tensor).any(), 'NaN found in tensor'

    def test_tensor_dtype(self):
        """Tensor is float32 (for FP16 training later)."""
        df1 = _make_synthetic(500, seed=42)
        feat1 = engineer_stock_features(df1)
        features_dict = {'STOCK_A': feat1}
        tensor, _, _ = build_feature_tensor(features_dict)
        assert tensor.dtype == np.float32


# ===========================
# Edge Case Tests
# ===========================

class TestEdgeCases:
    """Edge cases for robustness."""

    def test_short_history(self):
        """E2.1: Short stock history (< rolling window) still works."""
        df = _make_synthetic(300)  # 300 days — enough for 60 min_periods but < 252 window
        result = engineer_stock_features(df)
        # With 300 days, rolling z-score (min_periods=60) should leave ~240 rows
        assert len(result) > 0, 'Short history produced zero rows'
        feature_cols = [c for c in FEATURE_COLUMNS if c in result.columns]
        nan_count = result[feature_cols].isna().sum().sum()
        assert nan_count == 0, f'NaN found in short history output'

    def test_zero_volume_days(self):
        """E2.2: Zero volume doesn't crash volume_ratio."""
        df = _make_synthetic(500)
        df.loc[df.index[100:110], 'Volume'] = 0  # 10 zero-volume days
        result = engineer_stock_features(df)
        feature_cols = [c for c in FEATURE_COLUMNS if c in result.columns]
        nan_count = result[feature_cols].isna().sum().sum()
        assert nan_count == 0, f'NaN from zero volume: {nan_count}'

    def test_constant_price(self):
        """E2.3: Constant price doesn't crash z-score normalization."""
        df = _make_synthetic(500)
        df['Close'] = 100.0  # Constant price
        df['Adj Close'] = 100.0
        df['High'] = 101.0
        df['Low'] = 99.0
        result = engineer_stock_features(df)
        # Some features will be NaN (zero std -> NaN z-score -> dropped)
        # But it should not crash
        assert result is not None

    def test_single_stock_tensor(self):
        """E2.4: Feature tensor works with single stock."""
        df = _make_synthetic(500)
        feat = engineer_stock_features(df)
        features_dict = {'ONLY_ONE': feat}
        tensor, dates, tickers = build_feature_tensor(features_dict)
        assert tensor.shape[0] == 1
        assert tickers == ['ONLY_ONE']


class TestFeatureColumns:
    """T2.5: Feature list consistency."""

    def test_feature_columns_match_config(self):
        """Feature columns match what's listed in config."""
        cfg = get_config('features')
        config_indicators = cfg.get('indicators', [])
        # All config indicators should be in our FEATURE_COLUMNS
        for ind in config_indicators:
            assert ind in FEATURE_COLUMNS, f'Config indicator {ind} not in FEATURE_COLUMNS'

    def test_get_feature_columns(self):
        """get_feature_columns returns a copy, not the original."""
        cols1 = get_feature_columns()
        cols2 = get_feature_columns()
        assert cols1 == cols2
        cols1.append('dummy')
        assert 'dummy' not in get_feature_columns()
