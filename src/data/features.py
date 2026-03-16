"""Phase 2: Feature Engineering — technical indicators, normalization, feature matrix.

Generates 20+ features per stock from raw OHLCV data:
- Technical indicators (RSI, MACD, Bollinger, SMA/EMA, ATR, Stochastic, Volume)
- Return features (1d, 5d, 20d)
- Volatility features (20d, 60d)
- Z-score normalization with rolling window
- NaN rows dropped to prevent downstream issues
"""

import os
import pickle

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('features')


# ---------------------------------------------------------------------------
# Technical indicator calculations (pure pandas/numpy — no external TA lib
# dependency issues). We use the `ta` library where available, fallback to
# manual if needed.
# ---------------------------------------------------------------------------

def _rsi(close, period=14):
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(close, period=20, std_dev=2):
    """Bollinger Bands — upper, mid, lower."""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def _atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    denom = highest_high - lowest_low
    stoch_k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    stoch_d = stoch_k.rolling(d_period, min_periods=d_period).mean()
    return stoch_k, stoch_d


def compute_technical_indicators(df):
    """Compute all technical indicators for a single stock DataFrame.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Adj Close, Volume]
            and DatetimeIndex.

    Returns:
        DataFrame with original columns + all indicator columns.
    """
    out = df.copy()
    close = out['Close'].astype(float)
    high = out['High'].astype(float)
    low = out['Low'].astype(float)
    volume = out['Volume'].astype(float)

    # --- Trend Indicators ---
    out['rsi'] = _rsi(close)
    macd_line, macd_sig, macd_hist = _macd(close)
    out['macd'] = macd_line
    out['macd_signal'] = macd_sig
    out['macd_hist'] = macd_hist

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
    out['bb_upper'] = bb_upper
    out['bb_mid'] = bb_mid
    out['bb_lower'] = bb_lower

    # --- Moving Averages ---
    out['sma_20'] = close.rolling(20, min_periods=20).mean()
    out['sma_50'] = close.rolling(50, min_periods=50).mean()
    out['ema_12'] = close.ewm(span=12, min_periods=12).mean()
    out['ema_26'] = close.ewm(span=26, min_periods=26).mean()

    # --- Volatility ---
    out['atr'] = _atr(high, low, close)

    # --- Stochastic ---
    stoch_k, stoch_d = _stochastic(high, low, close)
    out['stoch_k'] = stoch_k
    out['stoch_d'] = stoch_d

    # --- Volume ---
    out['volume_sma'] = volume.rolling(20, min_periods=20).mean()
    vol_sma = out['volume_sma'].replace(0, np.nan)
    out['volume_ratio'] = volume / vol_sma

    # --- Return Features ---
    out['return_1d'] = close.pct_change(1)
    out['return_5d'] = close.pct_change(5)
    out['return_20d'] = close.pct_change(20)

    # --- Volatility Features ---
    daily_ret = close.pct_change()
    out['volatility_20d'] = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(248)
    out['volatility_60d'] = daily_ret.rolling(60, min_periods=60).std() * np.sqrt(248)

    return out


# ---------------------------------------------------------------------------
# Feature column list (order matters — this is the feature vector)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_mid', 'bb_lower',
    'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'atr', 'stoch_k', 'stoch_d',
    'volume_sma', 'volume_ratio',
    'return_1d', 'return_5d', 'return_20d',
    'volatility_20d', 'volatility_60d',
]


def get_feature_columns():
    """Return ordered list of feature column names."""
    return FEATURE_COLUMNS.copy()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(df, feature_cols=None, method='zscore',
                       window=252, clip_range=5.0, min_periods=60):
    """Normalize feature columns using rolling z-score.

    Args:
        df: DataFrame with feature columns.
        feature_cols: List of columns to normalize. Defaults to FEATURE_COLUMNS.
        method: 'zscore' (only method supported currently).
        window: Rolling window size for mean/std calculation.
        clip_range: Clip z-scores to [-clip_range, +clip_range].
        min_periods: Minimum periods for rolling stats.

    Returns:
        DataFrame with normalized feature columns (original columns replaced).
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    out = df.copy()

    if method == 'zscore':
        for col in feature_cols:
            series = out[col].astype(float)
            roll_mean = series.rolling(window, min_periods=min_periods).mean()
            roll_std = series.rolling(window, min_periods=min_periods).std()
            # Avoid division by zero
            roll_std = roll_std.replace(0, np.nan)
            z = (series - roll_mean) / roll_std
            # Clip extreme values
            z = z.clip(-clip_range, clip_range)
            out[col] = z
    else:
        raise ValueError(f'Unknown normalization method: {method}')

    return out


# ---------------------------------------------------------------------------
# Full pipeline: per-stock feature generation
# ---------------------------------------------------------------------------

def engineer_stock_features(df, normalize=True):
    """Full feature engineering pipeline for a single stock.

    1. Compute technical indicators
    2. Normalize features (rolling z-score)
    3. Drop NaN rows (from rolling windows)

    Args:
        df: Raw OHLCV DataFrame with DatetimeIndex.
        normalize: Whether to apply z-score normalization.

    Returns:
        Clean DataFrame with all features, no NaN values.
    """
    cfg_feat = get_config('features')
    cfg_data = get_config('data')

    # Step 1: Technical indicators
    featured = compute_technical_indicators(df)

    # Step 2: Normalize
    if normalize:
        featured = normalize_features(
            featured,
            method=cfg_feat.get('normalize', 'zscore'),
            window=cfg_feat.get('rolling_window', 252),
            clip_range=cfg_feat.get('clip_range', 5.0),
            min_periods=cfg_feat.get('min_periods', 60),
        )

    # Step 3: Drop NaN rows (from rolling window warm-up period)
    feature_cols = [c for c in FEATURE_COLUMNS if c in featured.columns]
    before_len = len(featured)
    featured = featured.dropna(subset=feature_cols)
    dropped = before_len - len(featured)
    if dropped > 0:
        logger.debug(f'Dropped {dropped} NaN rows ({dropped/before_len:.1%} of data)')

    return featured


# ---------------------------------------------------------------------------
# Batch processing: all stocks
# ---------------------------------------------------------------------------

def engineer_all_features(data_dir='data', output_dir='data/features',
                          save_csv=True, save_pickle=True):
    """Run feature engineering on all stock CSVs.

    Args:
        data_dir: Directory with raw stock CSVs.
        output_dir: Directory to save feature CSVs and pickle.
        save_csv: Save per-stock feature CSVs (for inspection).
        save_pickle: Save combined feature dict as pickle (for model loading).

    Returns:
        dict: {ticker: featured_DataFrame} for all successfully processed stocks.
    """
    from src.data.quality import DataQualityChecker
    from src.data.stocks import get_all_tickers

    os.makedirs(output_dir, exist_ok=True)
    qc = DataQualityChecker()

    tickers = get_all_tickers()
    all_features = {}
    failed = []

    for ticker in tickers:
        safe_name = ticker.replace('^', '').replace('.', '_')
        csv_path = os.path.join(data_dir, f'{safe_name}.csv')

        if not os.path.exists(csv_path):
            logger.warning(f'{ticker}: CSV not found at {csv_path}')
            failed.append(ticker)
            continue

        try:
            # Load and clean raw data
            raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            clean = qc.clean_stock(raw)

            # Engineer features
            featured = engineer_stock_features(clean)

            if len(featured) == 0:
                logger.warning(f'{ticker}: no data left after feature engineering')
                failed.append(ticker)
                continue

            all_features[ticker] = featured

            # Save per-stock CSV
            if save_csv:
                feat_path = os.path.join(output_dir, f'{safe_name}_features.csv')
                featured.to_csv(feat_path)

            logger.info(f'{ticker}: {len(featured)} rows, {len(FEATURE_COLUMNS)} features')

        except Exception as e:
            logger.error(f'{ticker}: feature engineering failed — {e}')
            failed.append(ticker)

    # Save combined pickle
    if save_pickle and all_features:
        pkl_path = os.path.join(output_dir, 'all_features.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_features, f)
        logger.info(f'Saved {len(all_features)} stock features to {pkl_path}')

    logger.info(f'Feature engineering complete: {len(all_features)} success, {len(failed)} failed')
    if failed:
        logger.warning(f'Failed: {failed}')

    return all_features


def build_feature_tensor(features_dict, feature_cols=None):
    """Build aligned 3D numpy array from features dict.

    Args:
        features_dict: {ticker: DataFrame} from engineer_all_features.
        feature_cols: Feature columns to include. Defaults to FEATURE_COLUMNS.

    Returns:
        tuple: (tensor, dates, tickers)
            - tensor: np.ndarray of shape (n_stocks, n_timesteps, n_features)
            - dates: DatetimeIndex of common dates
            - tickers: list of ticker strings in same order as tensor axis 0
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Find common date range across all stocks
    date_sets = [set(df.index) for df in features_dict.values()]
    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) == 0:
        raise ValueError('No common dates across stocks')

    dates = pd.DatetimeIndex(common_dates)
    tickers = sorted(features_dict.keys())
    n_stocks = len(tickers)
    n_time = len(dates)
    n_feat = len(feature_cols)

    tensor = np.zeros((n_stocks, n_time, n_feat), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        df = features_dict[ticker]
        aligned = df.loc[dates, feature_cols]
        tensor[i] = aligned.values

    # Final NaN safety check — replace any remaining NaN with 0
    nan_count = np.isnan(tensor).sum()
    if nan_count > 0:
        logger.warning(f'Found {nan_count} NaN in final tensor — replacing with 0')
        tensor = np.nan_to_num(tensor, nan=0.0)

    logger.info(f'Feature tensor: {tensor.shape} (stocks={n_stocks}, time={n_time}, features={n_feat})')
    return tensor, dates, tickers
