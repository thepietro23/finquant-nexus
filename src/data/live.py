"""Live data utilities — gap-fill CSV from yfinance + real-time price quotes.

Gap-fill: downloads only missing dates (last CSV date → today).
Live price: uses yf.Ticker.fast_info with CSV fallback.
"""

import os
import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from src.data.download import download_stock
from src.data.stocks import get_all_tickers, NIFTY_INDEX
from src.utils.logger import get_logger

logger = get_logger('live')

_UPDATE_LOCK = threading.Lock()
_last_update: datetime | None = None
_data_as_of: str = 'unknown'


# ── Public state ─────────────────────────────────────────────────────────────

def get_data_as_of() -> str:
    """Return the latest date in the price CSV (human-readable)."""
    return _data_as_of


def get_last_update_time() -> datetime | None:
    return _last_update


# ── Gap-fill ─────────────────────────────────────────────────────────────────

def update_price_data(data_dir: str = 'data', force: bool = False) -> dict:
    """Download only the missing date range from yfinance and append to CSV.

    Safe to call from a background thread — acquires a lock to prevent
    concurrent downloads.

    Returns: {'status': 'updated'|'skipped'|'error', 'added_rows': int, 'gap_days': int}
    """
    global _last_update, _data_as_of

    if not _UPDATE_LOCK.acquire(blocking=False):
        logger.info('update_price_data: another update already in progress, skipping')
        return {'status': 'skipped', 'added_rows': 0, 'gap_days': 0}

    try:
        csv_path = os.path.join(data_dir, 'all_close_prices.csv')
        nifty_path = os.path.join(data_dir, 'NIFTY50_INDEX.csv')

        if not os.path.exists(csv_path):
            logger.error(f'CSV not found at {csv_path} — run download first')
            return {'status': 'error', 'added_rows': 0, 'gap_days': 0}

        existing = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        last_date = existing.index.max()
        today = pd.Timestamp(datetime.now().date())

        _data_as_of = last_date.strftime('%d %b %Y')

        # Gap in calendar days (markets closed on weekends — actual trading gap may be smaller)
        gap_days = (today - last_date).days

        if gap_days < 1:
            # Data is already current — nothing to download regardless of force flag.
            logger.info(f'Price data is up to date (last: {last_date.date()})')
            _last_update = datetime.now()
            return {'status': 'skipped', 'added_rows': 0, 'gap_days': 0}

        if gap_days == 1 and not force:
            # Gap is just today — intraday data not finalized until market close (3:30 PM IST).
            # Skip to avoid hammering Yahoo with 45 requests for data that doesn't exist yet.
            logger.info(f'Gap is only today ({today.date()}) — skipping until market close')
            _last_update = datetime.now()
            return {'status': 'skipped', 'added_rows': 0, 'gap_days': gap_days}

        start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end = (today + timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance end is exclusive

        logger.info(f'Filling gap: {start} → {today.date()} ({gap_days} calendar days)')

        tickers = get_all_tickers()
        new_close: dict[str, pd.Series] = {}
        added_rows = 0

        # Batch download — one request for all 50 tickers using Chrome impersonation
        try:
            import yfinance as yf
            from src.data.download import _YF_SESSION
            batch = yf.download(
                tickers, start=start, end=end,
                progress=False, auto_adjust=False,
                group_by='column',
                session=_YF_SESSION,
            )
            if batch is not None and not batch.empty:
                if isinstance(batch.columns, pd.MultiIndex):
                    adj = batch['Adj Close'] if 'Adj Close' in batch.columns.get_level_values(0) else None
                else:
                    adj = batch[['Adj Close']] if 'Adj Close' in batch.columns else None

                if adj is not None:
                    for col in adj.columns:
                        series = adj[col].dropna()
                        if not series.empty:
                            new_close[col] = series
                            added_rows = max(added_rows, len(series))
                    logger.info(f'Batch download: {len(new_close)}/{len(tickers)} tickers OK')
        except Exception as e:
            logger.warning(f'Batch download failed ({e}) — falling back to per-ticker with longer delays')

        # Per-ticker fallback for anything missing from batch
        missing = [t for t in tickers if t not in new_close]
        if missing:
            logger.info(f'Per-ticker fallback for {len(missing)} tickers')
            if len(missing) == len(tickers):
                # Entire batch failed — Yahoo rate limit active; wait before per-ticker
                logger.info('Batch fully failed — waiting 30s before per-ticker fallback')
                time.sleep(30.0)
            for i, ticker in enumerate(missing):
                if i > 0:
                    time.sleep(5.0)  # 5s gap to stay under Yahoo rate limit
                try:
                    df = download_stock(ticker, start, end, retries=3, backoff=4.0)
                    if df is not None and not df.empty and 'Adj Close' in df.columns:
                        new_close[ticker] = df['Adj Close']
                        added_rows = max(added_rows, len(df))
                        logger.debug(f'{ticker}: +{len(df)} rows (fallback)')
                except Exception as exc:
                    logger.warning(f'{ticker}: gap-fill failed — {exc}')

        if not new_close:
            logger.warning('No new data downloaded — Yahoo may be rate-limiting or market closed')
            _last_update = datetime.now()
            return {'status': 'skipped', 'added_rows': 0, 'gap_days': gap_days}

        # Merge new rows into existing DataFrame
        new_df = pd.DataFrame(new_close)
        new_df.index.name = 'Date'

        # Only keep rows that are actually newer than what we have
        new_df = new_df[new_df.index > last_date]

        if new_df.empty:
            logger.info('Gap-fill: no new trading days found (market may have been closed)')
            _last_update = datetime.now()
            return {'status': 'skipped', 'added_rows': 0, 'gap_days': gap_days}

        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.to_csv(csv_path)

        new_last = combined.index.max()
        _data_as_of = new_last.strftime('%d %b %Y')
        logger.info(f'CSV updated: {last_date.date()} → {new_last.date()} (+{len(new_df)} rows)')

        # Gap-fill NIFTY index separately
        try:
            if os.path.exists(nifty_path):
                nifty_existing = pd.read_csv(nifty_path, parse_dates=['Date'], index_col='Date')
                nifty_new = download_stock(NIFTY_INDEX, start, end, retries=3, backoff=2.0)
                if nifty_new is not None and not nifty_new.empty:
                    nifty_new = nifty_new[nifty_new.index > nifty_existing.index.max()]
                    if not nifty_new.empty:
                        pd.concat([nifty_existing, nifty_new]).sort_index().to_csv(nifty_path)
                        logger.info(f'NIFTY index updated: +{len(nifty_new)} rows')
        except Exception as e:
            logger.warning(f'NIFTY index gap-fill failed: {e}')

        _last_update = datetime.now()
        return {'status': 'updated', 'added_rows': len(new_df), 'gap_days': gap_days}

    except Exception as e:
        logger.error(f'update_price_data failed: {e}', exc_info=True)
        return {'status': 'error', 'added_rows': 0, 'gap_days': 0}
    finally:
        _UPDATE_LOCK.release()


# ── Live price quote ──────────────────────────────────────────────────────────

def get_live_price(ticker: str, fallback_price: float | None = None) -> dict:
    """Fetch real-time price for a ticker using yfinance fast_info.

    Falls back to fallback_price (CSV last row) if Yahoo is unreachable.

    Returns:
        {'price': float, 'prev_close': float, 'is_live': bool}
    """
    import yfinance as yf

    ns_ticker = ticker if ticker.endswith('.NS') else f'{ticker}.NS'
    try:
        t = yf.Ticker(ns_ticker)
        fi = t.fast_info

        price = fi.get('last_price') or fi.get('regularMarketPrice')
        prev  = fi.get('previous_close') or fi.get('regularMarketPreviousClose')

        if price and price > 0:
            logger.debug(f'{ticker}: live price ₹{price:.2f}')
            return {
                'price': float(price),
                'prev_close': float(prev) if prev else float(price),
                'is_live': True,
            }
    except Exception as e:
        logger.warning(f'{ticker}: live price failed ({e}) — using CSV fallback')

    # Fallback: use provided CSV last-row value
    return {
        'price': float(fallback_price) if fallback_price else 0.0,
        'prev_close': float(fallback_price) if fallback_price else 0.0,
        'is_live': False,
    }
