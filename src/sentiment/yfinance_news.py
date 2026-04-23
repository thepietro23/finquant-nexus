"""Alternative news fetcher using yfinance / Yahoo Finance.

Google News RSS kept returning months-old cached articles (Jan/Feb).
Yahoo Finance via yfinance returns genuinely recent articles (last 24-72h)
for NSE stocks with no API key required.

Sources include: Moneycontrol, Economic Times, Business Standard, LiveMint.
"""
from __future__ import annotations

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger('yfinance_news')

# Conservative concurrency — Yahoo Finance rate-limits aggressively on burst requests
_MAX_WORKERS = 3
_RETRY_DELAY = 2.0   # seconds to wait before one retry on rate-limit


def fetch_yfinance_news(ticker: str, max_results: int = 5) -> list[dict]:
    """Fetch recent news for one ticker via Yahoo Finance.

    Returns list of {title, published (datetime|None), link, source}.
    Retries once after a short sleep on rate-limit errors.
    """
    for attempt in range(2):
        try:
            t = yf.Ticker(ticker)
            raw: list = t.news or []

            results = []
            for item in raw[:max_results]:
                ts = item.get('providerPublishTime', 0)
                published = datetime.fromtimestamp(ts) if ts else None
                title = item.get('title', '').strip()
                if not title:
                    continue
                results.append({
                    'title': title,
                    'published': published,
                    'link': item.get('link', ''),
                    'source': item.get('publisher', ''),
                })
            return results

        except Exception as e:
            err = str(e)
            if 'RateLimit' in err or 'Too Many Requests' in err or '429' in err:
                if attempt == 0:
                    logger.warning(f'{ticker}: rate limited, retrying in {_RETRY_DELAY}s')
                    time.sleep(_RETRY_DELAY)
                    continue
            logger.warning(f'yfinance news failed for {ticker}: {e}')
            return []

    return []


def fetch_all_yfinance_news(
    tickers: list[str],
    max_per_ticker: int = 5,
) -> dict[str, list[dict]]:
    """Fetch news for multiple tickers with limited concurrency to avoid rate limits."""
    result: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(fetch_yfinance_news, t, max_per_ticker): t for t in tickers}
        for f in as_completed(futs):
            ticker = futs[f]
            try:
                result[ticker] = f.result()
            except Exception:
                result[ticker] = []
    return result
