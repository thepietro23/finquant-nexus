"""Indian financial news via RSS feeds — no API key, no rate limits.

Fetches from Economic Times Markets, Business Standard, Moneycontrol, LiveMint.
Uses feedparser (already a project dependency via news_fetcher.py).
Articles are associated with NIFTY 50 tickers via keyword matching on titles.
"""
from __future__ import annotations

import socket
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser

from src.utils.logger import get_logger

logger = get_logger('indian_rss')

RSS_FEEDS = [
    ('https://economictimes.indiatimes.com/markets/rss.cms',      'Economic Times'),
    ('https://www.business-standard.com/rss/markets-106.rss',     'Business Standard'),
    ('https://www.moneycontrol.com/rss/marketsnews.xml',           'Moneycontrol'),
    ('https://www.livemint.com/rss/markets',                       'LiveMint'),
]

# Longest / most specific keywords first so "HDFC BANK" matches before "HDFC"
KEYWORD_TICKER: list[tuple[str, str]] = [
    ('TATA CONSULTANCY', 'TCS.NS'),
    ('HDFC BANK',        'HDFCBANK.NS'),
    ('ICICI BANK',       'ICICIBANK.NS'),
    ('STATE BANK',       'SBIN.NS'),
    ('BHARTI AIRTEL',    'BHARTIARTL.NS'),
    ('HINDUSTAN UNILEVER', 'HINDUNILVR.NS'),
    ('TATA MOTORS',      'TATAMOTORS.NS'),
    ('TATA STEEL',       'TATASTEEL.NS'),
    ('SUN PHARMA',       'SUNPHARMA.NS'),
    ('SUN PHARMACEUTICAL', 'SUNPHARMA.NS'),
    ('LARSEN & TOUBRO',  'LT.NS'),
    ('BAJAJ FINANCE',    'BAJFINANCE.NS'),
    ('MARUTI SUZUKI',    'MARUTI.NS'),
    ('KOTAK MAHINDRA',   'KOTAKBANK.NS'),
    ('ADANI ENTERPRISES','ADANIENT.NS'),
    ('INFOSYS',          'INFY.NS'),
    ('RELIANCE',         'RELIANCE.NS'),
    ('AIRTEL',           'BHARTIARTL.NS'),
    ('WIPRO',            'WIPRO.NS'),
    ('MARUTI',           'MARUTI.NS'),
    ('NTPC',             'NTPC.NS'),
    ('TITAN',            'TITAN.NS'),
    ('ADANI',            'ADANIENT.NS'),
    ('KOTAK',            'KOTAKBANK.NS'),
    ('SBI',              'SBIN.NS'),
    ('TCS',              'TCS.NS'),
    ('ITC',              'ITC.NS'),
    ('HUL',              'HINDUNILVR.NS'),
]


def _fetch_one_feed(url: str, source_name: str) -> list[dict]:
    try:
        old_to = socket.getdefaulttimeout()
        socket.setdefaulttimeout(10)
        try:
            feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
        finally:
            socket.setdefaulttimeout(old_to)

        results = []
        for entry in feed.entries[:25]:
            dt = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    dt = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass
            title = (entry.get('title') or '').strip()
            if title:
                results.append({
                    'title': title,
                    'published': dt,
                    'link': entry.get('link', ''),
                    'source': source_name,
                })
        logger.info(f'RSS {source_name}: {len(results)} articles')
        return results
    except Exception as e:
        logger.warning(f'RSS fetch failed ({source_name}): {e}')
        return []


def _match_ticker(title: str) -> str:
    """Return NSE ticker (e.g. 'TCS.NS') or '' for market-wide articles."""
    upper = title.upper()
    for kw, ticker in KEYWORD_TICKER:
        if kw in upper:
            return ticker
    return ''


def fetch_indian_news(max_age_days: int = 3) -> list[dict]:
    """Fetch all RSS feeds in parallel, return normalised article list.

    Each article dict has keys:
        title, published (datetime|None), source, ticker (e.g. 'TCS.NS' or ''),
        short_ticker ('TCS' or 'MARKET'), sector.

    Sorted newest-first. Older-than-max_age_days articles are dropped.
    """
    from datetime import timedelta
    from src.data.stocks import get_sector

    cutoff = datetime.now() - timedelta(days=max_age_days)

    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_fetch_one_feed, url, name) for url, name in RSS_FEEDS]
        for f in as_completed(futs):
            try:
                raw.extend(f.result())
            except Exception:
                pass

    seen: set[str] = set()
    result: list[dict] = []
    for art in raw:
        title = art['title']
        if title in seen:
            continue
        seen.add(title)

        dt = art['published']
        if dt and dt < cutoff:
            continue   # too old

        ticker = _match_ticker(title)
        short = ticker.replace('.NS', '') if ticker else 'MARKET'
        sector = get_sector(ticker) if ticker else 'Market'

        result.append({
            'title':        title,
            'published':    dt,
            'source':       art['source'],
            'ticker':       ticker,
            'short_ticker': short,
            'sector':       sector or 'Market',
        })

    # Newest-first; no-date articles go to bottom
    result.sort(
        key=lambda x: x['published'].timestamp() if x['published'] else 0.0,
        reverse=True,
    )
    logger.info(f'indian_rss: {len(result)} articles total (≤{max_age_days} days)')
    return result
