from __future__ import annotations
import time
import pathlib
import pandas as pd
import requests
import requests_cache
from ..utils.logging import get_logger

_BASE = 'https://www.alphavantage.co/query'
logger = get_logger(__name__)


def _ensure_cache_dir(path: str) -> None:
    p = pathlib.Path(path).parent
    p.mkdir(parents=True, exist_ok=True)


def fetch_equity_daily_adjusted(symbol: str, api_key: str) -> pd.DataFrame:
    """Fetch TIME_SERIES_DAILY and return DataFrame with columns: [open, high, low, close, volume].
    - close is unadjusted close price (field "4. close")
    - uses requests + requests_cache (sqlite DB at ./cache, expire_after=12h)
    - sleeps ~12s on cache miss to respect free-tier rate limits
    - raises clear errors on invalid key / throttling
    """
    if not api_key or api_key == 'REPLACE_ME':
        raise ValueError(f"Invalid API key for Alpha Vantage. Please set ALPHAVANTAGE_API_KEY environment variable.")

    cache_path = 'cache/alphavantage.sqlite'
    _ensure_cache_dir(cache_path)
    session = requests_cache.CachedSession(cache_path, expire_after=60 * 60 * 12)

    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full',
        'datatype': 'json',
    }

    try:
        resp = session.get(_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Network error fetching {symbol}: {e}")

    # Check for API errors
    if 'Error Message' in data:
        msg = data['Error Message']
        if 'Invalid API call' in msg or 'Invalid API key' in msg:
            raise ValueError(f"Invalid API key for Alpha Vantage: {msg}")
        else:
            raise RuntimeError(f"Alpha Vantage API error for {symbol}: {msg}")

    if 'Information' in data:
        msg = data['Information']
        if 'premium' in msg.lower() or 'subscribe' in msg.lower():
            raise ValueError(f"Invalid or missing API key for Alpha Vantage. Please check your ALPHAVANTAGE_API_KEY environment variable. Response: {msg}")
        logger.info(f"Alpha Vantage info for {symbol}: {msg}")

    if 'Note' in data:
        note = data['Note']
        if 'higher API call frequency' in note or 'rate limit' in note.lower():
            raise RuntimeError(f"Alpha Vantage rate limit exceeded: {note}")
        logger.warning(f"Alpha Vantage note for {symbol}: {note}")

    if 'Time Series (Daily)' not in data:
        raise RuntimeError(f"No data returned for {symbol}. Response keys: {list(data.keys())}")

    ts = data['Time Series (Daily)']
    if not ts:
        raise RuntimeError(f"Empty time series data for {symbol}")

    df = (
        pd.DataFrame.from_dict(ts, orient='index')
        .rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
        })[['open', 'high', 'low', 'close', 'volume']]
        .sort_index()
    )

    if df.empty:
        raise RuntimeError(f"No data points for {symbol}")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.astype(float)

    # Sleep only on cache miss to respect rate limits
    if not getattr(resp, 'from_cache', False):
        logger.info(f"Fetched {symbol} from API (cache miss), sleeping 12s...")
        time.sleep(12)
    else:
        logger.info(f"Loaded {symbol} from cache")

    return df


