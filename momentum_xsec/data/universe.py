from __future__ import annotations
from typing import List
import pandas as pd
from ..utils.logging import get_logger

from .alpha_vantage import fetch_equity_daily_adjusted

logger = get_logger(__name__)


def build_universe(equities: List[str], start_date: str, api_key: str) -> pd.DataFrame:
    """Build long panel DataFrame from list of equities.
    - Calls fetch_equity_daily_adjusted for each symbol
    - Filters to date >= start_date
    - Includes asset_class='equity'
    - Skips symbols that hard-fail, logs warnings
    - Returns long panel with columns: [date, symbol, asset_class, open, high, low, close, volume]
    """
    frames = []
    start_dt = pd.to_datetime(start_date, utc=True)

    for sym in equities:
        try:
            df = fetch_equity_daily_adjusted(sym, api_key)
            df = df.loc[df.index >= start_dt]
            if df.empty:
                logger.warning(f"No data for {sym} after {start_date}, skipping")
                continue

            df = df.assign(symbol=sym, asset_class='equity')
            frames.append(df.reset_index().rename(columns={'index': 'date'}))

        except Exception as e:
            logger.warning(f"Failed to fetch {sym}: {e}, skipping")
            continue

    if not frames:
        raise RuntimeError(f"No valid data fetched for any symbols in universe")

    long_panel = pd.concat(frames, ignore_index=True)
    long_panel = long_panel[['date', 'symbol', 'asset_class', 'open', 'high', 'low', 'close', 'volume']]
    long_panel['date'] = pd.to_datetime(long_panel['date'])
    logger.info(f"Built universe with {len(long_panel['symbol'].unique())} symbols and {len(long_panel)} total rows")
    return long_panel


def wide_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long panel to wide price format.
    - Pivot to close prices by date/symbol
    - Sort index
    - Forward fill missing values
    - Drop columns with any NaN (strict clean panel)
    """
    px = df.pivot_table(index='date', columns='symbol', values='close').sort_index()
    px = px.ffill()
    px = px.dropna(axis=1, how='any')
    logger.info(f"Wide prices: {px.shape[0]} dates × {px.shape[1]} symbols")
    return px


def wide_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long panel to wide volume format.
    - Pivot to volume by date/symbol
    - Sort index
    - Forward fill missing values
    """
    vol = df.pivot_table(index='date', columns='symbol', values='volume').sort_index()
    vol = vol.ffill()
    # Note: volume can have NaN values, don't drop columns
    logger.info(f"Wide volume: {vol.shape[0]} dates × {vol.shape[1]} symbols")
    return vol


def monthly_endpoints(px: pd.DataFrame) -> pd.DatetimeIndex:
    """Compute monthly endpoints intersected with actual available days.
    - Generate month-end dates
    - Intersect with px.index to get only available dates
    """
    month_ends = px.resample('ME').last().index
    available_ends = px.index.intersection(month_ends)
    logger.info(f"Monthly endpoints: {len(available_ends)} months")
    return available_ends


