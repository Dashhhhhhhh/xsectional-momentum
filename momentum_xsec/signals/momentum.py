from __future__ import annotations
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger(__name__)


def lookback_return(px: pd.DataFrame, asof: pd.Timestamp, lb_days: int, skip_days: int) -> pd.Series:
    """Compute lookback momentum returns with skip window to avoid look-ahead bias.

    Args:
        px: Wide price DataFrame (index=datetime, columns=symbols)
        asof: Timestamp for which to compute signal (use data strictly before this)
        lb_days: Lookback window length in days
        skip_days: Skip window to avoid look-ahead (e.g., 1 month = 21 days)

    Returns:
        Series of returns indexed by symbol, empty if insufficient history
    """
    # Find the index position for 'asof'
    try:
        end_idx = px.index.get_loc(asof)
    except KeyError:
        logger.warning(f"Date {asof} not found in price data")
        return pd.Series(dtype=float)

    # Calculate indices: end_idx is the skip-adjusted end point
    skip_idx = max(0, end_idx - skip_days)
    start_idx = max(0, skip_idx - lb_days)

    # Check if we have sufficient history
    if start_idx >= skip_idx or skip_idx >= len(px):
        logger.debug(f"Insufficient history for {asof}: start_idx={start_idx}, skip_idx={skip_idx}, data_len={len(px)}")
        return pd.Series(dtype=float)

    # Get price series at start and end points
    start_prices = px.iloc[start_idx]  # Series: symbol -> price at start
    end_prices = px.iloc[skip_idx]     # Series: symbol -> price at end

    # Compute returns: (end_price / start_price) - 1
    returns = (end_prices / start_prices) - 1.0

    # Drop NaN values (symbols with missing data)
    returns = returns.dropna()

    logger.debug(f"Computed momentum for {len(returns)} symbols as of {asof}")
    return returns


def rank_deciles(signal: pd.Series, top: float, bottom: float, min_names: int = 20) -> tuple[list[str], list[str]]:
    """Rank momentum signals into top/bottom deciles for long/short positions.

    Args:
        signal: Series of momentum signals indexed by symbol
        top: Top quantile threshold (e.g., 0.1 for top decile)
        bottom: Bottom quantile threshold (e.g., 0.1 for bottom decile)
        min_names: Minimum number of names required, else return empty lists

    Returns:
        Tuple of (long_symbols, short_symbols) lists
    """
    if signal.empty:
        logger.debug("Signal series is empty, returning empty position lists")
        return [], []

    n = len(signal)
    if n < min_names:
        logger.debug(f"Only {n} signals available, need at least {min_names}, returning empty lists")
        return [], []

    # Sort by signal value descending (highest momentum first)
    # For momentum: higher returns = better for longs, lower returns = better for shorts
    sorted_signal = signal.sort_values(ascending=False, kind='mergesort')

    # Calculate number of positions for each side
    k_top = max(1, int(n * top))
    k_bot = max(1, int(n * bottom))

    # Select longs from top of sorted list (highest momentum)
    longs = list(sorted_signal.iloc[:k_top].index)

    # Select shorts from bottom of sorted list (lowest momentum)
    shorts = list(sorted_signal.iloc[-k_bot:].index)

    logger.info(f"Selected {len(longs)} longs and {len(shorts)} shorts from {n} available signals")
    return longs, shorts


