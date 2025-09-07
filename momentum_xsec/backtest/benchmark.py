from __future__ import annotations
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger(__name__)


def equal_weight_static(px: pd.DataFrame, start_idx: int) -> pd.Series:
    """Create equal-weight buy-and-hold benchmark equity curve.

    Builds static equal weights at start_idx and tracks performance through the end of px.
    Returns equity series starting at 1.0 and aligned to px.index[start_idx:].

    Args:
        px: Wide price DataFrame (index=datetime, columns=symbols)
        start_idx: Index position to start the benchmark (inclusive)

    Returns:
        Equity series starting at 1.0, indexed by px.index[start_idx:]
    """
    if px.empty:
        logger.debug("Empty price data provided")
        return pd.Series(dtype=float)

    if start_idx >= len(px.index):
        logger.warning(f"Start index {start_idx} >= data length {len(px.index)}")
        return pd.Series(dtype=float)

    if start_idx < 0:
        logger.warning(f"Negative start index: {start_idx}")
        return pd.Series(dtype=float)

    # Get prices from start_idx onwards
    px_subset = px.iloc[start_idx:]

    if px_subset.empty:
        logger.debug("No price data available from start index")
        return pd.Series(dtype=float)

    # Get starting prices (first row of subset)
    start_prices = px_subset.iloc[0]

    # Filter to symbols with valid starting prices
    valid_symbols = start_prices.dropna()
    if len(valid_symbols) == 0:
        logger.warning("No valid starting prices found")
        return pd.Series(dtype=float)

    symbols = list(valid_symbols.index)
    logger.debug(f"Creating equal-weight benchmark with {len(symbols)} symbols starting at index {start_idx}")

    # Calculate returns for each symbol relative to start prices
    # Shape: (n_dates, n_symbols)
    symbol_returns = px_subset[symbols].div(start_prices[symbols])

    # Average across symbols to get equal-weight portfolio return
    # This gives us the return series: first value should be 1.0
    portfolio_returns = symbol_returns.mean(axis=1)

    # Verify first value is 1.0 (or very close due to floating point precision)
    if abs(portfolio_returns.iloc[0] - 1.0) > 1e-10:
        logger.warning(f"First portfolio return should be 1.0, got {portfolio_returns.iloc[0]}")

    # The portfolio_returns series already represents the equity curve
    # starting at 1.0 and tracking the equal-weight portfolio performance
    equity_series = portfolio_returns.copy()
    equity_series.name = 'benchmark_equity'

    logger.debug(f"Benchmark equity curve: {len(equity_series)} points, "
                ".1%")

    return equity_series


