from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger(__name__)


def equal_weight(symbols: list[str]) -> pd.Series:
    """Create equal-weight portfolio across symbols.

    Args:
        symbols: List of symbol names

    Returns:
        Series with equal weights summing to 1.0, indexed by symbol
    """
    if not symbols:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / len(symbols), index=symbols, name='weight')
    logger.debug(f"Created equal weights for {len(symbols)} symbols")
    return w


def risk_scaled_weights(symbols: list[str], ret: pd.DataFrame, asof: pd.Timestamp, trail_days: int) -> pd.Series:
    """Create risk-scaled weights (1/vol) using trailing return data.

    Args:
        symbols: List of symbol names to weight
        ret: Wide returns DataFrame (index=datetime, columns=symbols)
        asof: Timestamp for which to compute weights (uses data up to asof, exclusive)
        trail_days: Number of trailing days for volatility calculation

    Returns:
        Series of risk-scaled weights summing to 1.0, indexed by symbol
    """
    if not symbols:
        return pd.Series(dtype=float)

    # Find the index for asof (data used should be strictly before asof)
    try:
        end_idx = ret.index.get_loc(asof)
    except KeyError:
        logger.warning(f"Date {asof} not found in returns data")
        return equal_weight(symbols)

    # Calculate trailing window: end_idx is exclusive (up to but not including asof)
    start_idx = max(0, end_idx - trail_days)

    if start_idx >= end_idx:
        logger.warning(f"Insufficient trailing data for {asof}: start_idx={start_idx}, end_idx={end_idx}")
        return equal_weight(symbols)

    # Extract trailing returns for the symbol universe
    sub = ret.iloc[start_idx:end_idx][symbols]

    # Calculate volatility for each symbol
    vol = sub.std()

    # Handle zero/nan volatilities
    vol = vol.replace(0.0, np.nan)

    if vol.isna().all():
        logger.warning("All symbols have zero/nan volatility, falling back to equal weight")
        return equal_weight(symbols)

    # Calculate inverse volatility weights
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.fillna(0.0)

    # Normalize to sum to 1
    total_inv_vol = inv_vol.sum()
    if total_inv_vol == 0:
        logger.warning("Sum of inverse volatilities is zero, falling back to equal weight")
        return equal_weight(symbols)

    w = inv_vol / total_inv_vol
    w.name = 'weight'
    logger.debug(f"Created risk-scaled weights for {len(symbols)} symbols using {len(sub)} trailing days")
    return w


def cap_weights_per_name(w: pd.Series, cap: float | None) -> pd.Series:
    """Cap per-name weights within each sign group and renormalize.

    For long-short portfolios, caps should be applied separately to long and short
    sleeves, then recombined to maintain net-zero exposure.

    Args:
        w: Weight series (can be positive and negative)
        cap: Maximum absolute weight per name (None or <=0 to disable)

    Returns:
        Weight series with per-name caps applied and renormalized
    """
    if cap is None or cap <= 0:
        return w

    # Split into long and short positions
    long_mask = w > 0
    short_mask = w < 0

    w_capped = w.copy()

    # Cap long positions
    if long_mask.any():
        long_weights = w[long_mask]
        long_capped = np.sign(long_weights) * np.minimum(np.abs(long_weights), cap)

        # Renormalize long sleeve to preserve gross exposure
        long_gross_before = np.abs(long_weights).sum()
        long_gross_after = np.abs(long_capped).sum()

        if long_gross_after > 0:
            long_capped = long_capped * (long_gross_before / long_gross_after)

        w_capped.loc[long_mask] = long_capped

    # Cap short positions (note: short weights are negative)
    if short_mask.any():
        short_weights = w[short_mask]
        short_capped = np.sign(short_weights) * np.minimum(np.abs(short_weights), cap)

        # Renormalize short sleeve to preserve gross exposure
        short_gross_before = np.abs(short_weights).sum()
        short_gross_after = np.abs(short_capped).sum()

        if short_gross_after > 0:
            short_capped = short_capped * (short_gross_before / short_gross_after)

        w_capped.loc[short_mask] = short_capped

    logger.debug(f"Applied {cap:.1%} cap to {len(w)} positions")
    return w_capped


def trailing_cov(ret: pd.DataFrame, asof: pd.Timestamp, trail_days: int) -> pd.DataFrame:
    """Compute trailing covariance matrix ending at asof (exclusive).

    Args:
        ret: Wide returns DataFrame (index=datetime, columns=symbols)
        asof: End timestamp for covariance window (uses data up to asof, exclusive)
        trail_days: Number of trailing days for covariance calculation

    Returns:
        Covariance matrix DataFrame
    """
    try:
        end_idx = ret.index.get_loc(asof)
    except KeyError:
        logger.warning(f"Date {asof} not found in returns data")
        return pd.DataFrame()

    # Calculate trailing window: end_idx is exclusive
    start_idx = max(0, end_idx - trail_days)

    if start_idx >= end_idx:
        logger.warning(f"Insufficient data for covariance: start_idx={start_idx}, end_idx={end_idx}")
        return pd.DataFrame()

    # Extract trailing returns
    sub = ret.iloc[start_idx:end_idx]

    if sub.empty:
        logger.warning("No data available for covariance calculation")
        return pd.DataFrame()

    cov = sub.cov()
    logger.debug(f"Computed {cov.shape[0]}×{cov.shape[1]} covariance matrix using {len(sub)} observations")
    return cov


def ann_vol_from_cov(w: pd.Series, cov: pd.DataFrame) -> float:
    """Calculate annualized volatility from weight vector and covariance matrix.

    Args:
        w: Weight vector Series
        cov: Covariance matrix DataFrame

    Returns:
        Annualized volatility (√252 scaling), or NaN if invalid inputs
    """
    if cov.empty or w.empty:
        logger.debug("Empty covariance matrix or weight vector")
        return np.nan

    # Find common symbols between weights and covariance
    common_symbols = [s for s in w.index if s in cov.index]
    if not common_symbols:
        logger.debug("No common symbols between weights and covariance matrix")
        return np.nan

    # Extract relevant submatrix
    w_common = w.loc[common_symbols]
    cov_common = cov.loc[common_symbols, common_symbols]

    # Calculate portfolio variance: w' * Σ * w
    try:
        port_var = np.dot(w_common.values, np.dot(cov_common.values, w_common.values))
        if port_var < 0:
            logger.warning(f"Negative portfolio variance: {port_var}")
            return np.nan

        # Annualize: √(variance) * √(252)
        ann_vol = np.sqrt(port_var) * np.sqrt(252.0)
        return float(ann_vol)
    except Exception as e:
        logger.warning(f"Error calculating portfolio volatility: {e}")
        return np.nan


def apply_vol_target(w: pd.Series, cov: pd.DataFrame, target_ann_vol: float, max_leverage: float = 5.0) -> pd.Series:
    """Apply volatility targeting by scaling weights to achieve target annualized volatility.

    Args:
        w: Weight vector to scale
        cov: Covariance matrix for volatility calculation
        target_ann_vol: Target annualized volatility
        max_leverage: Maximum leverage factor to apply (default 5.0)

    Returns:
        Scaled weight vector, or original weights if volatility calculation fails
    """
    if w.empty:
        return w

    current_vol = ann_vol_from_cov(w, cov)

    # Check for valid volatility calculation
    if not np.isfinite(current_vol) or current_vol <= 0:
        logger.warning(f"Invalid current volatility ({current_vol}), returning original weights")
        return w

    # If current volatility is already above target, don't leverage up
    if current_vol >= target_ann_vol:
        logger.debug(f"Current vol ({current_vol:.1%}) already >= target ({target_ann_vol:.1%}), no leverage applied")
        return w

    # Calculate leverage factor
    leverage = target_ann_vol / current_vol

    # Cap leverage to prevent extreme values
    if leverage > max_leverage:
        logger.warning(f"Leverage factor {leverage:.1f}x capped at {max_leverage:.1f}x")
        leverage = max_leverage

    # Apply leverage
    w_scaled = w * leverage

    logger.debug(f"Applied {leverage:.3f}x leverage to achieve {target_ann_vol:.1%} target vol (current: {current_vol:.1%})")
    return w_scaled


