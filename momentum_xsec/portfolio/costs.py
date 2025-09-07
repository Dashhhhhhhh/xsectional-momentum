from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger(__name__)


def adv_usd(px: pd.DataFrame, vol: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate Average Daily Volume in USD using rolling window.

    Args:
        px: Wide price DataFrame (index=datetime, columns=symbols)
        vol: Wide volume DataFrame (index=datetime, columns=symbols)
        window: Rolling window size for ADV calculation

    Returns:
        DataFrame of ADV values, forward-filled to handle NaNs where possible
    """
    if px.empty or vol.empty:
        logger.warning("Empty price or volume data provided to adv_usd")
        return pd.DataFrame()

    # Calculate dollar volume
    dollar_vol = px * vol

    # Apply rolling mean
    adv = dollar_vol.rolling(window=window).mean()

    # Forward fill to handle initial NaNs, but don't fill across symbols
    adv = adv.ffill()

    # Replace any remaining NaNs with 0 (for completely missing data)
    adv = adv.fillna(0.0)

    logger.debug(f"Calculated ADV for {adv.shape[1]} symbols using {window}-day window")
    return adv


def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """Calculate portfolio turnover as sum of absolute weight changes.

    Args:
        prev_w: Previous portfolio weights
        new_w: New portfolio weights

    Returns:
        Turnover as fraction (0.0 to 2.0 for long-short portfolio)
    """
    if prev_w.empty and new_w.empty:
        return 0.0

    # Align indices and fill missing values with 0
    idx = prev_w.index.union(new_w.index)
    prev_w_aligned = prev_w.reindex(idx).fillna(0.0)
    new_w_aligned = new_w.reindex(idx).fillna(0.0)

    # Calculate absolute weight changes
    weight_changes = np.abs(new_w_aligned - prev_w_aligned)

    turnover_value = float(weight_changes.sum())

    logger.debug(f"Calculated turnover: {turnover_value:.4f}")
    return turnover_value


def cost_bps_for_order(order_value: float, adv_value: float | None, fixed_bps: int, k: int) -> float:
    """Calculate trading cost in basis points for a single order.

    Uses fixed cost + square-root impact model: fixed_bps + k * sqrt(order_size / ADV)

    Args:
        order_value: Dollar value of the order
        adv_value: Average daily volume in USD (None if unavailable)
        fixed_bps: Fixed cost in basis points
        k: Square-root impact parameter

    Returns:
        Total cost in basis points
    """
    # Fixed cost component
    total_bps = float(fixed_bps)

    # Impact cost component (only if ADV is available)
    if adv_value is not None and adv_value > 0:
        # Calculate order size as fraction of ADV
        size_ratio = abs(order_value) / adv_value

        # Cap the size ratio at 1.0 to prevent excessive impact costs
        capped_ratio = min(1.0, size_ratio)

        # Square-root impact model
        impact_bps = k * np.sqrt(capped_ratio) * 100.0  # Convert to basis points

        total_bps += impact_bps

        logger.debug(".1f")

    return total_bps


def rebalance_cost(port_value: float, prev_w: pd.Series, new_w: pd.Series,
                  prices_row: pd.Series, adv_row: pd.Series, fixed_bps: int, k: int) -> float:
    """Calculate total trading costs for a portfolio rebalance.

    Args:
        port_value: Current portfolio value in dollars
        prev_w: Previous portfolio weights
        new_w: Target portfolio weights
        prices_row: Current prices for all symbols
        adv_row: Current ADV values for all symbols
        fixed_bps: Fixed cost in basis points
        k: Square-root impact parameter

    Returns:
        Total trading cost in dollars
    """
    if prev_w.empty and new_w.empty:
        return 0.0

    # Align weight vectors
    idx = prev_w.index.union(new_w.index)
    prev_w_aligned = prev_w.reindex(idx).fillna(0.0)
    new_w_aligned = new_w.reindex(idx).fillna(0.0)

    # Calculate weight changes (trades)
    trades_w = new_w_aligned - prev_w_aligned

    total_cost = 0.0
    trade_count = 0

    for sym, w_delta in trades_w.items():
        if w_delta == 0:
            continue

        # Calculate notional trade value
        notional = port_value * w_delta

        # Skip dust trades (minimum $50 or 0.05% of portfolio)
        min_trade_size = max(50.0, 0.0005 * port_value)
        if abs(notional) < min_trade_size:
            logger.debug(".1f")
            continue

        # Get price and ADV for this symbol
        price = prices_row.get(sym)
        adv_value = adv_row.get(sym)

        if price is None or np.isnan(price) or price <= 0:
            logger.warning(f"No valid price for {sym}, skipping trade")
            continue

        # Calculate cost in basis points
        bps_cost = cost_bps_for_order(notional, adv_value, fixed_bps, k)

        # Convert to dollar cost
        dollar_cost = abs(notional) * (bps_cost / 10000.0)

        total_cost += dollar_cost
        trade_count += 1

        logger.debug(".1f")

    logger.info(".1f")
    return total_cost


