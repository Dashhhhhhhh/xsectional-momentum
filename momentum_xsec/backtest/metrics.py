from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger(__name__)


def annualized_return(equity: pd.Series) -> float:
    """Calculate annualized return from equity curve.

    Args:
        equity: Time series of portfolio equity values

    Returns:
        Annualized return as decimal (e.g., 0.10 for 10%), or NaN for invalid data
    """
    if equity.empty:
        logger.debug("Empty equity series provided")
        return np.nan

    if len(equity) < 2:
        logger.debug("Equity series must have at least 2 points")
        return np.nan

    # Get start and end values
    start_value = equity.iloc[0]
    end_value = equity.iloc[-1]

    if start_value <= 0:
        logger.warning(f"Invalid starting equity value: {start_value}")
        return np.nan

    # Calculate total return
    total_return = end_value / start_value

    # Calculate time period in years
    start_date = equity.index[0]
    end_date = equity.index[-1]

    if isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
        days = (end_date - start_date).days
        years = days / 365.25
    else:
        # Fallback for non-datetime indices
        years = len(equity) / 252.0  # Assume daily data

    if years <= 0:
        logger.debug(f"Invalid time period: {years} years")
        return np.nan

    # Annualize the return
    annualized_return = total_return ** (1.0 / years) - 1.0

    logger.debug(".1%")
    return float(annualized_return)


def annualized_vol(ret: pd.Series) -> float:
    """Calculate annualized volatility from return series.

    Args:
        ret: Time series of daily returns

    Returns:
        Annualized volatility as decimal (e.g., 0.15 for 15%), or NaN for invalid data
    """
    if ret.empty:
        logger.debug("Empty return series provided")
        return np.nan

    if len(ret) < 2:
        logger.debug("Return series must have at least 2 points")
        return np.nan

    # Calculate standard deviation of returns
    daily_vol = ret.std()

    if np.isnan(daily_vol) or daily_vol < 0:
        logger.warning(f"Invalid daily volatility: {daily_vol}")
        return np.nan

    # Annualize (assuming 252 trading days per year)
    annualized_vol = daily_vol * np.sqrt(252.0)

    logger.debug(".1%")
    return float(annualized_vol)


def sharpe(ret: pd.Series, rf: float = 0.0) -> float:
    """Calculate Sharpe ratio from return series.

    Args:
        ret: Time series of daily returns
        rf: Annual risk-free rate as decimal (e.g., 0.02 for 2%)

    Returns:
        Sharpe ratio, or NaN for invalid data
    """
    if ret.empty:
        logger.debug("Empty return series provided")
        return np.nan

    if len(ret) < 2:
        logger.debug("Return series must have at least 2 points")
        return np.nan

    # Calculate daily risk-free rate
    daily_rf = rf / 252.0

    # Calculate excess returns
    excess_returns = ret - daily_rf

    # Calculate mean excess return and volatility
    mean_excess = excess_returns.mean()
    vol = ret.std()

    if np.isnan(vol) or vol <= 0:
        logger.warning(f"Invalid volatility for Sharpe calculation: {vol}")
        return np.nan

    if np.isnan(mean_excess):
        logger.warning("Invalid mean excess return for Sharpe calculation")
        return np.nan

    # Calculate Sharpe ratio (annualized)
    sharpe_ratio = (mean_excess / vol) * np.sqrt(252.0)

    logger.debug(".2f")
    return float(sharpe_ratio)


def max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity: Time series of portfolio equity values

    Returns:
        Maximum drawdown as decimal (e.g., -0.15 for -15%), or NaN for invalid data
    """
    if equity.empty:
        logger.debug("Empty equity series provided")
        return np.nan

    if len(equity) < 2:
        logger.debug("Equity series must have at least 2 points")
        return np.nan

    # Find rolling maximum (peak equity)
    peak = equity.cummax()

    # Calculate drawdown: (current - peak) / peak
    drawdown = (equity - peak) / peak

    # Find minimum drawdown (most negative)
    max_dd = drawdown.min()

    if np.isnan(max_dd):
        logger.warning("Could not calculate valid drawdown")
        return np.nan

    logger.debug(".1%")
    return float(max_dd)


