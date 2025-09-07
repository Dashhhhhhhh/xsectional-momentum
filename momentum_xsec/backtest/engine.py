from __future__ import annotations
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..utils.logging import get_logger
from ..data.universe import monthly_endpoints
from ..signals.momentum import lookback_return, rank_deciles
from ..portfolio.construct import (
    risk_scaled_weights, cap_weights_per_name, trailing_cov, apply_vol_target,
)
from ..portfolio.costs import adv_usd, turnover, rebalance_cost
from .metrics import annualized_return, annualized_vol, sharpe, max_drawdown
from .benchmark import equal_weight_static

logger = get_logger(__name__)


def run_backtest(px: pd.DataFrame, vol: pd.DataFrame, settings) -> dict:
    """Run complete momentum strategy backtest with monthly rebalances.

    Args:
        px: Wide price DataFrame (index=datetime, columns=symbols)
        vol: Wide volume DataFrame (index=datetime, columns=symbols)
        settings: Configuration object with strategy parameters

    Returns:
        Dict containing performance DataFrame and metrics
    """
    # Calculate returns and identify rebalance dates
    ret = px.pct_change().fillna(0.0)
    all_months = monthly_endpoints(px)

    # Select rebalance dates: skip early months with insufficient history
    # Need enough history for lookback + some buffer for covariance estimation
    min_history_months = max(6, settings.lookback_days // 21)  # At least 6 months or lookback period in months
    if min_history_months >= len(all_months):
        raise ValueError(f"Insufficient data: need at least {min_history_months} months, have {len(all_months)}")

    rebal_dates = all_months[min_history_months:]
    logger.info(f"Running backtest with {len(rebal_dates)} rebalance dates from {rebal_dates[0]} to {rebal_dates[-1]}")

    # Pre-calculate ADV for cost modeling
    adv = adv_usd(px, vol, window=settings.adv_window)

    # Initialize tracking variables
    equity = 1_000_000.0
    prev_w = pd.Series(dtype=float)

    # Daily tracking for performance DataFrame
    daily_equity = {}
    daily_turnover = {}
    daily_long_n = {}
    daily_short_n = {}

    # Process each rebalance date
    for i, rebal_date in enumerate(rebal_dates):
        logger.debug(f"Processing rebalance {i+1}/{len(rebal_dates)}: {rebal_date}")

        # 1) Calculate momentum signals
        mom_signal = lookback_return(px, rebal_date, settings.lookback_days, settings.skip_days)
        if mom_signal.empty:
            logger.warning(f"No momentum signal for {rebal_date}, skipping")
            continue

        # 2) Rank into deciles
        longs, shorts = rank_deciles(mom_signal, settings.top_decile, settings.bottom_decile, min_names=len(settings.universe))
        if not longs and not shorts:
            logger.warning(f"No positions selected for {rebal_date}, skipping")
            continue

        # 3) Build sleeve weights with risk scaling
        w_long_raw = pd.Series(dtype=float)
        w_short_raw = pd.Series(dtype=float)

        if longs:
            w_long_raw = risk_scaled_weights(longs, ret, rebal_date, settings.trail_cov_days)
            # Normalize long sleeve to sum to 1
            if not w_long_raw.empty and w_long_raw.sum() > 0:
                w_long_raw = w_long_raw / w_long_raw.sum()

        if shorts:
            w_short_raw = risk_scaled_weights(shorts, ret, rebal_date, settings.trail_cov_days)
            # Normalize short sleeve to sum to 1
            if not w_short_raw.empty and w_short_raw.sum() > 0:
                w_short_raw = w_short_raw / w_short_raw.sum()

        # 4) Create base target weights: +0.5*long_sleeve - 0.5*short_sleeve
        w_target_base = pd.Series(dtype=float)

        if not w_long_raw.empty:
            w_target_base = w_target_base.add(0.5 * w_long_raw, fill_value=0.0)

        if not w_short_raw.empty:
            w_target_base = w_target_base.add(-0.5 * w_short_raw, fill_value=0.0)

        if w_target_base.empty:
            logger.warning(f"No target weights for {rebal_date}, skipping")
            continue

        # 5) Apply per-name caps separately to long and short sides, then recombine
        w_long_capped = cap_weights_per_name(w_long_raw, settings.max_weight_per_name) if not w_long_raw.empty else pd.Series(dtype=float)
        w_short_capped = cap_weights_per_name(w_short_raw, settings.max_weight_per_name) if not w_short_raw.empty else pd.Series(dtype=float)

        # Reconstruct target weights after capping
        w_target_capped = pd.Series(dtype=float)
        if not w_long_capped.empty:
            w_target_capped = w_target_capped.add(0.5 * w_long_capped, fill_value=0.0)
        if not w_short_capped.empty:
            w_target_capped = w_target_capped.add(-0.5 * w_short_capped, fill_value=0.0)

        # Apply volatility targeting using trailing covariance
        cov = trailing_cov(ret, rebal_date, settings.trail_cov_days)
        w_target_final = apply_vol_target(w_target_capped, cov, settings.target_ann_vol, max_leverage=3.0)

        # Get market data for the rebalance date
        rebal_idx = px.index.get_loc(rebal_date)
        prices_row = px.iloc[rebal_idx]
        adv_row = adv.iloc[rebal_idx] if rebal_idx < len(adv) else pd.Series(dtype=float)

        # 6) Calculate turnover and trading costs
        monthly_turnover = turnover(prev_w, w_target_final)
        trading_cost = rebalance_cost(
            equity, prev_w, w_target_final, prices_row, adv_row,
            settings.fixed_cost_bps, settings.impact_k
        )

        # Deduct trading costs from equity
        equity -= trading_cost
        logger.debug(".1f")

        # Store daily values at rebalance date
        daily_equity[rebal_date] = equity
        daily_turnover[rebal_date] = monthly_turnover
        daily_long_n[rebal_date] = len(longs)
        daily_short_n[rebal_date] = len(shorts)

        # 7) Hold positions until next rebalance or end of data
        next_rebal_idx = i + 1
        if next_rebal_idx < len(rebal_dates):
            next_rebal_date = rebal_dates[next_rebal_idx]
            # Find all dates between current rebalance and next rebalance
            date_mask = (px.index > rebal_date) & (px.index <= next_rebal_date)
            holding_dates = px.index[date_mask]

            for hold_date in holding_dates:
                # Calculate daily P&L
                hold_idx = px.index.get_loc(hold_date)
                daily_ret = (px.iloc[hold_idx][w_target_final.index] / px.iloc[hold_idx-1][w_target_final.index] - 1.0).fillna(0.0)
                portfolio_pnl = (w_target_final * daily_ret).sum()
                equity *= (1.0 + portfolio_pnl)

                # Store daily equity (turnover and counts are 0 on non-rebalance days)
                daily_equity[hold_date] = equity
                daily_turnover[hold_date] = 0.0
                daily_long_n[hold_date] = 0  # Will be forward-filled
                daily_short_n[hold_date] = 0

        # Update previous weights for next iteration
        prev_w = w_target_final

        logger.info(".0f")

    # Create performance DataFrame with daily index
    if not daily_equity:
        raise RuntimeError("No valid rebalances completed")

    all_dates = sorted(daily_equity.keys())
    equity_series = pd.Series(daily_equity).sort_index()
    turnover_series = pd.Series(daily_turnover).reindex(all_dates, fill_value=0.0)
    long_n_series = pd.Series(daily_long_n).reindex(all_dates, method='ffill', fill_value=0)
    short_n_series = pd.Series(daily_short_n).reindex(all_dates, method='ffill', fill_value=0)

    # Calculate daily returns
    ret_series = equity_series.pct_change().fillna(0.0)

    # Create performance DataFrame
    perf = pd.DataFrame({
        'equity': equity_series,
        'ret': ret_series,
        'turnover': turnover_series,
        'long_n': long_n_series,
        'short_n': short_n_series
    })

    # Split into IS/OOS periods
    is_end_date = pd.to_datetime(settings.is_end, utc=True)
    is_mask = perf.index <= is_end_date
    oos_mask = perf.index > is_end_date

    is_perf = perf[is_mask]
    oos_perf = perf[oos_mask]

    # Calculate metrics for each period
    def calc_metrics(perf_subset):
        if perf_subset.empty or len(perf_subset) < 2:
            return {'CAGR': np.nan, 'Vol': np.nan, 'Sharpe': np.nan, 'MDD': np.nan}
        return {
            'CAGR': annualized_return(perf_subset['equity']),
            'Vol': annualized_vol(perf_subset['ret']),
            'Sharpe': sharpe(perf_subset['ret']),
            'MDD': max_drawdown(perf_subset['equity'])
        }

    is_metrics = calc_metrics(is_perf)
    oos_metrics = calc_metrics(oos_perf)

    # Calculate benchmark Sharpe ratio
    # Find first date used in strategy
    first_strategy_date = perf.index[0]
    start_idx = px.index.get_loc(first_strategy_date)
    benchmark_equity = equal_weight_static(px, start_idx)

    # Align benchmark with strategy dates
    aligned_benchmark = benchmark_equity.reindex(perf.index, method='ffill')
    benchmark_returns = aligned_benchmark.pct_change().fillna(0.0)
    bnh_sharpe = sharpe(benchmark_returns)

    # Calculate average turnover (only on rebalance days)
    rebal_turnover = perf.loc[perf['turnover'] > 0, 'turnover']
    avg_turnover = rebal_turnover.mean() if not rebal_turnover.empty else np.nan

    logger.info(".0f")
    logger.info(".2f")

    return {
        'perf': perf,
        'metrics': {
            'is': is_metrics,
            'oos': oos_metrics,
            'bnh_sharpe': bnh_sharpe,
            'avg_turnover': avg_turnover
        }
    }


