from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils.logging import get_logger
from .costs import cost_bps_for_order

logger = get_logger(__name__)


class SimBroker:
    """Simple broker simulation for backtesting purposes.

    Tracks cash and positions, executes trades at next bar open with realistic costs.
    """

    def __init__(self, initial_cash: float = 1_000_000.0, positions: dict[str, float] | None = None):
        """Initialize broker with starting cash and positions.

        Args:
            initial_cash: Starting cash balance
            positions: Initial positions as dict of symbol -> shares
        """
        self.cash = float(initial_cash)
        self.positions = positions.copy() if positions else {}
        logger.info(".0f")

    def get_equity(self, prices_row: pd.Series) -> float:
        """Calculate current portfolio equity (cash + positions at current prices).

        Args:
            prices_row: Current prices for all symbols

        Returns:
            Total portfolio value
        """
        position_value = 0.0
        for symbol, shares in self.positions.items():
            price = prices_row.get(symbol)
            if price is not None and not np.isnan(price):
                position_value += shares * price

        total_equity = self.cash + position_value
        return total_equity

    def target_to_trades(self, prev_w: pd.Series, new_w: pd.Series, equity: float) -> dict[str, float]:
        """Calculate notional trade values needed to move from prev_w to new_w.

        Args:
            prev_w: Current portfolio weights
            new_w: Target portfolio weights
            equity: Current portfolio equity value

        Returns:
            Dict of symbol -> notional trade value (positive = buy, negative = sell)
        """
        if equity <= 0:
            logger.warning(f"Invalid equity value: {equity}")
            return {}

        # Align weight vectors
        idx = prev_w.index.union(new_w.index)
        prev_w_aligned = prev_w.reindex(idx).fillna(0.0)
        new_w_aligned = new_w.reindex(idx).fillna(0.0)

        # Calculate weight changes
        weight_changes = new_w_aligned - prev_w_aligned

        # Convert to notional values
        trades = (weight_changes * equity).to_dict()

        # Filter out zero trades
        trades = {k: v for k, v in trades.items() if abs(v) > 1e-6}

        logger.debug(f"Calculated {len(trades)} non-zero trades for equity ${equity:,.0f}")
        return trades

    def apply_trades(self, trades: dict[str, float], next_open_prices: pd.Series,
                    adv_row: pd.Series, fixed_bps: int, k: int) -> float:
        """Execute trades at next bar open prices with transaction costs.

        Args:
            trades: Dict of symbol -> notional trade value
            next_open_prices: Next day's open prices
            adv_row: Current ADV values
            fixed_bps: Fixed cost in basis points
            k: Square-root impact parameter

        Returns:
            Total transaction cost in dollars
        """
        total_cost = 0.0
        executed_trades = 0

        # Calculate portfolio notional for dust trade threshold
        portfolio_notional = sum(abs(v) for v in trades.values())

        for symbol, notional in trades.items():
            if abs(notional) < 1e-6:  # Skip effectively zero trades
                continue

            # Skip dust trades
            min_trade_size = max(50.0, 0.0005 * portfolio_notional)
            if abs(notional) < min_trade_size:
                logger.debug(".1f")
                continue

            # Get execution price
            exec_price = next_open_prices.get(symbol)
            if exec_price is None or np.isnan(exec_price) or exec_price <= 0:
                logger.warning(f"No valid execution price for {symbol}, skipping trade")
                continue

            # Calculate shares to trade
            shares_to_trade = notional / exec_price

            # Get ADV for cost calculation
            adv_value = adv_row.get(symbol)

            # Calculate transaction cost
            bps_cost = cost_bps_for_order(notional, adv_value, fixed_bps, k)
            dollar_cost = abs(notional) * (bps_cost / 10000.0)

            # Update cash (cost is deducted)
            self.cash -= dollar_cost

            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = 0.0

            self.positions[symbol] += shares_to_trade

            # Remove zero positions to keep clean
            if abs(self.positions[symbol]) < 1e-6:
                del self.positions[symbol]

            total_cost += dollar_cost
            executed_trades += 1

            logger.debug(".1f")

        logger.info(".1f")
        return total_cost

    def mark_to_market(self, prices_row: pd.Series) -> float:
        """Mark portfolio to market and return current equity value.

        Args:
            prices_row: Current prices for all symbols

        Returns:
            Current portfolio equity value
        """
        current_equity = self.get_equity(prices_row)
        logger.debug(".0f")
        return current_equity

    def get_positions_value(self, prices_row: pd.Series) -> dict[str, float]:
        """Get current position values at given prices.

        Args:
            prices_row: Current prices for all symbols

        Returns:
            Dict of symbol -> position value
        """
        position_values = {}
        for symbol, shares in self.positions.items():
            price = prices_row.get(symbol)
            if price is not None and not np.isnan(price):
                position_values[symbol] = shares * price
            else:
                position_values[symbol] = 0.0
        return position_values

    def __str__(self) -> str:
        """String representation of broker state."""
        return ".0f"

    def __repr__(self) -> str:
        """Detailed representation of broker state."""
        return ".0f"


