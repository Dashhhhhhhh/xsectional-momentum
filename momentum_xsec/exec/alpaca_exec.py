from __future__ import annotations
from typing import Dict
import pandas as pd


def get_account_equity(api) -> float:
    acct = api.get_account()
    return float(acct.equity)


def get_positions(api) -> Dict[str, float]:
    pos = api.list_positions()
    return {p.symbol: float(p.market_value) for p in pos}


def submit_target_weights(api, target_w: pd.Series, equity_notional: float, tif: str = 'day') -> None:
    for sym, w in target_w.items():
        notional = equity_notional * float(w)
        if abs(notional) < max(50.0, 0.0005 * equity_notional):
            continue
        side = 'buy' if notional > 0 else 'sell'
        api.submit_order(symbol=sym, notional=abs(notional), side=side, type='market', time_in_force=tif)


