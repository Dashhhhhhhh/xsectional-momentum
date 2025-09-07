from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from momentum_xsec.config import load_settings
from momentum_xsec.data.universe import build_universe, wide_prices
from momentum_xsec.signals.momentum import lookback_return, rank_deciles
from momentum_xsec.portfolio.construct import risk_scaled_weights, cap_weights_per_name


def main():
    load_dotenv()
    settings_path = Path('configs/settings.yaml')
    if not settings_path.exists():
        settings_path = Path('configs/settings.example.yaml')
    settings = load_settings(settings_path)

    api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
    df = build_universe(settings.universe, settings.start_date, api_key)
    px = wide_prices(df)

    asof = px.index[-1]
    signal = lookback_return(px, asof, settings.lookback_days, settings.skip_days)
    longs, shorts = rank_deciles(signal, settings.top_decile, settings.bottom_decile)

    ret = px.pct_change().fillna(0.0)
    w_long = cap_weights_per_name(risk_scaled_weights(longs, ret, asof, settings.trail_cov_days), settings.max_weight_per_name)
    w_short = -cap_weights_per_name(risk_scaled_weights(shorts, ret, asof, settings.trail_cov_days), settings.max_weight_per_name)
    target_w = (pd.concat([w_long, w_short]).groupby(level=0).sum()).sort_index()

    print('Target weights (last date):')
    print(target_w.to_string())


if __name__ == '__main__':
    main()


