from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from momentum_xsec.utils.logging import get_logger
from momentum_xsec.config import load_settings
from momentum_xsec.data.universe import build_universe, wide_prices, wide_volume
from momentum_xsec.backtest.engine import run_backtest

logger = get_logger(__name__)


def format_metric(value: float, fmt: str = '.1%') -> str:
    """Format metric value, handling NaN cases."""
    if pd.isna(value):
        return 'NaN'
    if fmt == '.1%':
        return f'{value:.1%}'
    elif fmt == '.2f':
        return f'{value:.2f}'
    else:
        return str(value)


def main():
    """Run the cross-sectional momentum backtest and generate outputs."""

    # Load configuration
    load_dotenv()
    settings_path = Path('configs/settings.yaml')
    if not settings_path.exists():
        settings_path = Path('configs/settings.example.yaml')
        logger.info(f"Using example settings: {settings_path}")

    settings = load_settings(settings_path)
    logger.info(f"Loaded settings for universe of {len(settings.universe)} symbols")

    # Validate API key
    api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
    if not api_key or api_key == 'REPLACE_ME':
        raise ValueError("Please set ALPHAVANTAGE_API_KEY in .env file")

    # Fetch and prepare data
    logger.info("Fetching data from Alpha Vantage...")
    df = build_universe(settings.universe, settings.start_date, api_key)
    px = wide_prices(df)
    vol = wide_volume(df)

    logger.info(f"Data prepared: {px.shape[0]} days × {px.shape[1]} symbols")

    # Run backtest
    logger.info("Running backtest...")
    result = run_backtest(px, vol, settings)

    # Extract results
    perf = result['perf']
    metrics = result['metrics']

    # Create output directory
    outdir = Path('out')
    outdir.mkdir(parents=True, exist_ok=True)

    # Print key metrics as requested
    print("\n" + "="*60)
    print("CROSS-SECTIONAL MOMENTUM BACKTEST RESULTS")
    print("="*60)

    # Main metrics table
    is_metrics = metrics['is']
    oos_metrics = metrics['oos']
    bnh_sharpe = metrics['bnh_sharpe']
    avg_turnover = metrics['avg_turnover']

    print("\nPERFORMANCE METRICS:")
    print("                  IS        OOS       B&H")
    print("-" * 40)
    print("Sharpe:      {:>8}  {:>8}  {:>8}".format(
        format_metric(is_metrics['Sharpe'], '.2f'),
        format_metric(oos_metrics['Sharpe'], '.2f'),
        format_metric(bnh_sharpe, '.2f')
    ))
    print("CAGR:        {:>8}  {:>8}".format(
        format_metric(is_metrics['CAGR']),
        format_metric(oos_metrics['CAGR'])
    ))
    print("Volatility:  {:>8}  {:>8}".format(
        format_metric(is_metrics['Vol']),
        format_metric(oos_metrics['Vol'])
    ))
    print("Max DD:      {:>8}  {:>8}".format(
        format_metric(is_metrics['MDD']),
        format_metric(oos_metrics['MDD'])
    ))

    print("\nAvg Monthly Turnover: {}".format(format_metric(avg_turnover)))

    # Done-when flags
    print("\nDONE-WHEN FLAGS:")
    print("-" * 30)

    # OOS Sharpe > B&H Sharpe
    oos_sharpe = oos_metrics['Sharpe']
    oos_vs_bnh = "PASS" if (not pd.isna(oos_sharpe) and not pd.isna(bnh_sharpe) and
                           oos_sharpe > bnh_sharpe) else "FAIL"
    print("OOS Sharpe > B&H Sharpe: {}".format(oos_vs_bnh))

    # Avg turnover <= 0.6 (60%)
    turnover_limit = 0.6  # Could be made configurable
    turnover_flag = "PASS" if (not pd.isna(avg_turnover) and avg_turnover <= turnover_limit) else "FAIL"
    print("Avg Turnover <= {:.0%}: {}".format(turnover_limit, turnover_flag))

    # Slippage modeled
    slippage_flag = "PASS" if (settings.fixed_cost_bps > 0 or settings.impact_k > 0) else "FAIL"
    print("Slippage Modeled: {}".format(slippage_flag))

    # Overall assessment
    flags_passed = sum([
        oos_vs_bnh == "PASS",
        turnover_flag == "PASS",
        slippage_flag == "PASS"
    ])
    print("\nOverall: {}/3 flags passed".format(flags_passed))

    # Save artifacts
    logger.info("Saving artifacts to ./out/...")

    # Save performance data
    perf.to_csv(outdir / 'xsec_perf.csv')
    logger.info("Saved xsec_perf.csv")

    # Save summary text file
    with open(outdir / 'summary.txt', 'w') as f:
        f.write("CROSS-SECTIONAL MOMENTUM BACKTEST SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write("IS Sharpe: {}\n".format(format_metric(is_metrics['Sharpe'], '.2f')))
        f.write("OOS Sharpe: {}\n".format(format_metric(oos_metrics['Sharpe'], '.2f')))
        f.write("B&H Sharpe: {}\n".format(format_metric(bnh_sharpe, '.2f')))
        f.write("IS CAGR: {}\n".format(format_metric(is_metrics['CAGR'])))
        f.write("OOS CAGR: {}\n".format(format_metric(oos_metrics['CAGR'])))
        f.write("IS Volatility: {}\n".format(format_metric(is_metrics['Vol'])))
        f.write("OOS Volatility: {}\n".format(format_metric(oos_metrics['Vol'])))
        f.write("IS Max DD: {}\n".format(format_metric(is_metrics['MDD'])))
        f.write("OOS Max DD: {}\n".format(format_metric(oos_metrics['MDD'])))
        f.write("Avg Turnover: {}\n".format(format_metric(avg_turnover)))

        f.write("\nDONE-WHEN FLAGS:\n")
        f.write("-" * 20 + "\n")
        f.write("OOS Sharpe > B&H Sharpe: {}\n".format(oos_vs_bnh))
        f.write("Avg Turnover <= {:.0%}: {}\n".format(turnover_limit, turnover_flag))
        f.write("Slippage Modeled: {}\n".format(slippage_flag))
        f.write("Overall: {}/3 flags passed\n".format(flags_passed))

        f.write("\nBACKTEST CONFIGURATION:\n")
        f.write("-" * 25 + "\n")
        f.write("Universe Size: {}\n".format(len(settings.universe)))
        f.write("Lookback Days: {}\n".format(settings.lookback_days))
        f.write("Skip Days: {}\n".format(settings.skip_days))
        f.write("Top Decile: {:.1%}\n".format(settings.top_decile))
        f.write("Bottom Decile: {:.1%}\n".format(settings.bottom_decile))
        f.write("Target Vol: {:.1%}\n".format(settings.target_ann_vol))
        f.write("Max Weight: {:.1%}\n".format(settings.max_weight_per_name))
        f.write("Fixed Cost: {} bps\n".format(settings.fixed_cost_bps))
        f.write("Impact K: {}\n".format(settings.impact_k))
        f.write("IS End Date: {}\n".format(settings.is_end))

    logger.info("Saved summary.txt")

    # Create equity curve chart
    fig, ax = plt.subplots(figsize=(12, 6))
    perf['equity'].plot(ax=ax, linewidth=2, color='blue')
    ax.set_title('Equity Curve — XSec Momentum (Stocks)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Format y-axis as currency
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    plt.tight_layout()
    fig.savefig(outdir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved equity_curve.png")

    # Create drawdown chart
    eq = perf['equity']
    dd = eq / eq.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(12, 4))
    dd.plot(ax=ax, linewidth=2, color='red')
    ax.set_title('Drawdown — XSec Momentum (Stocks)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    fig.savefig(outdir / 'drawdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved drawdown.png")

    print("\n✅ Backtest complete! Artifacts saved to ./out/")
    print("   - xsec_perf.csv (performance data)")
    print("   - summary.txt (metrics and flags)")
    print("   - equity_curve.png (equity chart)")
    print("   - drawdown.png (drawdown chart)")

    logger.info("Backtest script completed successfully")


if __name__ == '__main__':
    main()


