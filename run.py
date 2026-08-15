#!/usr/bin/env python
"""CLI entry point.

    python run.py screen                 # what the universe screener picks today
    python run.py backtest               # full history, all four books
    python run.py montecarlo             # P(double) and P(ruin) per 30-day window
    python run.py once                   # one live paper tick (used by cron)
    python run.py daemon                 # always-on loop for a VM
    python run.py report                 # rebuild the HTML dashboard
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from bot import config as C


def cmd_screen(args):
    from bot import data
    universe = data.screen_universe()
    print("universe:", ", ".join(universe))


def _market(args):
    from bot import backtest, data
    symbols = args.symbols.split(",") if args.symbols else None
    if symbols is None:
        try:
            symbols = data.screen_universe()
        except Exception as exc:
            print(f"screen failed ({exc}); falling back to core symbols")
            symbols = list(C.CORE_SYMBOLS)
    print(f"symbols: {', '.join(symbols)}")
    print("loading market data (cached after the first run)...")
    return backtest.load_market(symbols, args.start)


def cmd_backtest(args):
    from bot import backtest
    index, aligned = _market(args)
    print(f"{len(index)} hourly bars, "
          f"{index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d}\n")

    pf = backtest.run(aligned, index, args.equity, progress=True)
    table = backtest.stats(pf)

    pd.set_option("display.width", 200)
    print("\n=== full sample ===")
    print(table.to_string(index=False))

    half = len(index) // 2
    for label, sl in (("first half", index[:half]), ("second half", index[half:])):
        sub = {s: {k: v.loc[sl] for k, v in b.items()} for s, b in aligned.items()}
        sub_pf = backtest.run(sub, sl, args.equity)
        print(f"\n=== {label} ({sl[0]:%Y-%m-%d} to {sl[-1]:%Y-%m-%d}) ===")
        print(backtest.stats(sub_pf).to_string(index=False))

    trades = pd.DataFrame(pf.trades)
    if not trades.empty:
        trades.to_csv("reports/backtest_trades.csv", index=False)
        print(f"\n{len(trades)} trade records -> reports/backtest_trades.csv")
    table.to_csv("reports/backtest_stats.csv", index=False)


def cmd_montecarlo(args):
    from bot import backtest
    index, aligned = _market(args)
    print(f"\nrolling {args.window}-day windows, one every {args.step} days...")
    mc = backtest.monte_carlo_30d(aligned, index, args.equity,
                                  window_days=args.window, step_days=args.step)
    summary = backtest.summarise_monte_carlo(mc, C.BOOKS)

    pd.set_option("display.width", 200)
    print(f"\n=== every {args.window}-day window, each starting fresh at "
          f"${args.equity:.0f} ===")
    print(summary.to_string(index=False))

    mc.to_csv("reports/montecarlo_windows.csv", index=False)
    summary.to_csv("reports/montecarlo_summary.csv", index=False)
    print("\n-> reports/montecarlo_summary.csv")


def cmd_once(args):
    from bot import live
    live.tick()


def cmd_daemon(args):
    from bot import live
    live.daemon(args.interval)


def cmd_report(args):
    from bot import report
    print(f"-> {report.build()}")
    print(f"-> {report.build_status_md()}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--symbols", default=None,
                        help="comma-separated, e.g. BTCUSDT,SOLUSDT")
        sp.add_argument("--start", default=C.BACKTEST_START)
        sp.add_argument("--equity", type=float, default=66.0,
                        help="starting USD (A$100 is roughly US$66)")

    sp = sub.add_parser("screen"); sp.set_defaults(func=cmd_screen)
    sp = sub.add_parser("backtest"); common(sp); sp.set_defaults(func=cmd_backtest)
    sp = sub.add_parser("montecarlo"); common(sp)
    sp.add_argument("--window", type=int, default=30)
    sp.add_argument("--step", type=int, default=5)
    sp.set_defaults(func=cmd_montecarlo)
    sp = sub.add_parser("once"); sp.set_defaults(func=cmd_once)
    sp = sub.add_parser("daemon")
    sp.add_argument("--interval", type=int, default=300)
    sp.set_defaults(func=cmd_daemon)
    sp = sub.add_parser("report"); sp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
