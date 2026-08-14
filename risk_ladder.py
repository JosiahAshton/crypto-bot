#!/usr/bin/env python
"""Which risk-per-trade actually maximises the chance of doubling in 30 days?

The four shipped books show P(double) rising from 0% at 2% risk to ~12% at 20%,
then collapsing to 0% at 100x. That shape has a peak in it. This finds it by
running the same 30-day Monte Carlo across a ladder of risk settings, all fed by
one signal engine so the trades are identical and only the sizing differs.

    python risk_ladder.py
"""

from __future__ import annotations

import pandas as pd

from bot import backtest, config as C
from bot.config import Book

SYMBOLS = ["BTCUSDT", "SOLUSDT", "ZECUSDT", "HYPEUSDT", "WLDUSDT", "1000PEPEUSDT"]
RISKS = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.28, 0.40, 0.55]


def ladder() -> list[Book]:
    books = [
        Book(f"r{int(r * 100):03d}", f"{int(r * 100)}% risk",
             risk_per_trade=r, fixed_leverage=None,
             max_concurrent=3, daily_loss_limit=1.0)
        for r in RISKS
    ]
    books.append(Book("lev100", "100x fixed", risk_per_trade=None,
                      fixed_leverage=100.0, max_concurrent=1,
                      daily_loss_limit=1.0))
    return books


def main():
    books = ladder()
    print(f"symbols: {', '.join(SYMBOLS)}")
    index, aligned = backtest.load_market(SYMBOLS, "2023-01-01", refresh=False)
    print(f"{len(index)} bars, {index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d}")
    print(f"ladder: {[b.label for b in books]}\n")

    mc = backtest.monte_carlo_30d(aligned, index, 66.0, window_days=30,
                                  step_days=5, books=books)
    summary = backtest.summarise_monte_carlo(mc, books)

    pd.set_option("display.width", 220)
    print("\n=== 30-day outcomes by risk per trade (255 windows, $66 start) ===")
    print(summary.to_string(index=False))

    best = summary.loc[summary["P(double)_%"].idxmax()]
    print(f"\npeak chance of doubling: {best['book']} "
          f"at {best['P(double)_%']}% of months "
          f"(median {best['median_x']}x, ruin {best['P(ruin)_%']}%)")

    summary.to_csv("reports/risk_ladder.csv", index=False)
    mc.to_csv("reports/risk_ladder_windows.csv", index=False)
    print("-> reports/risk_ladder.csv")


if __name__ == "__main__":
    main()
