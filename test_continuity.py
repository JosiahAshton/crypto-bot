#!/usr/bin/env python
"""The invariant the whole month-long run rests on.

A live tick processes a handful of new bars, writes the portfolio to JSON, and
exits. The next tick reloads that JSON and carries on. If any of that loses
state, the ledger quietly resets and thirty days of results mean nothing — and
it would look completely normal while doing it.

So: run a market once end to end, then run the identical market in chunks with
a full JSON round trip between every chunk, and require the two to finish
byte-identical.

    python test_continuity.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from bot import backtest, config as C, features
from bot.engine import Portfolio

BARS = 3000
SYMBOLS = ["BTCUSDT", "SOLUSDT", "ZECUSDT"]


def synthetic(seed: int, n: int = BARS) -> pd.DataFrame:
    """A random walk with volatility clustering, so both modules get signals."""
    rng = np.random.default_rng(seed)
    vol = 0.004 * (1 + 0.8 * np.sin(np.arange(n) / 220.0) ** 2)
    steps = rng.normal(0, 1, n) * vol
    # A couple of sustained drifts so the breakout module has trends to catch.
    steps[600:900] += 0.0016
    steps[1800:2100] -= 0.0014
    close = 100 * np.exp(np.cumsum(steps))

    spread = close * vol * 1.5
    high = close + np.abs(rng.normal(0, 1, n)) * spread
    low = close - np.abs(rng.normal(0, 1, n)) * spread
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(1e3, 1e4, n)},
                        index=idx)


def build_market():
    aligned = {}
    for i, sym in enumerate(SYMBOLS):
        feat = features.build(synthetic(seed=i + 1), C.PARAMS)
        idx = feat.index
        # A funding rate on every 8-hour boundary, as the exchange settles it.
        settle = pd.Series(np.nan, index=idx)
        hours = idx.hour.isin(C.FUNDING_HOURS)
        settle[hours] = 0.0001
        aligned[sym] = {"feat": feat, "tilt": pd.Series(0.0, index=idx),
                        "settle": settle}
    return aligned, list(aligned[SYMBOLS[0]]["feat"].index)


def fingerprint(pf: Portfolio) -> dict:
    return {
        "books": {
            name: {
                "cash": round(st.cash, 8),
                "trades": st.n_trades,
                "wins": st.n_wins,
                "liquidations": st.n_liquidations,
                "fees": round(st.fees_total, 8),
                "funding": round(st.funding_total, 8),
                "dead": st.dead,
                "positions": sorted(st.positions),
            }
            for name, st in pf.state.items()
        },
        "n_trade_rows": len(pf.trades),
    }


def main() -> int:
    aligned, index = build_market()
    index = pd.DatetimeIndex(index)
    print(f"{len(index)} synthetic bars across {len(SYMBOLS)} symbols")

    # One continuous run.
    whole = backtest.run(aligned, index, 66.0, C.PARAMS)

    # The same market in chunks, persisted to JSON and reloaded between each,
    # exactly as consecutive cron ticks would.
    chunked = Portfolio(66.0, C.PARAMS)
    bounds = [0, 900, 1400, 2300, len(index)]
    for a, b in zip(bounds, bounds[1:]):
        sl = index[a:b]
        sub = {s: {k: v.loc[sl] for k, v in blob.items()}
               for s, blob in aligned.items()}
        backtest.run(sub, sl, 66.0, C.PARAMS, portfolio=chunked)

        blob = json.loads(json.dumps(chunked.to_dict(), default=str))
        revived = Portfolio(66.0, C.PARAMS)
        revived.load_dict(blob)
        revived.trades = chunked.trades
        chunked = revived

    a, b = fingerprint(whole), fingerprint(chunked)
    ok = a == b

    # Free cash is a balance, not an overdraft. Anything below zero means a
    # cost was charged that the book could not actually pay.
    negative = {n: s["cash"] for n, s in a["books"].items() if s["cash"] < -1e-9}
    if negative:
        ok = False
        print(f"\nnegative free cash: {negative}")

    print(f"\n{'book':<20}{'continuous':>14}{'chunked+JSON':>16}")
    for name in a["books"]:
        ca = a["books"][name]["cash"]
        cb = b["books"][name]["cash"]
        flag = "" if abs(ca - cb) < 1e-6 else "   <-- MISMATCH"
        print(f"{name:<20}{ca:>14.4f}{cb:>16.4f}{flag}")

    if ok:
        print("\nPASS: chunked execution with JSON round trips is identical "
              "to one continuous run")
        return 0

    print("\nFAIL: state does not survive the tick boundary")
    print(json.dumps({"continuous": a, "chunked": b}, indent=2)[:2500])
    return 1


if __name__ == "__main__":
    sys.exit(main())
