"""The live paper run: one cron tick, or a long-lived loop.

A tick is idempotent and catch-up safe. It processes every bar that closed
since the last run, so a missed cron window (or a whole day of them) costs
accuracy on entry timing but never corrupts the ledger.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

import pandas as pd

from . import backtest, config as C, data, features
from .engine import Portfolio

LOOKBACK_BARS = 1400  # covers the 300-bar bandwidth window and the 200 EMA


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def load_state() -> dict:
    if not os.path.exists(C.STATE_FILE):
        return {}
    with open(C.STATE_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state: dict) -> None:
    os.makedirs(C.STATE_DIR, exist_ok=True)
    tmp = C.STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)
    os.replace(tmp, C.STATE_FILE)


def _append_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def initialise() -> dict:
    """First run: freeze the FX rate and pick the starting universe."""
    fx = data.aud_usd()
    start_usd = round(C.START_EQUITY_AUD * fx, 2)
    universe = data.screen_universe()
    now = data.utcnow()
    print(f"init: A${C.START_EQUITY_AUD:.2f} = US${start_usd:.2f} @ {fx:.4f}")
    print(f"init: universe {universe}")
    return {
        "created_at": str(now),
        "aud_usd": fx,
        "start_usd": start_usd,
        "universe": universe,
        "last_screen": str(now),
        "last_bar": None,
        "portfolio": None,
        "ticks": 0,
    }


# --------------------------------------------------------------------------
# One tick
# --------------------------------------------------------------------------


def tick(verbose: bool = True) -> dict:
    state = load_state() or initialise()
    now = data.utcnow()

    pf = Portfolio(state["start_usd"], C.PARAMS)
    if state.get("portfolio"):
        pf.load_dict(state["portfolio"])

    held = {sym for st in pf.state.values() for sym in st.positions}
    universe = list(dict.fromkeys(state["universe"]))

    # Weekly re-screen. Symbols with open positions stay loaded until flat, so
    # a rotation can never orphan a live position.
    last_screen = pd.Timestamp(state.get("last_screen") or now)
    if now - last_screen >= timedelta(days=C.SCREEN_EVERY_DAYS):
        try:
            fresh = data.screen_universe()
            dropped = [s for s in universe if s not in fresh and s in held]
            universe = list(dict.fromkeys(fresh + dropped))
            state["universe"] = fresh
            state["last_screen"] = str(now)
            if verbose:
                print(f"re-screened universe -> {fresh}"
                      + (f" (holding {dropped})" if dropped else ""))
        except Exception as exc:
            print(f"screen failed, keeping universe: {exc}")
    universe = list(dict.fromkeys(universe + sorted(held)))

    start = now - timedelta(hours=LOOKBACK_BARS)
    feats: dict[str, pd.DataFrame] = {}
    tilts: dict[str, pd.Series] = {}
    settles: dict[str, pd.Series] = {}

    for sym in universe:
        try:
            raw = data.load_klines(sym, C.TIMEFRAME, start.strftime("%Y-%m-%d"))
        except Exception as exc:
            print(f"  {sym}: kline fetch failed ({exc})")
            continue
        if len(raw) < 400:
            continue
        feats[sym] = features.build(raw.tail(LOOKBACK_BARS), C.PARAMS)
        try:
            fr = data.load_funding(sym, start.strftime("%Y-%m-%d"))
        except Exception:
            fr = pd.Series(dtype=float)
        tilts[sym] = fr
        settles[sym] = fr

    if not feats:
        raise RuntimeError("no market data available this tick")

    index = None
    for df in feats.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()

    aligned = {}
    for sym, df in feats.items():
        d = df.reindex(index)
        tilt = (tilts[sym].reindex(index, method="ffill").fillna(0.0)
                if len(tilts[sym]) else pd.Series(0.0, index=index))
        settle = (settles[sym].reindex(index)
                  if len(settles[sym]) else pd.Series(float("nan"), index=index))
        aligned[sym] = {"feat": d, "tilt": tilt, "settle": settle}

    last_bar = pd.Timestamp(state["last_bar"]) if state.get("last_bar") else None
    todo = index[index > last_bar] if last_bar is not None else index[-3:]

    before = len(pf.trades)
    if len(todo):
        vol = backtest._vol_flags(aligned, index)
        sub = {
            sym: {
                "feat": blob["feat"].loc[todo],
                "tilt": blob["tilt"].loc[todo],
                "settle": blob["settle"].loc[todo],
            }
            for sym, blob in aligned.items()
        }
        subvol = {k: v.loc[todo] for k, v in vol.items()}
        # Reuse the backtest loop so live and historical results come from
        # identical code, continuing this portfolio rather than a new one.
        backtest.run(sub, todo, state["start_usd"], C.PARAMS,
                     vol_cache=subvol, portfolio=pf)
        state["last_bar"] = str(todo[-1])

    # Execute anything queued by the newest closed bar against the bar that is
    # forming right now — this is the trade a person would have placed.
    forming_bars, forming_feats = {}, {}
    for sym in feats:
        try:
            fb = data.forming_bar(sym)
        except Exception:
            fb = None
        if fb is None:
            continue
        forming_bars[sym] = fb
        forming_feats[sym] = aligned[sym]["feat"].iloc[-1]

    if forming_bars and any(pf.pending.values()):
        pf.fill_pending(now, forming_bars, forming_feats)

    prices = {}
    for sym in feats:
        if sym in forming_bars:
            prices[sym] = float(forming_bars[sym]["close"])
        else:
            c = aligned[sym]["feat"]["close"].dropna()
            if len(c):
                prices[sym] = float(c.iloc[-1])

    new_trades = pf.trades[before:]
    _append_csv(C.TRADE_LOG, new_trades)

    equity_row = {"time": str(now)}
    for name, st in pf.state.items():
        equity_row[name] = round(st.equity(prices), 4)
    equity_row["aud_rate"] = state["aud_usd"]
    _append_csv(C.EQUITY_LOG, [equity_row])

    state["portfolio"] = pf.to_dict()
    state["ticks"] = state.get("ticks", 0) + 1
    state["updated_at"] = str(now)
    state["last_prices"] = prices
    state["summary"] = pf.summary(prices)
    save_state(state)

    if verbose:
        _print_tick(state, pf, prices, new_trades)
    return state


def _print_tick(state: dict, pf: Portfolio, prices: dict, new_trades: list) -> None:
    fx = state["aud_usd"]
    print(f"\n{state['updated_at']}  tick #{state['ticks']}")
    print(f"universe: {', '.join(state['universe'])}")
    for t in new_trades:
        arrow = "OPEN " if t["reason"].startswith("OPEN") else f"{t['reason']:<10}"
        print(f"  [{t['book']:<15}] {arrow} {t['side']:<5} {t['symbol']:<12}"
              f" @ {t['entry']}  pnl={t['pnl']}")
    print(f"{'book':<20}{'USD':>10}{'AUD':>10}{'ret%':>8}{'trades':>8}{'open':>6}")
    for row in state["summary"]:
        aud = row["equity"] / fx
        print(f"{row['book']:<20}{row['equity']:>10.2f}{aud:>10.2f}"
              f"{row['return_pct']:>8.1f}{row['trades']:>8}{row['open']:>6}"
              + ("  DEAD" if row["dead"] else ""))


def daemon(interval_seconds: int = 300) -> None:
    """Always-on mode for a VM. Same tick, just on a timer."""
    import time as _t
    while True:
        try:
            tick()
        except Exception as exc:  # a bad tick must never end the month-long run
            print(f"tick error: {type(exc).__name__}: {exc}")
        _t.sleep(interval_seconds)
