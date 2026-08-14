#!/usr/bin/env python
"""Signal-level research: measure each module's edge in R, net of costs.

The portfolio backtest answers "what would the account have done". This answers
the prior question — "is there an edge at all" — by stripping out position
sizing and expressing every trade as a multiple of the risk taken. Costs are
converted into R too, because a 0.1% round trip is trivial against a 3% stop
and fatal against a 0.15% one.

    python research.py breakout
    python research.py meanrev
    python research.py confirm      # best-of settings, per symbol and per year
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot import config as C, data, features

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "ZECUSDT", "HYPEUSDT", "WLDUSDT", "1000PEPEUSDT"]
START = "2023-01-01"


# --------------------------------------------------------------------------
# Exit management
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitPlan:
    stop_atr: float = 2.0
    breakeven_at_R: float | None = None
    partial_at_R: float | None = None
    partial_frac: float = 0.5
    trail_atr: float | None = None
    trail_after_R: float = 0.0
    target_R: float | None = None
    max_bars: int = 96


def simulate(feat: pd.DataFrame, sides: np.ndarray, plan: ExitPlan,
             symbol: str, order_type: str, funding: np.ndarray) -> pd.DataFrame:
    """Walk every signal forward and return its net R multiple.

    One position at a time per symbol, entered at the next bar's open, exactly
    as the live engine does.
    """
    o = feat["open"].to_numpy(float)
    h = feat["high"].to_numpy(float)
    lo = feat["low"].to_numpy(float)
    cl = feat["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float)
    n = len(feat)

    hs = C.HALF_SPREAD_BPS.get(symbol, C.DEFAULT_HALF_SPREAD_BPS) / 1e4
    entry_fee = C.MAKER_FEE if order_type == "maker" else C.TAKER_FEE
    rows: list[dict] = []

    # Only bars that actually carry a signal are worth visiting, and a new
    # trade cannot start before the previous one has closed.
    candidates = np.nonzero(sides)[0]
    busy_until = -1

    for i in candidates:
        i = int(i)
        if i <= busy_until or i >= n - 2:
            continue
        s = int(sides[i])
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        e = i + 1
        ref = o[e]
        if not np.isfinite(ref) or ref <= 0:
            continue

        risk = plan.stop_atr * atr[i]
        stop_pct = risk / ref
        if stop_pct <= 0:
            continue

        atr_pct = atr[i] / ref
        entry_slip = (0.0 if order_type == "maker"
                      else hs + C.DELAY_SLIPPAGE_FRACTION * atr_pct)
        entry = ref * (1 + s * entry_slip)

        stop = entry - s * risk
        extreme = entry
        target = entry + s * plan.target_R * risk if plan.target_R else None
        remaining, realised, n_exits = 1.0, 0.0, 1
        partial_done = be_done = False
        funding_cost = 0.0
        exit_R, exit_j, why = 0.0, min(e + plan.max_bars, n - 1), "time"

        for j in range(e, min(e + plan.max_bars, n)):
            funding_cost += s * funding[j] * remaining

            adverse = lo[j] if s > 0 else h[j]
            favourable = h[j] if s > 0 else lo[j]

            if (adverse <= stop) if s > 0 else (adverse >= stop):
                exit_R, exit_j, why = s * (stop - entry) / risk, j, "stop"
                break

            if target is not None and ((favourable >= target) if s > 0
                                       else (favourable <= target)):
                exit_R, exit_j, why = s * (target - entry) / risk, j, "target"
                break

            fav_R = s * (favourable - entry) / risk

            if plan.partial_at_R and not partial_done and fav_R >= plan.partial_at_R:
                realised += plan.partial_frac * plan.partial_at_R
                remaining -= plan.partial_frac
                partial_done = True
                n_exits += 1
                stop = max(stop, entry) if s > 0 else min(stop, entry)

            if plan.breakeven_at_R and not be_done and fav_R >= plan.breakeven_at_R:
                stop = max(stop, entry) if s > 0 else min(stop, entry)
                be_done = True

            if plan.trail_atr and fav_R >= plan.trail_after_R and np.isfinite(atr[j]):
                extreme = max(extreme, favourable) if s > 0 else min(extreme, favourable)
                proposed = extreme - s * plan.trail_atr * atr[j]
                stop = max(stop, proposed) if s > 0 else min(stop, proposed)
        else:
            exit_R = s * (cl[exit_j] - entry) / risk

        exit_cost = (hs + C.TAKER_FEE) * n_exits
        cost_R = (entry_slip + entry_fee + exit_cost) / stop_pct
        gross = realised + remaining * exit_R
        net = gross - cost_R - funding_cost / stop_pct

        rows.append({"i": i, "side": s, "gross_R": gross, "cost_R": cost_R,
                     "R": net, "bars": exit_j - e + 1, "why": why})
        busy_until = exit_j

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Entry rules
# --------------------------------------------------------------------------


def donchian(feat: pd.DataFrame, period: int):
    return (feat["high"].rolling(period).max().shift(1).to_numpy(float),
            feat["low"].rolling(period).min().shift(1).to_numpy(float))


def breakout_sides(feat: pd.DataFrame, period: int, er_min: float,
                   chase_max: float, use_ema: bool, need_vol_exp: bool) -> np.ndarray:
    hi, lo = donchian(feat, period)
    c = feat["close"].to_numpy(float)
    atr = feat["atr"].to_numpy(float)
    er = feat["er"].to_numpy(float)
    ema = feat["ema_trend"].to_numpy(float)
    med = feat["atr"].rolling(50).median().to_numpy(float)

    ok = (er >= er_min)
    if need_vol_exp:
        ok &= np.isfinite(med) & (atr > med)
    if use_ema:
        long_ok, short_ok = ok & (c > ema), ok & (c < ema)
    else:
        long_ok = short_ok = ok

    with np.errstate(invalid="ignore"):
        up = long_ok & (c > hi) & (((c - hi) / atr) <= chase_max)
        dn = short_ok & (c < lo) & (((lo - c) / atr) <= chase_max)
    return np.where(up, 1, np.where(dn, -1, 0)).astype(int)


def meanrev_sides(feat: pd.DataFrame, z_entry: float, er_max: float,
                  bw_max: float | None) -> np.ndarray:
    z = feat["z"].to_numpy(float)
    er = feat["er"].to_numpy(float)
    bw = feat["bw_pct"].to_numpy(float)
    ok = er <= er_max
    if bw_max is not None:
        ok &= np.isfinite(bw) & (bw <= bw_max)
    return np.where(ok & (z <= -z_entry), 1,
                    np.where(ok & (z >= z_entry), -1, 0)).astype(int)


# --------------------------------------------------------------------------
# Market loading
# --------------------------------------------------------------------------


def load(symbols=SYMBOLS, params=C.PARAMS):
    out = {}
    for sym in symbols:
        try:
            raw = data.load_klines(sym, "1h", START, refresh=False)
        except Exception:
            continue
        if len(raw) < 3000:
            continue
        feat = features.build(raw, params)
        try:
            fr = data.load_funding(sym, START)
            farr = fr.reindex(feat.index).fillna(0.0).to_numpy(float)
        except Exception:
            farr = np.zeros(len(feat))
        out[sym] = (feat, farr)
    return out


def collect(market, side_fn, plan, order_type) -> pd.DataFrame:
    """Every trade the rule would have taken, across every symbol, timestamped."""
    frames = []
    for sym, (feat, farr) in market.items():
        sides = side_fn(feat)
        res = simulate(feat, sides, plan, sym, order_type, farr)
        if not res.empty:
            res["symbol"] = sym
            res["time"] = feat.index[res["i"].to_numpy()]
            frames.append(res)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def agg(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "grossR": 0.0, "meanR": 0.0, "totalR": 0.0,
                "R_per_month": 0.0, "win": 0.0, "pf": 0.0, "cost_R": 0.0,
                "bars": 0.0}
    wins = trades[trades["R"] > 0]["R"]
    losses = trades[trades["R"] <= 0]["R"]
    span = (trades["time"].max() - trades["time"].min()).days / 30.44
    return {
        "n": len(trades),
        # Gross tells you whether the signal has any edge at all; net tells you
        # whether it survives contact with the fee schedule. When gross is
        # positive and net is not, the strategy is fine and the costs are not.
        "grossR": round(float(trades["gross_R"].mean()), 3),
        "meanR": round(float(trades["R"].mean()), 3),
        "totalR": round(float(trades["R"].sum()), 1),
        "R_per_month": round(float(trades["R"].sum()) / max(span, 1), 2),
        "win": round(float((trades["R"] > 0).mean() * 100), 1),
        "pf": round(float(wins.sum() / abs(losses.sum())) if losses.sum() else 99, 2),
        "cost_R": round(float(trades["cost_R"].mean()), 3),
        "bars": round(float(trades["bars"].mean()), 1),
    }


def score(market, side_fn, plan, order_type) -> dict:
    return agg(collect(market, side_fn, plan, order_type))


def show(rows: list[dict], keys: list[str], top: int = 18):
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results")
        return df
    df = df.sort_values("R_per_month", ascending=False)
    pd.set_option("display.width", 220)
    print(df[keys + ["n", "win", "grossR", "meanR", "totalR", "R_per_month",
                     "pf", "cost_R", "bars"]].head(top).to_string(index=False))
    return df


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def sweep_breakout(market):
    rows = []
    grid = itertools.product(
        [48, 96, 168, 240],          # donchian period
        [0.30, 0.38, 0.46],          # efficiency ratio floor
        [1.5, 2.0, 3.0],             # stop, in ATR
        [None, 2.0, 3.0],            # chandelier trail
        [None, 1.0],                 # partial take-profit, in R
        [True, False],               # EMA200 direction filter
    )
    for period, er, stop, trail, partial, ema in grid:
        plan = ExitPlan(stop_atr=stop, trail_atr=trail, trail_after_R=0.0,
                        partial_at_R=partial, partial_frac=0.5,
                        breakeven_at_R=1.0 if partial is None else None,
                        max_bars=96)
        s = score(market,
                  lambda f, p=period, e=er, m=ema: breakout_sides(f, p, e, 1.0, m, True),
                  plan, "taker")
        s.pop("_trades", None)
        rows.append({"dc": period, "er": er, "stop": stop, "trail": trail,
                     "partial": partial, "ema": ema, **s})
    return show(rows, ["dc", "er", "stop", "trail", "partial", "ema"])


def sweep_meanrev(market):
    rows = []
    grid = itertools.product(
        [2.0, 2.5, 3.0],             # z entry
        [0.22, 0.30, 0.40, 1.01],    # efficiency ratio ceiling (1.01 = no filter)
        [None, 35.0, 60.0],          # bandwidth percentile ceiling
        [1.0, 1.5, 2.0],             # stop, in ATR
        [1.0, 1.5, 2.0],             # target, in R
        [12, 24],                    # max bars
    )
    for z, er, bw, stop, target, mb in grid:
        plan = ExitPlan(stop_atr=stop, target_R=target, max_bars=mb)
        s = score(market,
                  lambda f, z=z, e=er, b=bw: meanrev_sides(f, z, e, b),
                  plan, "maker")
        s.pop("_trades", None)
        rows.append({"z": z, "er_max": er, "bw_max": bw, "stop": stop,
                     "target": target, "max_bars": mb, **s})
    return show(rows, ["z", "er_max", "bw_max", "stop", "target", "max_bars"])


SPLIT = pd.Timestamp("2025-01-01", tz="UTC")


def confirm(market):
    """Refined grid, chosen in-sample, then judged out-of-sample.

    The first sweep tested 216 combinations, so its winner is part edge and part
    luck. The only honest referee is data that took no part in the choosing:
    settings are ranked on 2023-2024 and then reported on 2025-2026, untouched.
    """
    settled = dict(trail_atr=None, partial_at_R=None, breakeven_at_R=None)
    rows = []

    grid = itertools.product(
        [120, 168, 240, 336],    # donchian period
        [0.46, 0.52, 0.58],      # efficiency ratio floor
        [2.5, 3.0, 4.0],         # stop, in ATR
        [96, 168, 240],          # time stop, in bars
    )
    for period, er, stop, mb in grid:
        plan = ExitPlan(stop_atr=stop, max_bars=mb, **settled)
        trades = collect(
            market,
            lambda f, p=period, e=er: breakout_sides(f, p, e, 1.0, False, True),
            plan, "taker")
        if trades.empty:
            continue
        is_t = trades[trades["time"] < SPLIT]
        oos = trades[trades["time"] >= SPLIT]
        a, b = agg(is_t), agg(oos)
        rows.append({
            "dc": period, "er": er, "stop": stop, "bars": mb,
            "IS_n": a["n"], "IS_R/mo": a["R_per_month"], "IS_mean": a["meanR"],
            "IS_pf": a["pf"],
            "OOS_n": b["n"], "OOS_R/mo": b["R_per_month"], "OOS_mean": b["meanR"],
            "OOS_pf": b["pf"], "OOS_gross": b["grossR"],
        })

    df = pd.DataFrame(rows).sort_values("IS_R/mo", ascending=False)
    pd.set_option("display.width", 240)
    print("=== ranked on 2023-2024, then shown on 2025-2026 ===")
    print(df.head(12).to_string(index=False))

    print("\n=== how the in-sample top 10 actually did out-of-sample ===")
    top = df.head(10)
    print(f"  in-sample  R/month: median {top['IS_R/mo'].median():.2f}")
    print(f"  out-of-sample R/month: median {top['OOS_R/mo'].median():.2f}"
          f"   positive in {int((top['OOS_R/mo'] > 0).sum())}/10")
    print(f"  full grid out-of-sample positive in "
          f"{int((df['OOS_R/mo'] > 0).sum())}/{len(df)} configs")

    best = df.iloc[0]
    plan = ExitPlan(stop_atr=float(best["stop"]), max_bars=int(best["bars"]),
                    **settled)
    trades = collect(
        market,
        lambda f: breakout_sides(f, int(best["dc"]), float(best["er"]),
                                 1.0, False, True),
        plan, "taker")

    print(f"\n=== best config (dc={int(best['dc'])} er={best['er']} "
          f"stop={best['stop']} bars={int(best['bars'])}) by symbol ===")
    per = trades.groupby("symbol").apply(
        lambda g: pd.Series(agg(g)), include_groups=False)
    print(per[["n", "win", "grossR", "meanR", "totalR", "pf"]].to_string())

    print("\n=== same config, by year ===")
    trades["year"] = trades["time"].dt.year
    per_y = trades.groupby("year").apply(
        lambda g: pd.Series(agg(g)), include_groups=False)
    print(per_y[["n", "win", "grossR", "meanR", "totalR", "pf"]].to_string())
    return df


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "breakout"
    print("loading cached market data...")
    market = load()
    span = {s: f"{f.index[0]:%Y-%m}->{f.index[-1]:%Y-%m} ({len(f)})"
            for s, (f, _) in market.items()}
    for k, v in span.items():
        print(f"  {k:<14} {v}")
    print()

    if what == "breakout":
        sweep_breakout(market)
    elif what == "meanrev":
        sweep_meanrev(market)
    elif what == "confirm":
        confirm(market)
    else:
        print(f"unknown sweep: {what}")


if __name__ == "__main__":
    main()
