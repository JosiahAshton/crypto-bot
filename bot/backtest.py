"""Backtest, walk-forward, and the Monte Carlo that answers the actual question.

The headline number is not CAGR. It is this: across every 30-day window in the
sample, how often did a $100 account actually double, and how often did it die?
That is the question being asked, so that is what gets measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, data, features
from .config import StrategyParams
from .engine import Portfolio


# --------------------------------------------------------------------------
# Market assembly
# --------------------------------------------------------------------------


def load_market(symbols: list[str], start: str = C.BACKTEST_START,
                params: StrategyParams = C.PARAMS, refresh: bool = True):
    """Fetch, cache and feature-ise every symbol onto one shared hourly clock."""
    feats: dict[str, pd.DataFrame] = {}
    funding_tilt: dict[str, pd.Series] = {}
    funding_settle: dict[str, pd.Series] = {}

    for sym in symbols:
        raw = data.load_klines(sym, C.TIMEFRAME, start, refresh=refresh)
        if raw.empty or len(raw) < 500:
            print(f"  skip {sym}: only {len(raw)} bars")
            continue
        feats[sym] = features.build(raw, params)
        try:
            fr = data.load_funding(sym, start)
        except Exception:
            fr = pd.Series(dtype=float)
        funding_tilt[sym] = fr
        funding_settle[sym] = fr

    if not feats:
        raise RuntimeError("no symbols loaded")

    index = None
    for df in feats.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()

    aligned = {}
    for sym, df in feats.items():
        d = df.reindex(index)
        tilt = (funding_tilt[sym].reindex(index, method="ffill").fillna(0.0)
                if len(funding_tilt[sym]) else pd.Series(0.0, index=index))
        settle = (funding_settle[sym].reindex(index)
                  if len(funding_settle[sym]) else pd.Series(np.nan, index=index))
        aligned[sym] = {"feat": d, "tilt": tilt, "settle": settle}

    return index, aligned


def _vol_flags(aligned: dict, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Precompute the ATR-above-its-own-median test for every bar."""
    out = {}
    for sym, blob in aligned.items():
        a = blob["feat"]["atr"]
        med = a.rolling(50).median()
        out[sym] = (a > med) & a.notna() & med.notna()
    return out


# --------------------------------------------------------------------------
# Core run
# --------------------------------------------------------------------------


def run(aligned: dict, index: pd.DatetimeIndex, start_usd: float,
        params: StrategyParams = C.PARAMS,
        vol_cache: dict[str, pd.Series] | None = None,
        progress: bool = False, books: list | None = None,
        portfolio: Portfolio | None = None) -> Portfolio:
    # The live engine passes an existing portfolio so a tick *continues* the
    # month instead of starting a fresh one. Without this the ledger silently
    # resets to the opening balance on every run.
    pf = portfolio if portfolio is not None else Portfolio(start_usd, params,
                                                           books=books)
    vol = vol_cache if vol_cache is not None else _vol_flags(aligned, index)

    cols = ["open", "high", "low", "close", "atr", "atr_pct", "er", "bw_pct",
            "dc_high", "dc_low", "ema_trend", "vwap", "z", "regime"]
    frames = {s: b["feat"][cols] for s, b in aligned.items()}

    total = len(index)
    for i, when in enumerate(index):
        bars, feats_now, fund_tilt, vflags = {}, {}, {}, {}
        settle = False

        for sym, df in frames.items():
            row = df.loc[when]
            if not np.isfinite(row["close"]):
                continue
            bars[sym] = row
            feats_now[sym] = row
            fund_tilt[sym] = float(aligned[sym]["tilt"].loc[when])
            vflags[sym] = bool(vol[sym].loc[when])
            if np.isfinite(aligned[sym]["settle"].loc[when]):
                settle = True

        if not bars:
            continue

        fund_now = fund_tilt
        if settle:
            fund_now = {
                s: (float(aligned[s]["settle"].loc[when])
                    if np.isfinite(aligned[s]["settle"].loc[when])
                    else 0.0)
                for s in bars
            }

        pf.step(when, bars, feats_now, fund_now, vflags, settle)

        if progress and i % 2000 == 0:
            print(f"  {i}/{total} {when:%Y-%m-%d}")

    return pf


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def stats(pf: Portfolio) -> pd.DataFrame:
    eq = pd.DataFrame(pf.equity_curve)
    if eq.empty:
        return pd.DataFrame()
    eq["time"] = pd.to_datetime(eq["time"], utc=True)
    eq = eq.set_index("time")

    trades = pd.DataFrame(pf.trades)
    closed = pd.DataFrame()
    if not trades.empty:
        closed = trades[trades["pnl"] != ""].copy()
        if not closed.empty:
            closed["pnl"] = closed["pnl"].astype(float)

    rows = []
    hours = max((eq.index[-1] - eq.index[0]).total_seconds() / 3600, 1)
    years = hours / 8760

    for name, st in pf.state.items():
        curve = eq[name]
        final = float(curve.iloc[-1])
        peak = curve.cummax()
        dd = float((1 - curve / peak).max() * 100)
        rets = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        sharpe = (float(rets.mean() / rets.std() * np.sqrt(8760))
                  if len(rets) > 10 and rets.std() > 0 else 0.0)
        cagr = ((final / pf.start_usd) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        bt = closed[closed["book"] == name] if not closed.empty else pd.DataFrame()
        wins = bt[bt["pnl"] > 0]["pnl"] if not bt.empty else pd.Series(dtype=float)
        losses = bt[bt["pnl"] <= 0]["pnl"] if not bt.empty else pd.Series(dtype=float)
        pf_ratio = (wins.sum() / abs(losses.sum())
                    if len(losses) and losses.sum() != 0 else np.nan)

        rows.append({
            "book": st.label,
            "final": round(final, 2),
            "total_%": round((final / pf.start_usd - 1) * 100, 1),
            "CAGR_%": round(cagr, 1),
            "maxDD_%": round(dd, 1),
            "Sharpe": round(sharpe, 2),
            "trades": st.n_trades,
            "win_%": round(100 * st.n_wins / st.n_trades, 1) if st.n_trades else 0,
            "profit_factor": round(pf_ratio, 2) if np.isfinite(pf_ratio) else None,
            "liquidations": st.n_liquidations,
            "fees_$": round(st.fees_total, 2),
            "funding_$": round(st.funding_total, 2),
            "dead": st.dead,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The 30-day question
# --------------------------------------------------------------------------


def monte_carlo_30d(aligned: dict, index: pd.DatetimeIndex, start_usd: float,
                    params: StrategyParams = C.PARAMS, window_days: int = 30,
                    step_days: int = 5, warmup_bars: int = 400,
                    books: list | None = None) -> pd.DataFrame:
    """Start a fresh $100 account on every Nth day and see where it lands.

    Each window is an independent life for the account, which is exactly the
    experiment being run live: one month, one starting balance, no do-overs.
    """
    vol = _vol_flags(aligned, index)
    window_bars = window_days * 24
    step_bars = step_days * 24
    starts = range(warmup_bars, len(index) - window_bars, step_bars)

    results: list[dict] = []
    for n, s in enumerate(starts):
        sl = index[s: s + window_bars]
        sub = {
            sym: {
                "feat": blob["feat"].loc[sl],
                "tilt": blob["tilt"].loc[sl],
                "settle": blob["settle"].loc[sl],
            }
            for sym, blob in aligned.items()
        }
        subvol = {k: v.loc[sl] for k, v in vol.items()}
        pf = run(sub, sl, start_usd, params, vol_cache=subvol, books=books)

        prices = {}
        for sym, blob in sub.items():
            c = blob["feat"]["close"].dropna()
            if len(c):
                prices[sym] = float(c.iloc[-1])

        row = {"start": sl[0].strftime("%Y-%m-%d")}
        for name, st in pf.state.items():
            row[name] = round(st.equity(prices) / start_usd, 4)
        results.append(row)

        if n % 20 == 0:
            print(f"  window {n + 1}/{len(list(starts))} from {row['start']}")

    return pd.DataFrame(results)


def summarise_monte_carlo(mc: pd.DataFrame, books) -> pd.DataFrame:
    rows = []
    for b in books:
        if b.name not in mc.columns:
            continue
        m = mc[b.name]
        rows.append({
            "book": b.label,
            "windows": len(m),
            "median_x": round(float(m.median()), 2),
            "mean_x": round(float(m.mean()), 2),
            "P(double)_%": round(float((m >= 2.0).mean() * 100), 1),
            "P(profit)_%": round(float((m > 1.0).mean() * 100), 1),
            "P(-50%)_%": round(float((m <= 0.5).mean() * 100), 1),
            "P(ruin)_%": round(float((m <= 0.10).mean() * 100), 1),
            "best_x": round(float(m.max()), 2),
            "worst_x": round(float(m.min()), 3),
        })
    return pd.DataFrame(rows)
