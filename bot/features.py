"""Indicators and regime features.

Every column here describes the bar it sits on, using only that bar and earlier
ones. The strategy layer is responsible for shifting before acting, so that
lookahead has exactly one place it could creep in and it is easy to audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyParams


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman efficiency ratio: net move over gross path.

    Near 1.0 the market is travelling in a straight line (trend); near 0 it is
    churning (chop). More robust than ADX for regime switching because it has
    no smoothing lag baked in.
    """
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (direction / volatility.replace(0, np.nan)).fillna(0.0)


def bandwidth_percentile(close: pd.Series, period: int, lookback: int) -> pd.Series:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    bandwidth = (4 * sd) / mid.replace(0, np.nan)
    return bandwidth.rolling(lookback, min_periods=period * 2).rank(pct=True) * 100


def rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).rolling(period).sum()
    v = df["volume"].rolling(period).sum()
    return pv / v.replace(0, np.nan)


def vwap_zscore(df: pd.DataFrame, period: int) -> pd.Series:
    vwap = rolling_vwap(df, period)
    spread = df["close"] - vwap
    sd = spread.rolling(period).std(ddof=0)
    return (spread / sd.replace(0, np.nan)).fillna(0.0)


def build(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Attach every feature the strategy needs to a raw OHLCV frame."""
    out = df.copy()

    out["atr"] = atr(out, p.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]

    out["ema_trend"] = out["close"].ewm(span=p.trend_ema, adjust=False).mean()
    out["er"] = efficiency_ratio(out["close"], p.efficiency_period)
    out["bw_pct"] = bandwidth_percentile(
        out["close"], p.bandwidth_period, p.bandwidth_lookback
    )

    # Donchian channel excludes the current bar, so a "breakout" means breaking
    # a level that was already established before this bar started printing.
    out["dc_high"] = out["high"].rolling(p.donchian_period).max().shift(1)
    out["dc_low"] = out["low"].rolling(p.donchian_period).min().shift(1)

    out["vwap"] = rolling_vwap(out, p.vwap_period)
    out["z"] = vwap_zscore(out, p.vwap_period)

    out["regime"] = np.where(
        out["er"] >= p.trending_threshold, "trend",
        np.where(
            (out["er"] <= p.chop_threshold) & (out["bw_pct"] <= p.compressed_pct),
            "chop", "neutral",
        ),
    )

    return out


def tradeable(row: pd.Series, p: StrategyParams) -> bool:
    """Refuse dead markets and unhinged ones alike."""
    if not np.isfinite(row.get("atr_pct", np.nan)):
        return False
    return p.min_atr_pct <= row["atr_pct"] <= p.max_atr_pct
