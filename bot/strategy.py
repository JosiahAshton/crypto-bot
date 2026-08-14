"""The regime router.

Two modules that make money in opposite conditions, and a classifier that
decides which one is allowed to speak on any given bar:

  trend regime -> Donchian breakout, chandelier trail, ~35% win rate, big Rs
  chop  regime -> VWAP mean reversion, ~65% win rate, small Rs
  neutral      -> stand down

On top of that, a funding-rate tilt. When perp funding is extremely positive
the long side is crowded and paying to stay there, so new longs are blocked.
It is free from the exchange and almost nobody wires it into a retail bot.

Signals carry *distances*, not prices. The broker applies them to the real
fill, so slippage never silently widens or tightens the risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyParams
from .features import tradeable

MAX_CHASE_ATR = 1.0  # never enter more than this far beyond the broken level
COOLDOWN_BARS = 6  # bars to wait after an exit before re-entering the same symbol


@dataclass
class Signal:
    symbol: str
    side: int  # +1 long, -1 short
    module: str  # "breakout" | "meanrev"
    ref_price: float  # close of the signal bar, for logging only
    stop_dist: float  # absolute price distance from fill to stop
    target_dist: float | None  # absolute distance to take-profit, None = trail only
    trail_atr: float | None  # chandelier multiple, None = no trail
    atr: float
    max_bars: int
    order_type: str  # "taker" (market) | "maker" (resting limit)
    reason: str


def _funding_allows(side: int, funding: float, p: StrategyParams) -> bool:
    """Block trades that join an already-crowded and expensive side."""
    if side > 0 and funding > p.funding_block:
        return False
    if side < 0 and funding < -p.funding_block:
        return False
    return True


def _breakout(row: pd.Series, symbol: str, p: StrategyParams,
              vol_expanding: bool) -> Signal | None:
    if row["regime"] != "trend" or not vol_expanding:
        return None

    close, a = row["close"], row["atr"]
    dc_high, dc_low = row["dc_high"], row["dc_low"]
    if not np.isfinite(dc_high) or not np.isfinite(dc_low) or a <= 0:
        return None

    above = close > row["ema_trend"] if p.use_ema_filter else True
    below = close < row["ema_trend"] if p.use_ema_filter else True

    side = 0
    if close > dc_high and above:
        side, extension = 1, (close - dc_high) / a
    elif close < dc_low and below:
        side, extension = -1, (dc_low - close) / a
    else:
        return None

    # Do not chase. Entering three ATRs past the level is how breakout systems
    # turn a 2.5R winner into a 0.4R one.
    if extension > MAX_CHASE_ATR:
        return None

    return Signal(
        symbol=symbol,
        side=side,
        module="breakout",
        ref_price=close,
        stop_dist=p.breakout_stop_atr * a,
        target_dist=None,
        trail_atr=p.breakout_trail_atr,
        atr=a,
        max_bars=p.breakout_max_bars,
        order_type="taker",
        reason=f"donchian break er={row['er']:.2f} ext={extension:.2f}atr",
    )


def _mean_reversion(row: pd.Series, symbol: str, p: StrategyParams) -> Signal | None:
    if row["regime"] != "chop":
        return None

    close, a, z = row["close"], row["atr"], row["z"]
    vwap = row["vwap"]
    if not np.isfinite(vwap) or a <= 0 or not np.isfinite(z):
        return None

    if z <= -p.z_entry:
        side = 1
    elif z >= p.z_entry:
        side = -1
    else:
        return None

    # A fixed R target, not the distance back to the mean. The sweep tested
    # fixed multiples, so this is the rule that was actually validated —
    # substituting a distance-to-VWAP target here would ship something the
    # research never measured.
    stop_dist = p.mr_stop_atr * a
    target_dist = p.mr_target_r * stop_dist

    if target_dist / stop_dist < p.min_reward_risk:
        return None

    return Signal(
        symbol=symbol,
        side=side,
        module="meanrev",
        ref_price=close,
        stop_dist=stop_dist,
        target_dist=target_dist,
        trail_atr=None,
        atr=a,
        max_bars=p.mr_max_bars,
        order_type="maker",
        reason=f"z={z:.2f} bw={row['bw_pct']:.0f}pct rr={target_dist/stop_dist:.2f}",
    )


def generate(row: pd.Series, symbol: str, funding: float, p: StrategyParams,
             vol_expanding: bool) -> Signal | None:
    """Evaluate one closed bar for one symbol. Returns at most one signal."""
    if not tradeable(row, p):
        return None

    for candidate in (_breakout(row, symbol, p, vol_expanding),
                      _mean_reversion(row, symbol, p)):
        if candidate is None:
            continue
        if not _funding_allows(candidate.side, funding, p):
            continue
        return candidate
    return None


def vol_expanding(feat: pd.DataFrame, i: int, window: int = 50) -> bool:
    """True when current ATR sits above its own recent median.

    Breakouts that fire while volatility is contracting are usually noise
    poking through a stale channel.
    """
    if i < window:
        return False
    recent = feat["atr"].iloc[i - window: i + 1]
    med = float(recent.median())
    cur = float(feat["atr"].iloc[i])
    return np.isfinite(med) and med > 0 and cur > med


def update_trail(side: int, extreme: float, atr_now: float,
                 trail_atr: float, current_stop: float) -> float:
    """Chandelier stop: ratchet toward price, never away from it."""
    proposed = (extreme - trail_atr * atr_now if side > 0
                else extreme + trail_atr * atr_now)
    return max(current_stop, proposed) if side > 0 else min(current_stop, proposed)
