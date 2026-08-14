"""Paper broker and risk engine.

This is where a paper bot either tells the truth or quietly lies to you. The
four things almost every retail backtest omits, all modelled here:

  1. Fees on the *notional*, not the equity. At 100x a round trip costs 10% of
     a $100 account before the market does anything at all.
  2. Funding every 8 hours on the notional. At 100x that is roughly 3% of the
     account per day just to hold a position.
  3. Real liquidation, from real maintenance-margin rates. A 100x position dies
     at about 0.6% adverse, which BTC covers several times a day.
  4. Arriving late. The live engine wakes on a cron, so entries land minutes
     after the bar closed. The backtest pays the same penalty.

Margin model is isolated per position, cross-checked against free cash, which
is how a small account actually runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from . import config as C
from .config import Book

# Leverage is chosen so the stop is comfortably inside the liquidation band.
# At 0.5 the exchange needs price to travel twice as far as our stop before it
# takes the position away from us.
STOP_LIQ_SAFETY = 0.5


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class Position:
    symbol: str
    side: int
    module: str
    qty: float  # base units
    entry_price: float
    notional: float  # at entry
    leverage: float
    margin: float  # equity locked, lost entirely on liquidation
    stop: float
    target: float | None
    trail_atr: float | None
    liq_price: float
    opened_at: str
    bars_held: int = 0
    max_bars: int = 96
    extreme: float = 0.0  # best price seen, for the chandelier trail
    fees_paid: float = 0.0
    funding_paid: float = 0.0

    def unrealised(self, price: float) -> float:
        return self.side * (price - self.entry_price) * self.qty


@dataclass
class BookState:
    name: str
    label: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak_equity: float = 0.0
    day_key: str = ""
    day_start_equity: float = 0.0
    halted_until: str | None = None
    dead: bool = False
    n_trades: int = 0
    n_wins: int = 0
    n_liquidations: int = 0
    fees_total: float = 0.0
    funding_total: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym, pos.entry_price)
            total += pos.margin + pos.unrealised(px)
        return total

    def free_cash(self) -> float:
        return self.cash

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["positions"] = {k: asdict(v) for k, v in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BookState":
        positions = {k: Position(**v) for k, v in d.pop("positions", {}).items()}
        state = cls(**d)
        state.positions = positions
        return state


def new_book(book: Book, start_usd: float) -> BookState:
    return BookState(
        name=book.name,
        label=book.label,
        cash=start_usd,
        peak_equity=start_usd,
        day_start_equity=start_usd,
    )


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------


def canon(symbol: str) -> str:
    """BTC-USDT-SWAP -> BTCUSDT.

    The cost and leverage tables are keyed on Binance-style names, but the
    engine may be fed OKX instrument ids when Binance is unreachable. Without
    this every OKX symbol silently falls through to the defaults.
    """
    if symbol.endswith("-SWAP"):
        parts = symbol.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}{parts[1]}"
    return symbol


def half_spread(symbol: str) -> float:
    return C.HALF_SPREAD_BPS.get(canon(symbol), C.DEFAULT_HALF_SPREAD_BPS) / 1e4


def impact(notional: float) -> float:
    return (C.IMPACT_BPS_PER_10K / 1e4) * (notional / 10_000.0)


def entry_slippage(symbol: str, notional: float, atr_pct: float,
                   order_type: str) -> float:
    """Fraction of price given up on entry, always against us.

    Maker entries rest at a fixed limit price, so they cross no spread and take
    no delay penalty — they pay in fill risk instead, handled by the caller.
    """
    if order_type == "maker":
        return impact(notional)
    return (half_spread(symbol) + impact(notional)
            + C.DELAY_SLIPPAGE_FRACTION * atr_pct)


def exit_slippage(symbol: str, notional: float) -> float:
    """Exits are always market orders — stops and targets both take liquidity."""
    return half_spread(symbol) + impact(notional)


def max_leverage(symbol: str) -> float:
    return float(C.MAX_LEVERAGE.get(canon(symbol), C.DEFAULT_MAX_LEVERAGE))


def maintenance_margin(symbol: str) -> float:
    return C.MAINTENANCE_MARGIN.get(canon(symbol), C.DEFAULT_MAINTENANCE_MARGIN)


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


@dataclass
class Sizing:
    qty: float
    notional: float
    leverage: float
    margin: float
    stop_pct: float
    liq_pct: float
    rejected: str | None = None


def size_position(book: Book, state: BookState, symbol: str, price: float,
                  stop_dist: float, equity: float) -> Sizing:
    """Turn a signal's stop distance into a position, or refuse to.

    Risk-based books solve for notional from the stop, then pick the lowest
    leverage that still fits the margin, which keeps liquidation far away.
    The fixed-leverage book does the opposite: leverage is nailed to 100x and
    the stop has to squeeze inside whatever liquidation band that leaves.
    """
    mmr = maintenance_margin(symbol)
    lev_cap = max_leverage(symbol)
    stop_pct = stop_dist / price

    if stop_pct <= 0 or not math.isfinite(stop_pct):
        return Sizing(0, 0, 0, 0, 0, 0, "bad stop distance")

    if book.fixed_leverage is not None:
        # Exactly as specified: maximum leverage on every trade. On most alts
        # the exchange caps out well below 100x, and that cap is real.
        lev = min(book.fixed_leverage, lev_cap)
        liq_pct = (1.0 / lev) - mmr
        if liq_pct <= 0:
            return Sizing(0, 0, 0, 0, 0, 0, "leverage exceeds maintenance margin")
        # Fees are charged on notional, so at 100x they are 5% of the margin per
        # side. Committing every last dollar as margin leaves nothing to pay
        # them with and the order is rejected, so hold back a round trip.
        margin = state.free_cash() / (1 + 2 * lev * C.TAKER_FEE)
        if margin <= 0:
            return Sizing(0, 0, 0, 0, 0, 0, "no free cash")
        # The stop must sit inside liquidation or the exchange closes us first
        # and takes the entire margin instead of the intended loss.
        stop_pct = min(stop_pct, C.LIQ_STOP_FRACTION * liq_pct)
        notional = margin * lev
    else:
        risk_usd = equity * book.risk_per_trade
        notional = risk_usd / stop_pct

        # Lowest leverage that keeps the stop at half the liquidation distance.
        lev_for_safety = STOP_LIQ_SAFETY / (stop_pct + mmr)
        lev = min(lev_cap, max(1.0, lev_for_safety))
        margin = notional / lev

        if margin > state.free_cash():
            # Not enough free cash: scale the position down rather than skip it.
            margin = state.free_cash() / (1 + 2 * lev * C.TAKER_FEE)
            notional = margin * lev
            if notional <= 0:
                return Sizing(0, 0, 0, 0, 0, 0, "no free cash")

        liq_pct = (1.0 / lev) - mmr
        if liq_pct <= 0:
            return Sizing(0, 0, 0, 0, 0, 0, "leverage exceeds maintenance margin")
        if stop_pct >= liq_pct:
            stop_pct = C.LIQ_STOP_FRACTION * liq_pct

    qty = notional / price
    if qty <= 0 or notional < 5.0:  # below exchange minimum notional
        return Sizing(0, 0, 0, 0, 0, 0, "notional below exchange minimum")

    return Sizing(qty, notional, lev, margin, stop_pct, liq_pct)


# --------------------------------------------------------------------------
# Opening and closing
# --------------------------------------------------------------------------


def open_position(state: BookState, book: Book, signal, fill_price: float,
                  atr_pct: float, when: pd.Timestamp, prices: dict[str, float],
                  ) -> tuple[Position | None, str]:
    if state.dead or signal.symbol in state.positions:
        return None, "already in position"
    if len(state.positions) >= book.max_concurrent:
        return None, "max concurrent positions"

    equity = state.equity(prices)
    sizing = size_position(state=state, book=book, symbol=signal.symbol,
                           price=fill_price, stop_dist=signal.stop_dist,
                           equity=equity)
    if sizing.rejected:
        return None, sizing.rejected

    slip = entry_slippage(signal.symbol, sizing.notional, atr_pct,
                          signal.order_type)
    entry = fill_price * (1 + signal.side * slip)
    qty = sizing.notional / entry

    fee_rate = C.MAKER_FEE if signal.order_type == "maker" else C.TAKER_FEE
    fee = sizing.notional * fee_rate

    if fee + sizing.margin > state.cash:
        return None, "insufficient cash for margin plus fee"

    stop = entry * (1 - signal.side * sizing.stop_pct)
    liq = entry * (1 - signal.side * sizing.liq_pct)
    target = (entry + signal.side * signal.target_dist
              if signal.target_dist is not None else None)

    state.cash -= sizing.margin + fee
    state.fees_total += fee

    pos = Position(
        symbol=signal.symbol, side=signal.side, module=signal.module,
        qty=qty, entry_price=entry, notional=sizing.notional,
        leverage=sizing.leverage, margin=sizing.margin,
        stop=stop, target=target, trail_atr=signal.trail_atr,
        liq_price=liq, opened_at=str(when), max_bars=signal.max_bars,
        extreme=entry, fees_paid=fee,
    )
    state.positions[signal.symbol] = pos
    return pos, "opened"


def close_position(state: BookState, pos: Position, exit_price: float,
                   reason: str, when: pd.Timestamp, charge_fee: bool = True,
                   ) -> dict:
    """Settle a position back into cash and return a trade-log row."""
    notional_now = abs(pos.qty * exit_price)
    fee = notional_now * C.TAKER_FEE if charge_fee else 0.0

    if reason == "liquidated":
        # The exchange takes the whole isolated margin. Nothing comes back.
        pnl = -pos.margin
        state.cash += 0.0
        state.n_liquidations += 1
    else:
        gross = pos.unrealised(exit_price)
        pnl = gross - fee
        state.cash += pos.margin + gross - fee

    state.fees_total += fee
    state.n_trades += 1
    if pnl > 0:
        state.n_wins += 1

    state.positions.pop(pos.symbol, None)

    return {
        "time": str(when), "book": state.name, "symbol": pos.symbol,
        "module": pos.module, "side": "long" if pos.side > 0 else "short",
        "entry": round(pos.entry_price, 6), "exit": round(exit_price, 6),
        "qty": pos.qty, "notional": round(pos.notional, 2),
        "leverage": round(pos.leverage, 1), "margin": round(pos.margin, 4),
        "pnl": round(pnl, 4), "fees": round(pos.fees_paid + fee, 4),
        "funding": round(pos.funding_paid, 4), "bars": pos.bars_held,
        "reason": reason, "cash_after": round(state.cash, 4),
    }


# --------------------------------------------------------------------------
# Per-bar position management
# --------------------------------------------------------------------------


def apply_funding(state: BookState, pos: Position, rate: float,
                  price: float) -> None:
    """Longs pay shorts when the rate is positive. Charged on notional.

    Under isolated margin a shortfall in free cash comes out of the position's
    own margin, and less margin means liquidation sits closer. That feedback is
    the high-leverage death spiral in miniature: funding erodes the buffer, the
    liquidation price walks toward the entry, and a smaller move finishes it.
    """
    cost = pos.side * rate * abs(pos.qty * price)
    state.cash -= cost
    pos.funding_paid += cost
    state.funding_total += cost

    if state.cash < 0:
        pos.margin += state.cash  # cash is negative; this is the shortfall
        state.cash = 0.0

        mmr = maintenance_margin(pos.symbol)
        if pos.margin <= 0 or pos.notional <= 0:
            pos.liq_price = pos.entry_price  # nothing left to absorb a move
        else:
            liq_pct = (pos.margin / pos.notional) - mmr
            pos.liq_price = pos.entry_price * (1 - pos.side * max(liq_pct, 0.0))


def step_position(state: BookState, pos: Position, bar: pd.Series,
                  atr_now: float, when: pd.Timestamp) -> dict | None:
    """Walk one position through one bar.

    Checks run adverse-first: liquidation, then stop, then target. When a bar's
    range covers both the stop and the target there is no way to know which
    came first without tick data, so we assume the bad one did.
    """
    from .strategy import update_trail

    pos.bars_held += 1
    high, low, open_px = float(bar["high"]), float(bar["low"]), float(bar["open"])
    adverse = low if pos.side > 0 else high
    favourable = high if pos.side > 0 else low

    def beyond(price: float, level: float) -> bool:
        return price <= level if pos.side > 0 else price >= level

    # The stop always sits nearer than the liquidation price, so within a bar
    # price has to cross the stop first. Only a gap can skip it — checking
    # liquidation first regardless would hand every high-leverage trade a total
    # loss when it should have been stopped out for a fraction of that.
    if beyond(open_px, pos.liq_price):
        return close_position(state, pos, pos.liq_price, "liquidated", when,
                              charge_fee=False)

    if beyond(adverse, pos.stop):
        gapped = beyond(open_px, pos.stop)
        raw = open_px if gapped else pos.stop
        if beyond(raw, pos.liq_price):
            return close_position(state, pos, pos.liq_price, "liquidated", when,
                                  charge_fee=False)
        fill = raw * (1 - pos.side * exit_slippage(pos.symbol, pos.notional))
        return close_position(state, pos, fill, "stop", when)

    if pos.target is not None:
        hit = (favourable >= pos.target) if pos.side > 0 else (favourable <= pos.target)
        if hit:
            fill = pos.target * (1 - pos.side * exit_slippage(pos.symbol, pos.notional))
            return close_position(state, pos, fill, "target", when)

    if pos.trail_atr is not None and atr_now > 0:
        pos.extreme = (max(pos.extreme, favourable) if pos.side > 0
                       else min(pos.extreme, favourable))
        pos.stop = update_trail(pos.side, pos.extreme, atr_now,
                                pos.trail_atr, pos.stop)

    if pos.bars_held >= pos.max_bars:
        close_px = float(bar["close"])
        fill = close_px * (1 - pos.side * exit_slippage(pos.symbol, pos.notional))
        return close_position(state, pos, fill, "time stop", when)

    return None


# --------------------------------------------------------------------------
# Book-level guards
# --------------------------------------------------------------------------


def roll_day(state: BookState, book: Book, when: pd.Timestamp,
             prices: dict[str, float]) -> None:
    key = when.strftime("%Y-%m-%d")
    if state.day_key != key:
        state.day_key = key
        state.day_start_equity = state.equity(prices)
        state.halted_until = None


def check_guards(state: BookState, book: Book, when: pd.Timestamp,
                 prices: dict[str, float]) -> None:
    """Daily loss limit and the terminal kill switch."""
    equity = state.equity(prices)
    state.peak_equity = max(state.peak_equity, equity)

    if equity <= book.dead_below and not state.positions:
        state.dead = True
        return

    if state.day_start_equity > 0:
        drop = 1 - equity / state.day_start_equity
        if drop >= book.daily_loss_limit:
            state.halted_until = str(when.normalize() + pd.Timedelta(days=1))


def can_trade(state: BookState, when: pd.Timestamp) -> bool:
    if state.dead:
        return False
    if state.halted_until and when < pd.Timestamp(state.halted_until):
        return False
    return True
