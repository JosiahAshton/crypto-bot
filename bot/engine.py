"""The bar loop, shared by the backtest and the live run.

Backtest and live trading go through exactly the same code here. That is
deliberate: the most common way a paper bot flatters itself is by running a
tidy vectorised backtest and a completely different live path, so the two never
have to agree. Here they cannot disagree.

Order of operations on each bar, chosen so nothing can peek forward:

  1. fill orders that were queued on the *previous* closed bar
  2. settle funding, if this bar crosses a settlement hour
  3. walk open positions through this bar (liquidation, stop, target, trail)
  4. apply the daily loss limit and kill switch
  5. read this bar's features and queue orders for the *next* bar
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from . import broker, config as C, strategy
from .config import BOOKS, StrategyParams
from .strategy import Signal


def _signal_to_dict(s: Signal) -> dict:
    return asdict(s)


def _signal_from_dict(d: dict) -> Signal:
    return Signal(**d)


class Portfolio:
    """All four books, one signal engine, one clock."""

    def __init__(self, start_usd: float, params: StrategyParams = C.PARAMS,
                 books: list | None = None):
        self.params = params
        books = list(books) if books is not None else list(BOOKS)
        self.books = {b.name: b for b in books}
        self.state = {b.name: broker.new_book(b, start_usd) for b in books}
        self.pending: dict[str, list[Signal]] = {b.name: [] for b in books}
        self.cooldown: dict[tuple[str, str], int] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.start_usd = start_usd

    # -- one bar ---------------------------------------------------------

    def step(self, when: pd.Timestamp, bars: dict[str, pd.Series],
             feats: dict[str, pd.Series], funding: dict[str, float],
             vol_flags: dict[str, bool], settle_funding: bool) -> None:
        prices = {s: float(b["close"]) for s, b in bars.items()}

        for name, book in self.books.items():
            st = self.state[name]
            if st.dead:
                continue

            broker.roll_day(st, book, when, prices)
            self._fill_pending(st, book, bars, feats, when, prices)

            if settle_funding:
                for sym, pos in list(st.positions.items()):
                    if sym in prices:
                        broker.apply_funding(st, pos, funding.get(sym, 0.0),
                                             prices[sym])

            for sym, pos in list(st.positions.items()):
                if sym not in bars:
                    continue
                atr_now = float(feats[sym].get("atr", 0.0) or 0.0)
                row = broker.step_position(st, pos, bars[sym], atr_now, when)
                if row:
                    self.trades.append(row)
                    self.cooldown[(name, sym)] = strategy.COOLDOWN_BARS

            broker.check_guards(st, book, when, prices)

        for key in list(self.cooldown):
            self.cooldown[key] -= 1
            if self.cooldown[key] <= 0:
                del self.cooldown[key]

        self._queue_signals(when, feats, funding, vol_flags, prices)
        self._record_equity(when, prices)

    def fill_pending(self, when: pd.Timestamp, bars: dict[str, pd.Series],
                     feats: dict[str, pd.Series]) -> None:
        """Execute queued orders against a bar that has not closed yet.

        Live, the cron wakes minutes after a bar closes, and the order it queued
        belongs at the *next* bar's open — which is the bar currently forming.
        Its open price is already fixed and its running range already tells us
        whether a resting limit has traded, so this is the same fill the
        backtest models, taken at the same point in the sequence.
        """
        prices = {s: float(b["close"]) for s, b in bars.items()}
        for name, book in self.books.items():
            st = self.state[name]
            if st.dead:
                continue
            broker.roll_day(st, book, when, prices)
            self._fill_pending(st, book, bars, feats, when, prices)

    # -- execution -------------------------------------------------------

    def _fill_pending(self, st: broker.BookState, book, bars, feats,
                      when: pd.Timestamp, prices: dict[str, float]) -> None:
        queued, self.pending[st.name] = self.pending[st.name], []

        for sig in queued:
            if sig.symbol not in bars:
                continue
            if not broker.can_trade(st, when):
                continue

            bar = bars[sig.symbol]
            fill = self._try_fill(sig, bar)
            if fill is None:
                continue  # resting limit never traded; the order simply expires

            atr_pct = float(feats[sig.symbol].get("atr_pct", 0.0) or 0.0)
            pos, _ = broker.open_position(st, book, sig, fill, atr_pct, when,
                                          prices)
            if pos is not None:
                self.trades.append({
                    "time": str(when), "book": st.name, "symbol": sig.symbol,
                    "module": sig.module,
                    "side": "long" if sig.side > 0 else "short",
                    "entry": round(pos.entry_price, 6), "exit": "",
                    "qty": pos.qty, "notional": round(pos.notional, 2),
                    "leverage": round(pos.leverage, 1),
                    "margin": round(pos.margin, 4), "pnl": "",
                    "fees": round(pos.fees_paid, 4), "funding": 0.0,
                    "bars": 0, "reason": f"OPEN {sig.reason}",
                    "cash_after": round(st.cash, 4),
                })

    @staticmethod
    def _try_fill(sig: Signal, bar: pd.Series) -> float | None:
        """Market orders take the open. Resting limits need price to come to them."""
        open_px = float(bar["open"])
        if sig.order_type == "taker":
            return open_px

        limit = sig.ref_price
        if sig.side > 0:
            if open_px <= limit:
                return open_px  # gapped through — we get the better price
            return limit if float(bar["low"]) <= limit else None
        if open_px >= limit:
            return open_px
        return limit if float(bar["high"]) >= limit else None

    # -- signal generation ------------------------------------------------

    def _queue_signals(self, when: pd.Timestamp, feats: dict[str, pd.Series],
                       funding: dict[str, float], vol_flags: dict[str, bool],
                       prices: dict[str, float]) -> None:
        fresh: list[Signal] = []
        for sym, row in feats.items():
            sig = strategy.generate(row, sym, funding.get(sym, 0.0), self.params,
                                    vol_flags.get(sym, False))
            if sig is not None:
                fresh.append(sig)

        if not fresh:
            return

        for name, book in self.books.items():
            st = self.state[name]
            if st.dead or not broker.can_trade(st, when):
                continue
            room = book.max_concurrent - len(st.positions)
            if room <= 0:
                continue

            eligible = [s for s in fresh
                        if s.symbol not in st.positions
                        and (name, s.symbol) not in self.cooldown]

            # When more signals fire than there is room for, prefer the tighter
            # stop: same dollar risk, more size, faster resolution.
            eligible.sort(key=lambda s: s.stop_dist / max(s.ref_price, 1e-9))
            self.pending[name] = eligible[:room]

    # -- bookkeeping ------------------------------------------------------

    def _record_equity(self, when: pd.Timestamp, prices: dict[str, float]) -> None:
        row = {"time": str(when)}
        for name, st in self.state.items():
            row[name] = round(st.equity(prices), 4)
        self.equity_curve.append(row)

    def summary(self, prices: dict[str, float]) -> list[dict]:
        out = []
        for name, st in self.state.items():
            eq = st.equity(prices)
            out.append({
                "book": st.label,
                "equity": round(eq, 2),
                "return_pct": round((eq / self.start_usd - 1) * 100, 1),
                "peak": round(st.peak_equity, 2),
                "max_dd_pct": round((1 - eq / st.peak_equity) * 100, 1)
                if st.peak_equity else 0.0,
                "trades": st.n_trades,
                "win_rate": round(100 * st.n_wins / st.n_trades, 1)
                if st.n_trades else 0.0,
                "liquidations": st.n_liquidations,
                "fees": round(st.fees_total, 2),
                "funding": round(st.funding_total, 2),
                "open": len(st.positions),
                "dead": st.dead,
            })
        return out

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_usd": self.start_usd,
            "state": {k: v.to_dict() for k, v in self.state.items()},
            "pending": {k: [_signal_to_dict(s) for s in v]
                        for k, v in self.pending.items()},
            "cooldown": {f"{k[0]}|{k[1]}": v for k, v in self.cooldown.items()},
        }

    def load_dict(self, d: dict[str, Any]) -> None:
        self.start_usd = d.get("start_usd", self.start_usd)
        for k, v in d.get("state", {}).items():
            if k in self.state:
                self.state[k] = broker.BookState.from_dict(v)
        self.pending = {k: [_signal_from_dict(s) for s in v]
                        for k, v in d.get("pending", {}).items()}
        for k, v in d.get("cooldown", {}).items():
            book, sym = k.split("|", 1)
            self.cooldown[(book, sym)] = v
