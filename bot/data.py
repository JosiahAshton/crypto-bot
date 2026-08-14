"""Binance USDⓈ-M public market data, with an incremental on-disk cache.

Public endpoints only — no API key, nothing to leak, safe in a public repo.
Everything here returns UTC-indexed DataFrames.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config as C

BASE = "https://fapi.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "crypto-bot/1.0"})

_MS_PER_BAR = {"1h": 3_600_000, "15m": 900_000, "4h": 14_400_000, "1d": 86_400_000}


def _get(path: str, params: dict | None = None, retries: int = 5):
    """GET with exponential backoff. Binance 418/429 means slow down, not fail."""
    delay = 1.0
    for attempt in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=30)
            if r.status_code in (418, 429, 503):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {path}")


def as_utc(ts) -> pd.Timestamp:
    """Accept naive strings or already-aware timestamps, always return UTC."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _to_ms(ts) -> int:
    return int(as_utc(ts).timestamp() * 1000)


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("s")


# --------------------------------------------------------------------------
# Klines
# --------------------------------------------------------------------------

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def fetch_klines(symbol: str, interval: str, start, end=None) -> pd.DataFrame:
    """Paginate klines from `start` to `end` (exclusive of unclosed bars)."""
    step = _MS_PER_BAR[interval]
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(time.time() * 1000)
    frames = []
    cursor = start_ms

    while cursor < end_ms:
        rows = _get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval,
             "startTime": cursor, "limit": 1500},
        )
        if not rows:
            break
        df = pd.DataFrame(rows, columns=_KLINE_COLS)
        frames.append(df)
        last_open = int(df["open_time"].iloc[-1])
        if last_open + step >= end_ms or len(rows) < 1500:
            break
        cursor = last_open + step
        time.sleep(0.12)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="open_time").sort_values("open_time")
    num = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    out[num] = out[num].astype(float)
    out["open_time"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)

    # Drop the still-forming bar. Trading a bar that has not closed is the
    # single easiest way to fabricate a backtest edge that does not exist.
    now_ms = int(time.time() * 1000)
    out = out[out["close_time"].astype("int64") // 1_000_000 < now_ms]

    return out.set_index("open_time")[num]


def forming_bar(symbol: str, interval: str = C.TIMEFRAME) -> pd.Series | None:
    """The bar currently being built.

    Its `open` is the price at which the last closed bar's signal could have
    been acted on, and its running high/low tell us whether a resting limit
    would already have been hit. Both are needed to execute live the way the
    backtest assumes.
    """
    rows = _get("/fapi/v1/klines",
                {"symbol": symbol, "interval": interval, "limit": 2})
    if not rows:
        return None
    last = rows[-1]
    if int(last[6]) <= int(time.time() * 1000):
        return None  # already closed; nothing is forming yet
    return pd.Series({
        "open_time": pd.to_datetime(int(last[0]), unit="ms", utc=True),
        "open": float(last[1]), "high": float(last[2]),
        "low": float(last[3]), "close": float(last[4]),
        "volume": float(last[5]),
    })


def _cache_path(symbol: str, interval: str) -> str:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    return os.path.join(C.CACHE_DIR, f"{symbol}_{interval}.csv.gz")


def load_klines(symbol: str, interval: str = C.TIMEFRAME,
                start: str = C.BACKTEST_START, refresh: bool = True) -> pd.DataFrame:
    """Cached klines, topped up incrementally on each call."""
    path = _cache_path(symbol, interval)
    cached = pd.DataFrame()

    if os.path.exists(path):
        # Parse the index explicitly: pandas 3.0 stopped applying parse_dates
        # to the index column, which silently leaves it as strings.
        cached = pd.read_csv(path, index_col=0)
        cached.index = pd.to_datetime(cached.index, utc=True, format="ISO8601")

    # A cache built from a later start date cannot be topped up backwards, so
    # asking for more history than it holds means refetching the whole range.
    if not cached.empty and cached.index[0] > as_utc(start):
        cached = pd.DataFrame()

    if cached.empty:
        fresh = fetch_klines(symbol, interval, start)
    elif refresh:
        last = cached.index[-1]
        fresh = fetch_klines(symbol, interval, last)
    else:
        return cached

    if fresh.empty:
        return cached

    combined = (
        pd.concat([cached, fresh])
        if not cached.empty else fresh
    )
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_csv(path)
    return combined


# --------------------------------------------------------------------------
# Funding
# --------------------------------------------------------------------------


def fetch_funding(symbol: str, start, end=None) -> pd.Series:
    """Historical realised funding rates (settled every 8h)."""
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(time.time() * 1000)
    rows: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        batch = _get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["fundingTime"])
        if len(batch) < 1000:
            break
        cursor = last + 1
        time.sleep(0.12)

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows).drop_duplicates(subset="fundingTime")
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype("int64"),
                                       unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df.set_index("fundingTime")["fundingRate"].sort_index()


def load_funding(symbol: str, start: str = C.BACKTEST_START) -> pd.Series:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    path = os.path.join(C.CACHE_DIR, f"{symbol}_funding.csv.gz")
    cached = pd.Series(dtype=float)

    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0).iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True, format="ISO8601")
        cached = s

    # Same rule as klines: a cache that starts later than we now need has to be
    # rebuilt, because topping up only ever extends the recent end.
    if len(cached) and cached.index[0] > as_utc(start):
        cached = pd.Series(dtype=float)

    fresh = fetch_funding(symbol, cached.index[-1] if len(cached) else start)
    combined = pd.concat([cached, fresh]) if len(cached) else fresh
    if len(combined) == 0:
        return combined
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_frame("fundingRate").to_csv(path)
    return combined


def current_funding(symbol: str) -> float:
    """Live predicted funding rate for the next settlement."""
    try:
        data = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(data["lastFundingRate"])
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# Universe screening
# --------------------------------------------------------------------------


def exchange_info() -> dict:
    return _get("/fapi/v1/exchangeInfo")


def screen_universe(slots: int = C.ROTATING_SLOTS) -> list[str]:
    """Core symbols plus the most volatile *tradeable* liquid alts.

    Tradeable is doing the work: a freshly listed coin up 1500% in a month tops
    every volatility ranking and is completely unusable — no history to reason
    from, spreads that eat the edge, and funding rates that can run -2% per 8h.
    """
    info = exchange_info()
    meta = {
        s["symbol"]: s
        for s in info["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }

    tickers = _get("/fapi/v1/ticker/24hr")
    candidates = []

    for t in tickers:
        sym = t["symbol"]
        if sym not in meta or sym in C.CORE_SYMBOLS:
            continue
        if float(t["quoteVolume"]) < C.MIN_QUOTE_VOLUME_USD:
            continue
        candidates.append((sym, float(t["quoteVolume"])))

    candidates.sort(key=lambda x: -x[1])
    scored = []

    for sym, vol in candidates[:40]:
        try:
            # One call covers both checks. Counting real daily bars beats
            # trusting exchangeInfo's onboardDate, which is simply absent for
            # some symbols — and a missing field read as 0 dates them to 1970
            # and waves every new listing straight through the age filter.
            rows = _get("/fapi/v1/klines",
                        {"symbol": sym, "interval": "1d", "limit": 500})
        except Exception:
            continue
        if len(rows) < C.MIN_LISTING_DAYS:
            continue

        daily = pd.DataFrame(rows, columns=_KLINE_COLS).tail(32)
        for col in ("high", "low", "close"):
            daily[col] = daily[col].astype(float)
        if len(daily) < 20:
            continue
        prev_close = daily["close"].shift(1)
        tr = pd.concat([
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_pct = float((tr / daily["close"]).mean() * 100)
        if not (1.0 < atr_pct < C.MAX_ATR_PCT_DAY):
            continue
        scored.append((sym, atr_pct, vol))
        time.sleep(0.1)

    scored.sort(key=lambda x: -x[1])
    return list(C.CORE_SYMBOLS) + [s[0] for s in scored[:slots]]


# --------------------------------------------------------------------------
# FX
# --------------------------------------------------------------------------


def aud_usd() -> float:
    """AUD/USD, frozen into state on first run so P&L is trading-only."""
    try:
        r = SESSION.get("https://api.frankfurter.app/latest",
                        params={"from": "AUD", "to": "USD"}, timeout=15)
        r.raise_for_status()
        return float(r.json()["rates"]["USD"])
    except Exception:
        return C.FALLBACK_AUD_USD
