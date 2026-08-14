"""OKX public market data — the fallback when Binance is geo-blocked.

GitHub's runners sit on US IPs and Binance answers them with HTTP 451, so the
bot cannot use its primary source in CI. OKX answers the same runners with 200,
carries USDT-margined perps with candle and funding history, and quotes the same
assets to within a few basis points of Binance, so the strategy that was
validated on Binance data behaves the same here.

Exposes the same four operations as the Binance layer, in the same shapes, so
callers do not care which venue answered.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE = "https://www.okx.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "crypto-bot/1.0"})

# OKX caps history pagination at 100 rows per request.
PAGE = 100
_BAR = {"1h": "1H", "15m": "15m", "4h": "4H", "1d": "1D"}
_MS = {"1h": 3_600_000, "15m": 900_000, "4h": 14_400_000, "1d": 86_400_000}


def _get(path: str, params: dict | None = None, retries: int = 5):
    delay = 1.0
    for attempt in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=30)
            if r.status_code in (429, 503):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            payload = r.json()
            if payload.get("code") not in ("0", 0):
                raise RuntimeError(f"okx error {payload.get('code')}: "
                                   f"{payload.get('msg')}")
            return payload["data"]
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {path}")


# --------------------------------------------------------------------------
# Symbol naming
# --------------------------------------------------------------------------


def inst_id(symbol: str) -> str:
    """BTCUSDT -> BTC-USDT-SWAP. Already-native ids pass straight through."""
    if symbol.endswith("-SWAP"):
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT-SWAP"
    return symbol


def canonical(inst: str) -> str:
    """BTC-USDT-SWAP -> BTCUSDT, for display alongside Binance names."""
    parts = inst.split("-")
    return f"{parts[0]}{parts[1]}" if len(parts) >= 2 else inst


# --------------------------------------------------------------------------
# Candles
# --------------------------------------------------------------------------


def klines(symbol: str, interval: str, start, end=None) -> pd.DataFrame:
    """Closed candles from `start` onward.

    OKX pages backwards from the present, so this walks back until it reaches
    `start` and then flips the result into ascending order.
    """
    inst = inst_id(symbol)
    bar = _BAR[interval]
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    cursor = int(pd.Timestamp(end).timestamp() * 1000) if end is not None else None

    rows: list[list] = []
    while True:
        params = {"instId": inst, "bar": bar, "limit": PAGE}
        if cursor is not None:
            params["after"] = cursor  # OKX: strictly older than this timestamp
        batch = _get("/api/v5/market/history-candles", params)
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[-1][0])
        if oldest <= start_ms or len(batch) < PAGE:
            break
        cursor = oldest
        time.sleep(0.12)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "vol", "volCcy", "volCcyQuote", "confirm"])
    df = df.drop_duplicates(subset="ts")
    # confirm == "0" means the candle is still forming.
    df = df[df["confirm"] == "1"]
    df["ts"] = df["ts"].astype("int64")
    df = df[df["ts"] >= start_ms].sort_values("ts")

    for col in ("open", "high", "low", "close", "volCcy"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volCcy"]
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "open_time"
    return df[["open", "high", "low", "close", "volume"]]


def forming_bar(symbol: str, interval: str) -> pd.Series | None:
    """The candle currently being built, if there is one."""
    data = _get("/api/v5/market/candles",
                {"instId": inst_id(symbol), "bar": _BAR[interval], "limit": 2})
    if not data:
        return None
    newest = data[0]  # OKX returns newest first
    if newest[8] == "1":
        return None  # already closed
    return pd.Series({
        "open_time": pd.to_datetime(int(newest[0]), unit="ms", utc=True),
        "open": float(newest[1]), "high": float(newest[2]),
        "low": float(newest[3]), "close": float(newest[4]),
        "volume": float(newest[6]),
    })


# --------------------------------------------------------------------------
# Funding
# --------------------------------------------------------------------------


def funding(symbol: str, start, end=None) -> pd.Series:
    inst = inst_id(symbol)
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    cursor = int(pd.Timestamp(end).timestamp() * 1000) if end is not None else None

    rows: list[dict] = []
    while True:
        params = {"instId": inst, "limit": PAGE}
        if cursor is not None:
            params["after"] = cursor
        batch = _get("/api/v5/public/funding-rate-history", params)
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[-1]["fundingTime"])
        if oldest <= start_ms or len(batch) < PAGE:
            break
        cursor = oldest
        time.sleep(0.12)

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows).drop_duplicates(subset="fundingTime")
    df["fundingTime"] = df["fundingTime"].astype("int64")
    df = df[df["fundingTime"] >= start_ms]
    df["fundingRate"] = df["fundingRate"].astype(float)
    df.index = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df["fundingRate"].sort_index()


def current_funding(symbol: str) -> float:
    try:
        data = _get("/api/v5/public/funding-rate", {"instId": inst_id(symbol)})
        return float(data[0]["fundingRate"]) if data else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------


def perp_tickers() -> list[tuple[str, float]]:
    """Every USDT perp with its 24h quote volume in USD."""
    tickers = _get("/api/v5/market/tickers", {"instType": "SWAP"})
    out = []
    for t in tickers:
        inst = t["instId"]
        if not inst.endswith("-USDT-SWAP"):
            continue
        try:
            # volCcy24h is denominated in the base asset for USDT-margined swaps.
            quote_vol = float(t["volCcy24h"]) * float(t["last"])
        except (ValueError, KeyError, TypeError):
            continue
        out.append((inst, quote_vol))
    return out


def instruments() -> dict[str, dict]:
    """Every SWAP instrument, keyed by instId.

    `listTime` here is authoritative for listing age, and one call covers the
    whole venue — far cheaper than counting daily bars per symbol.
    """
    return {d["instId"]: d for d in _get("/api/v5/public/instruments",
                                         {"instType": "SWAP"})}


def daily_candles(symbol: str, limit: int = 100) -> pd.DataFrame:
    data = _get("/api/v5/market/candles",
                {"instId": inst_id(symbol), "bar": "1D", "limit": limit})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close",
                                     "vol", "volCcy", "volCcyQuote", "confirm"])
    for col in ("high", "low", "close"):
        df[col] = df[col].astype(float)
    df["ts"] = df["ts"].astype("int64")
    return df.sort_values("ts")[["high", "low", "close"]]


def max_leverage(symbol: str) -> float | None:
    try:
        data = _get("/api/v5/public/instruments",
                    {"instType": "SWAP", "instId": inst_id(symbol)})
        return float(data[0]["lever"]) if data else None
    except Exception:
        return None


def ping() -> bool:
    try:
        _get("/api/v5/public/time")
        return True
    except Exception:
        return False
