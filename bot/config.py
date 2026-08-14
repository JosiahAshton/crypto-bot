"""Every tunable number in the system, in one place.

Kept deliberately flat and boring so the strategy modules stay readable and so
one file review tells you exactly what the bot is allowed to do.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Account
# --------------------------------------------------------------------------

START_EQUITY_AUD = 100.0
FALLBACK_AUD_USD = 0.66  # only used if the FX lookup fails; frozen into state

# Books are held in USD internally (perps are USD-denominated) and reported in
# both. The AUD/USD rate is fetched once at init and frozen, so P&L reflects
# trading only and never FX drift.

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

TIMEFRAME = "1h"
BAR_MINUTES = 60

CORE_SYMBOLS = ["BTCUSDT", "SOLUSDT"]  # always traded
ROTATING_SLOTS = 3  # re-screened weekly

# Screening rules for the rotating slots. The pump-coins that top the raw
# volatility table (freshly listed, +1000% in a month) have no usable history
# and ruinous slippage, so minimum listing age is doing a lot of work here.
MIN_QUOTE_VOLUME_USD = 100e6  # 24h notional
# Every alt in the research sample had a year or more of history. Trading a
# 90-day-old listing would mean betting on an instrument unlike anything the
# edge was measured on, so the live screen is held to the same standard.
MIN_LISTING_DAYS = 270
MAX_ATR_PCT_DAY = 12.0  # above this is a listing pump, not a tradeable market
SCREEN_EVERY_DAYS = 7

# --------------------------------------------------------------------------
# Costs — the part most paper bots quietly cheat on
# --------------------------------------------------------------------------

TAKER_FEE = 0.0005  # 0.05% — market orders (breakout entries, all stops)
MAKER_FEE = 0.0002  # 0.02% — resting limit orders (mean-reversion entries)

# Slippage = half-spread + size impact + delay. Delay matters because the live
# engine wakes on a cron rather than at the instant the bar closes, so an entry
# can land several minutes late. The backtest applies the same penalty so the
# historical numbers stay comparable to the live run.
HALF_SPREAD_BPS = {"BTCUSDT": 0.5, "ETHUSDT": 0.5, "SOLUSDT": 1.0}
DEFAULT_HALF_SPREAD_BPS = 3.0
IMPACT_BPS_PER_10K = 1.0  # additional bps per $10k of notional
CRON_DELAY_MINUTES = 10  # GitHub Actions cron granularity
DELAY_SLIPPAGE_FRACTION = 0.25  # fraction of one bar's ATR paid to arrive late

FUNDING_HOURS = (0, 8, 16)  # UTC hours when funding settles

# --------------------------------------------------------------------------
# Exchange limits — real, and they bite at high leverage
# --------------------------------------------------------------------------

MAX_LEVERAGE = {
    "BTCUSDT": 125,
    "ETHUSDT": 100,
    "SOLUSDT": 75,
    "XRPUSDT": 75,
    "BNBUSDT": 75,
    "DOGEUSDT": 75,
}
DEFAULT_MAX_LEVERAGE = 25  # most alt perps cap here or lower at tier 1

# Maintenance margin rate at the smallest notional tier. Liquidation distance
# is roughly (1/leverage - MMR), which is why 100x dies at ~0.6% adverse.
MAINTENANCE_MARGIN = {
    "BTCUSDT": 0.004,
    "ETHUSDT": 0.004,
    "SOLUSDT": 0.010,
}
DEFAULT_MAINTENANCE_MARGIN = 0.025

# --------------------------------------------------------------------------
# Strategy — regime router
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyParams:
    """Settings chosen on 2023-2024 and then judged on 2025-2026 untouched.

    Three results from `research.py confirm` drove every number here, and all
    three run against intuition:

      * The chandelier trail destroyed the edge. Every top configuration turned
        it off. Trailing a 1h crypto breakout just donates the move back.
      * Wider stops beat tighter ones, because cost is a fixed toll and a 4-ATR
        stop pays it over a bigger move: 0.074R per trade at 5 ATR against
        0.122R at 3 ATR.
      * Majors are not tradeable here. On BTC/ETH/SOL/XRP/DOGE the gross edge
        (+0.100R) and the costs (0.100R) cancel exactly. The high-volatility
        alts clear the same toll with +0.189R gross.
    """

    atr_period: int = 14

    # Regime classification
    efficiency_period: int = 20
    trending_threshold: float = 0.46  # Kaufman efficiency ratio above this = trend
    chop_threshold: float = 0.40  # below this = mean-reverting
    bandwidth_period: int = 20
    bandwidth_lookback: int = 300
    compressed_pct: float = 35.0  # bandwidth percentile below this = compressed

    # Trend module: Donchian breakout, wide stop, no trail
    donchian_period: int = 120  # 5 days of hourly bars
    trend_ema: int = 200
    use_ema_filter: bool = False  # made no difference once the ER floor is 0.46
    breakout_stop_atr: float = 4.0
    breakout_trail_atr: float | None = None  # trailing tested worse, everywhere
    breakout_max_bars: int = 240

    # Mean-reversion module: z-score vs rolling VWAP.
    # Marginal at best — profit factor 1.08 over 480 trades. Kept because it
    # trades in the regimes the breakout module sits out, not because it shines.
    vwap_period: int = 48
    z_entry: float = 3.0
    z_exit: float = 0.3
    mr_stop_atr: float = 2.0
    mr_target_r: float = 2.0
    mr_max_bars: int = 24

    # Funding tilt. Extreme positive funding means crowded longs, so new longs
    # are blocked and shorts get a small edge (and vice versa).
    funding_block: float = 0.0005  # 0.05% per 8h ~= 55% annualised

    # Quality gate. Fee drag at leverage is brutal, so a signal has to clear a
    # minimum expected reward-to-risk before it is worth paying the spread.
    min_reward_risk: float = 1.6

    # ATR floor/ceiling as a fraction of price — refuse to trade dead or
    # unhinged conditions.
    min_atr_pct: float = 0.0025
    max_atr_pct: float = 0.060


PARAMS = StrategyParams()

# --------------------------------------------------------------------------
# Books
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Book:
    name: str
    label: str
    risk_per_trade: float | None  # fraction of equity risked to the stop
    fixed_leverage: float | None  # if set, notional = equity * this, ignore risk%
    max_concurrent: int = 3
    daily_loss_limit: float = 0.30  # halt for 24h after losing this much in a day
    dead_below: float = 5.0  # USD; below this the book is marked dead


BOOKS: list[Book] = [
    Book("A_conservative", "Conservative 2%", risk_per_trade=0.02, fixed_leverage=None),
    Book("B_aggressive", "Aggressive 8%", risk_per_trade=0.08, fixed_leverage=None),
    Book("C_degen", "Degen 20%", risk_per_trade=0.20, fixed_leverage=None),
    # Measured optimum, not a guess. `risk_ladder.py` runs nine risk settings
    # through 255 rolling 30-day windows: the chance of doubling and the mean
    # outcome both peak here, at 13.3% and 1.14x. Past 16% the doubling odds
    # stop improving while ruin risk and drawdown keep climbing.
    Book("E_optimal", "Optimal 12%", risk_per_trade=0.12, fixed_leverage=None),
    # Exactly what was asked for: 100x notional on every trade, stop forced
    # inside the liquidation band. No daily loss limit — it runs until it dies,
    # because that outcome is the experiment.
    Book(
        "D_max_leverage",
        "Max Leverage 100x",
        risk_per_trade=None,
        fixed_leverage=100.0,
        max_concurrent=1,
        daily_loss_limit=1.0,
    ),
]

# Fraction of the distance-to-liquidation at which book D's stop is placed.
# Any closer and normal noise triggers it; any further and the exchange gets
# there first and takes the whole margin plus fees.
LIQ_STOP_FRACTION = 0.60

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

CACHE_DIR = "cache"
STATE_DIR = "state"
REPORT_DIR = "reports"
STATE_FILE = "state/live_state.json"
TRADE_LOG = "state/trades.csv"
EQUITY_LOG = "state/equity.csv"

BACKTEST_START = "2023-01-01"  # ~2.5 years of hourly bars
