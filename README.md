# Crypto perp paper bot — five books, one signal engine

### 👉 [**Live scoreboard — PnL and trade history**](STATUS.md)

Updated every 10 minutes by the bot itself. Renders on a phone.

---

An autonomous paper-trading bot for crypto perpetual futures. It runs itself on
a schedule, trades a screened universe of Binance USDⓈ-M perps on hourly bars,
and keeps five independent $100 ledgers side by side at different risk levels so
you can see what leverage actually does to the same set of trades.

It was built to answer one question honestly: **can a trading bot double $100 in
a month?** The short version of the answer is in [Findings](#findings), and it
is not the answer anyone wants.

---

## What it does

- Trades **BTC + SOL** plus **three high-volatility alt slots**, re-screened
  weekly for liquidity, listing age and realised volatility.
- Runs a **regime router**: a Donchian breakout module when the market is
  trending, a VWAP mean-reversion module when it is chopping, and nothing at all
  when it is neither.
- Applies a **funding-rate tilt** — when perp funding is extremely positive the
  long side is crowded and paying to stay there, so new longs are blocked.
- Books every trade through a **paper broker that does not flatter itself**:
  fees on notional, 8-hourly funding, real maintenance-margin liquidation,
  spread and size impact, and a penalty for arriving late because the cron woke
  up after the bar closed.
- Publishes a **self-contained HTML dashboard** you can open from anywhere.

### The five books

| Book | Sizing | Typical leverage |
|---|---|---|
| A — Conservative | 2% of equity risked per trade | ~3–8x |
| B — Aggressive | 8% risked per trade | ~10–25x |
| E — Optimal | 12% risked per trade — the measured peak | ~12–30x |
| C — Degen | 20% risked per trade | ~15–40x |
| D — Max Leverage | fixed 100x notional, stop forced inside liquidation | 100x |

All five see identical signals. Only the position sizing differs, which is what
makes the comparison worth anything.

---

## Findings

Settings were chosen on **2023–2024** and then judged on **2025–2026**, which
took no part in the choosing. Everything below is out-of-sample unless stated.

**1. There is a real edge, and it is small.**
Trend-following breakouts on crypto perps produce roughly **+0.15R gross per
trade** out-of-sample, positive in 90 of 108 parameter configurations tested.
The consistency matters more than the size — it is a broad plateau, not a lucky
cell in a grid.

**2. Costs eat most of it, and on majors they eat all of it.**

| Universe | Gross edge | Costs | Net |
|---|---|---|---|
| BTC/ETH/SOL/XRP/DOGE | +0.100R | 0.100R | **±0.000R** |
| High-volatility alts | +0.189R | 0.091R | **+0.098R** |

Same fee schedule, bigger moves. This is why the bot deliberately weights toward
volatile alts rather than the "safe" majors — on majors the strategy is exactly
break-even before it has done anything wrong.

**3. Wider stops beat tighter ones**, because cost is a fixed toll paid over a
variable distance: 0.074R per trade at a 5-ATR stop against 0.122R at 3 ATR.

**4. Trailing stops destroyed the edge.** Every top configuration turned the
chandelier trail off. Trailing a 1h crypto breakout donates the move back.

**5. Leverage does not amplify the edge. It deletes it.** Over the full sample,
on identical trades:

| Book | Final (from $66) | Return | Profit factor | Fees paid |
|---|---|---|---|---|
| Conservative 2% | $239.12 | **+262%** | 1.14 | $55 |
| Aggressive 8% | $12.66 | −81% | 1.01 | $125 |
| Degen 20% | $4.95 | −93% (dead) | 0.73 | $8 |
| Max Leverage 100x | $2.30 | −97% (dead, 1 trade) | 0.00 | $2 |

Book B took *the same trades* as Book A with a *positive* Sharpe of 0.80 and
still lost 81%, because 4x the notional means 4x the fees and the profit factor
falls from 1.14 to 1.01. That is the whole mechanism: **the edge scales with
price moves, the costs scale with leverage.**

**6. At 100x the arithmetic is fatal before the market is involved.** On $100:

- notional is $10,000, so a round trip costs **$10 — 10% of the account**
- entry slippage on a volatile alt costs roughly another **28% of margin**
- liquidation sits about **0.6%** away, which an hourly bar covers routinely
- funding bleeds around **3% of the account per day** just to hold

Book D was liquidated on its first trade and never took another.

### So — can it double $100 in a month?

Measured, not guessed. Every 30-day window in the sample was run as an
independent life for the account — 255 of them, each starting fresh at $100,
same trades, different sizing:

| Book | Median month | P(double) | P(profit) | P(lose half) | P(ruin) | Best | Worst |
|---|---|---|---|---|---|---|---|
| Conservative 2% | **1.00x** | 0.0% | 49.8% | 0.0% | 0.0% | 1.77x | 0.77x |
| Aggressive 8% | 0.91x | **9.8%** | 45.1% | 11.8% | 0.0% | 4.37x | 0.31x |
| Degen 20% | 0.68x | **11.8%** | 30.2% | 33.3% | 1.2% | 7.18x | 0.07x |
| Max Leverage 100x | 0.05x | **0.0%** | 0.4% | 97.6% | **94.5%** | 1.09x | 0.003x |

Read that table carefully, because it contains the whole answer:

- **Doubling is possible but uncommon.** At 8% risk it happened in about 1 month
  in 10. Not never — just not something to plan around.
- **The typical month loses money even when the average one wins.** Aggressive
  has a median of 0.91x and a mean of 1.12x. The edge is carried by rare large
  winners, so most months feel like failure while the average is fine.
- **Going from 8% to 20% risk is a bad trade.** It buys two extra percentage
  points of doubling odds (9.8% → 11.8%) and costs you a median month of 0.68x
  and a one-in-three chance of losing half.
- **100x doubled in zero of 255 months and was ruined in 94.5% of them.** Its
  best month ever was 1.09x. This is the finding worth keeping: at 100x the
  costs guarantee death long before the market gets a chance to be right.

### Where the doubling odds actually peak

`python risk_ladder.py` runs nine risk settings through the same 255 windows,
one signal engine feeding all of them, so only the sizing differs:

| Risk per trade | Median month | Mean | P(double) | P(ruin) | Worst |
|---|---|---|---|---|---|
| 2% | 1.00x | 1.04 | 0.0% | 0.0% | 0.77x |
| 5% | 0.96x | 1.08 | 5.1% | 0.0% | 0.51x |
| 8% | 0.91x | 1.13 | 9.8% | 0.0% | 0.31x |
| **12%** | 0.82x | **1.14** | **13.3%** | 0.0% | 0.14x |
| 16% | 0.77x | 1.11 | 13.3% | 0.0% | 0.13x |
| 20% | 0.68x | 1.04 | 12.5% | 2.4% | 0.06x |
| 28% | 0.61x | 1.00 | 12.2% | 2.7% | 0.04x |
| 40% | 0.48x | 1.01 | 11.4% | 6.7% | 0.05x |
| 55% | 0.48x | 1.00 | 12.2% | 8.2% | 0.05x |
| 100x fixed | 0.05x | 0.06 | **0.0%** | **94.5%** | 0.003x |

The doubling odds and the mean outcome peak together at **12% risk per trade**,
which is why book E exists. The important half of the table is what happens
after the peak: from 16% upward you buy **no additional chance of doubling**
while the median month falls from 0.77x to 0.48x and ruin risk climbs from zero
to 8%. That is pure downside, paid for voluntarily.

Best case at the optimum: doubling in roughly **1 month in 7.5**.

A positive-expectancy system here produces on the order of **1–2R per month**.
Doubling needs roughly 12R. That gap is not a tuning problem, and no amount of
leverage closes it — leverage moves you along the curve above, and past the peak
it moves you the wrong way.

---

## Running it

### Locally

```bash
python -m venv .venv && ./.venv/Scripts/pip install -r requirements.txt
```

```bash
python run.py screen
```

```bash
python run.py backtest --symbols BTCUSDT,SOLUSDT,ZECUSDT,HYPEUSDT,WLDUSDT
```

```bash
python run.py montecarlo --window 30 --step 5
```

```bash
python run.py once
```

```bash
python run.py report
```

`once` is a single tick and is what the cron calls. `daemon` is the same thing
on a timer for an always-on box.

### Autonomously, for free, with your PC off

Push this to a **public GitHub repo**. `.github/workflows/paper-trade.yml` runs
every 10 minutes on GitHub's own machines, commits the updated ledger back, and
costs nothing — public repos get unlimited Actions minutes and no card is
needed. Nothing secret is involved: the bot only reads public market data and
holds no keys.

1. Create a public repo and push.
2. **Settings → Actions → General → Workflow permissions → Read and write.**
3. **Actions → paper-trade → Run workflow** to start it immediately.

State lives in `state/live_state.json`, the trade log in `state/trades.csv`, and
the dashboard rebuilds to `reports/index.html` on every tick.

A tick is idempotent and catch-up safe: it processes every bar that closed since
the last run, so a missed window — or a whole day of them — costs entry timing
but never corrupts the ledger.

### On TradingView

`pine/regime_router.pine` is a Pine v6 twin of the signal logic. Paste it into
the Pine editor to see the regime shading, the Donchian channel and the entries
on a chart, and to cross-check against TradingView's own backtester.

It is the *signal* half only. Pine cannot express the four books, funding
payments, maintenance-margin liquidation or the cron delay, so its results will
look better than the engine's. Treat it as a visualiser, not a second opinion.

**TradingView cannot run this bot on its own paper account.** TradingView is a
charting engine, not an execution engine: strategy alerts need a paid plan for
webhooks, and those webhooks can only reach third-party bridges that trade a
real exchange account. Nothing routes back into the built-in paper account.
That constraint is why the engine here is standalone.

---

## Layout

```
bot/config.py     every tunable number, and why it has that value
bot/data.py       Binance public market data + incremental disk cache
bot/features.py   ATR, Donchian, efficiency ratio, bandwidth, VWAP z-score
bot/strategy.py   the regime router and its two modules
bot/broker.py     paper fills, fees, funding, margin, liquidation
bot/engine.py     the bar loop, shared by backtest and live
bot/backtest.py   historical run, walk-forward, 30-day Monte Carlo
bot/live.py       one cron tick, or the daemon
bot/report.py     self-contained HTML dashboard
research.py       signal-level R-multiple sweeps and out-of-sample confirmation
pine/             TradingView twin
```

Backtest and live trading share the same bar loop in `engine.py` on purpose. The
usual way a paper bot flatters itself is by running a tidy vectorised backtest
and a completely different live path, so the two never have to agree.

---

## Assumptions worth arguing with

- **Late fills.** Entries are charged 25% of one bar's ATR to model the cron
  arriving after the bar closed. It is applied always-adverse, which is
  pessimistic; real drift is random. Book D is very sensitive to this number.
- **Stop before liquidation.** Within a bar, the stop is always nearer than the
  liquidation price, so price must cross it first. Only a gap past the stop can
  liquidate.
- **Stop before target.** When a bar's range covers both, the stop is assumed to
  have come first. Without tick data there is no way to know, so the assumption
  is made against the account.
- **Leverage and margin caps** are conservative defaults per symbol, not fetched
  live. Most alts cap well below 100x in reality, and that cap is real.

## What this is not

Paper trading only. No exchange keys, no orders, no money. It is an experiment
whose purpose is to produce an honest answer, and the honest answer is already
above.
