# strategy-lab — a local crypto backtesting lab

A self-contained toolkit for fetching historical crypto candles and stress-testing
trading strategies offline — with a strong bias toward **honest, anti-overfit
evaluation** and **realistic maker (limit-order) execution** for a zero-fee liquidity
provider.

Everything runs locally against free public data. No API keys, no TradingView account,
no live trading.

## Setup (one time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # installs deps and the `sl-*` commands below
```

The code lives in the `strategylab` package under `src/`. The editable install
(`pip install -e .`) puts the `sl-*` console commands on your `PATH`; each command
is also runnable as `python -m strategylab.<module>` if you prefer.

```
src/strategylab/
├── core.py                 shared indicators + the mass config sweep
├── data/fetch.py           OHLCV downloader + updater    →  sl-fetch, sl-update
├── strategies/
│   ├── community.py        backtesting.py strategy library
│   └── scalp.py            zero-fee maker scalp engine   →  sl-scalp
├── backtest/
│   ├── run.py              community-strategy runner     →  sl-backtest
│   ├── maker.py            taker-vs-maker comparison     →  sl-maker
│   ├── regime.py           regime-aware maker ladder     →  sl-regime
│   └── scalp_optimize.py   out-of-sample scalp search    →  sl-scalp-opt
├── swarm/                  bot-swarm pattern search      →  sl-swarm
│   ├── genome.py           trait + perception sampling (reproducible per seed)
│   ├── features.py         feature pool + Binance Vision metrics downloader
│   ├── engine.py           vectorized sim: maker fills, ruin line, sizing
│   └── recap.py            yardstick / trait / persistence statistics
└── reports/
    ├── ema.py              EMA/SMA-cross tearsheet        →  sl-ema-report
    ├── combo.py            SMA50+RSI14 combo tearsheet    →  sl-combo-report
    └── daily_pnl.py        daily dollar-P&L distribution  →  sl-daily-pnl
```

Run everything from the project root — `data/` and `reports/` are resolved
relative to the current directory.

## 1. Get data

```bash
# 1 year of BTC/USDT hourly candles from Binance
sl-fetch --symbol BTC/USDT --timeframe 1h --since 2024-01-01

# Multiple pairs, 15m candles, saved as Parquet
sl-fetch -s BTC/USDT ETH/USDT SOL/USDT -t 15m --since 2023-06-01 -f parquet

# A different exchange with an explicit window
sl-fetch --exchange bybit -s BTC/USDT -t 4h --since 2022-01-01 --until 2023-01-01
```

Output lands in a per-pair folder under `data/ohlcv/`, e.g.
`data/ohlcv/BTC-USDT/binance_BTC-USDT_1h.csv`. Derivatives metrics
(`sl-swarm fetch-metrics` / `fetch-funding`) land in `data/metrics/<PAIR>/`,
and pinned dataset snapshots live in `data/snapshots/` (never auto-updated):

```
data/
├── ohlcv/BTC-USDT/binance_BTC-USDT_5m.csv     (+ .parquet sidecar caches)
├── metrics/BTC-USDT/BTCUSDT_metrics.csv
├── metrics/BTC-USDT/BTC-USDT-USDT_funding.csv
└── snapshots/pin-YYYYMMDD/...
```

Columns: `timestamp` (epoch ms), `datetime` (UTC), `open`, `high`, `low`, `close`, `volume`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--exchange`, `-e` | `binance` | ccxt exchange id (binance, bybit, okx, coinbase, kraken, kucoin, ...) |
| `--symbol`, `-s` | — (required) | One or more pairs, e.g. `BTC/USDT ETH/USDT` |
| `--timeframe`, `-t` | `1h` | Candle size: `1m 5m 15m 1h 4h 1d` ... |
| `--since` | 1 year ago | Start date `YYYY-MM-DD` (UTC) |
| `--until` | now | End date `YYYY-MM-DD` (UTC, exclusive) |
| `--format`, `-f` | `csv` | `csv` or `parquet` |
| `--out`, `-o` | `data` | Data root (files land under `<out>/ohlcv/<PAIR>/`) |

### Keeping data current

`sl-update` brings every dataset under `data/` up to the present
**incrementally** — it reads the last saved candle per file and fetches only
the missing span (re-finalizing the last candle and skipping the still-forming
one). Pinned `data/snapshots/` are left untouched:

```bash
sl-update            # update everything in data/
sl-update -d other/  # a different data directory
```

The chart viewer also runs the same update automatically in the background on
startup, and on demand via the dashboard's *refresh data* button
(`POST /api/refresh`).

## 2. Backtest & research

The commands fall into three groups. All take `--file data/ohlcv/<PAIR>/<something>.csv`
and default to the 1h BTC/USDT set.

### Strategy comparison

| Command | Module | What it does |
|---------|--------|--------------|
| `sl-backtest` | `strategylab.backtest.run` | Runs the classic community strategies (supertrend, EMA cross, ...) with realistic fees, ranked against Buy & Hold. |
| — | `strategylab.strategies.community` | Library of long-only community strategies for `backtesting.py`. No lookahead — orders fill at the next bar's open. |
| `sl-maker` | `strategylab.backtest.maker` | Same strategies run **two ways** — TAKER (guaranteed fill, pays fee) vs MAKER (zero fee, but fills only if price comes to your limit). Exposes the hidden cost of maker-only execution. |

### Overfit-resistant search

| Command | Module | What it does |
|---------|--------|--------------|
| `sl-sweep` | `strategylab.core` | Mass sweep of thousands of configs with a fast vectorized engine and a **train/test split** — the point is to separate real edge from multiple-testing luck. |
| `sl-scalp` | `strategylab.strategies.scalp` | Frequent low-timeframe scalps for a zero-fee maker, modeling adverse-selection cost as a tunable per-fill "edge cost" in bps. |
| `sl-scalp-opt` | `strategylab.backtest.scalp_optimize` | Grid search for a scalp with **out-of-sample surviving** edge at a realistic per-fill cost — no rigged reward:risk, no peeking at the test set. |
| `sl-regime` | `strategylab.backtest.regime` | Regime-aware, risk-managed maker strategy built as a cumulative ladder (trend filter → regime filter → vol targeting → circuit breaker) tested against a 0.03%/day target. |
| `sl-swarm` | `strategylab.swarm.run` | **Bot swarm**: thousands of sampled bots (perception rules over market features + behavioral traits) plus a random placebo group, judged strictly out-of-sample. `run` simulates and writes artifacts to `reports/swarm/<run_id>/` (`--maker-only` restricts entries to resting limits — stops still exit as taker; `--metrics` / `--funding` merge derivatives sentiment into the feature pool); `fetch-metrics` downloads open-interest / long-short-ratio history (5m, Sep 2020+) from the Binance Vision archive; `fetch-funding` pulls perp funding-rate history via ccxt; `report` rebuilds a recap. Results dashboard at `/swarm` in the viewer. Design rationale: `bot-swarm-discussion.md`. |

### Reports & P&L

| Command | Module | What it does |
|---------|--------|--------------|
| `sl-ema-report` | `strategylab.reports.ema` | Detailed anti-overfit tearsheet for the fixed EMA 12/26 cross (per-year, train/test, neighbour sweep), maker and taker. Writes HTML + Markdown to `reports/`. |
| `sl-combo-report` | `strategylab.reports.combo` | Same protocol for the trend-filtered dip-buy combo (`SMA50` + `RSI14<45`) that survived the `sl-sweep` search. |
| `sl-daily-pnl` | `strategylab.reports.daily_pnl` | Turns a scalp equity curve into a **daily dollar P&L distribution** on a fixed capital base — "can I make $X/day?". |

Generated reports land in `reports/` as paired `.html` / `.md` files.

## 3. Dashboard & chart viewer

A local web UI for the CSVs in `data/`, using vendored TradingView
lightweight-charts (no account; the only external calls are the incremental
data refreshes against the exchange's public API):

```bash
./.venv/bin/python viewer/server.py
# then open http://127.0.0.1:8000   (HOST/PORT configurable via .env)
```

- **`/` — dashboard**: per-dataset health cards (freshness, gaps, span),
  a multi-timeframe chart grid per symbol, latest signal states
  (EMA/SMA cross, supertrend, open FVGs), and the `reports/` folder as
  clickable links. Datasets auto-refresh in the background on startup;
  the *refresh data* button re-runs it anytime.
- **`/chart` — drill-down**: the full single-chart view with US-session
  shading, signal overlays, and the FVG event study
  (deep-linkable as `/chart?file=binance_BTC-USDT_15m.csv`).

## Design principles

- **Maker-first execution.** The zero-fee edge only exists if your limit orders
  actually fill. Every serious script models unfilled orders and adverse selection
  rather than assuming `fee = 0` and a guaranteed fill.
- **Out-of-sample or it doesn't count.** Sweeps are always judged on unseen test
  data; fixed-parameter reports show per-year and neighbour-sweep robustness so a
  result can't hide behind one lucky bull run.
- **Reconciling engines.** Report scripts reuse the same execution engines and
  metrics, so numbers across tearsheets line up.

## Requirements

`ccxt`, `pandas`, `pyarrow` (Parquet only) — declared in `pyproject.toml` and
installed by `pip install -e .` (also mirrored in `requirements.txt`). The
`sl-backtest` / community-strategy modules additionally need `backtesting.py`
(`pip install backtesting`); the vectorized engines don't.
