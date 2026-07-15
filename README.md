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
├── data/fetch.py           OHLCV downloader              →  sl-fetch
├── strategies/
│   ├── community.py        backtesting.py strategy library
│   └── scalp.py            zero-fee maker scalp engine   →  sl-scalp
├── backtest/
│   ├── run.py              community-strategy runner     →  sl-backtest
│   ├── maker.py            taker-vs-maker comparison     →  sl-maker
│   ├── regime.py           regime-aware maker ladder     →  sl-regime
│   └── scalp_optimize.py   out-of-sample scalp search    →  sl-scalp-opt
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

Output lands in `data/`, e.g. `data/binance_BTC-USDT_1h.csv`.
Columns: `timestamp` (epoch ms), `datetime` (UTC), `open`, `high`, `low`, `close`, `volume`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--exchange`, `-e` | `binance` | ccxt exchange id (binance, bybit, okx, coinbase, kraken, kucoin, ...) |
| `--symbol`, `-s` | — (required) | One or more pairs, e.g. `BTC/USDT ETH/USDT` |
| `--timeframe`, `-t` | `1h` | Candle size: `1m 5m 15m 1h 4h 1d` ... |
| `--since` | 1 year ago | Start date `YYYY-MM-DD` (UTC) |
| `--until` | now | End date `YYYY-MM-DD` (UTC, exclusive) |
| `--format`, `-f` | `csv` | `csv` or `parquet` |
| `--out`, `-o` | `data` | Output directory |

## 2. Backtest & research

The commands fall into three groups. All take `--file data/<something>.csv` and default to
the 1h BTC/USDT set.

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

### Reports & P&L

| Command | Module | What it does |
|---------|--------|--------------|
| `sl-ema-report` | `strategylab.reports.ema` | Detailed anti-overfit tearsheet for the fixed EMA 12/26 cross (per-year, train/test, neighbour sweep), maker and taker. Writes HTML + Markdown to `reports/`. |
| `sl-combo-report` | `strategylab.reports.combo` | Same protocol for the trend-filtered dip-buy combo (`SMA50` + `RSI14<45`) that survived the `sl-sweep` search. |
| `sl-daily-pnl` | `strategylab.reports.daily_pnl` | Turns a scalp equity curve into a **daily dollar P&L distribution** on a fixed capital base — "can I make $X/day?". |

Generated reports land in `reports/` as paired `.html` / `.md` files.

## 3. Chart viewer

A tiny offline web viewer for the CSVs in `data/`, using vendored
TradingView lightweight-charts (no account, no external calls):

```bash
./.venv/bin/python viewer/server.py
# then open http://127.0.0.1:8000
```

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
