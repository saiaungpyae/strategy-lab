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
pip install -r requirements.txt
```

## 1. Get data

```bash
# 1 year of BTC/USDT hourly candles from Binance
python fetch_ohlcv.py --symbol BTC/USDT --timeframe 1h --since 2024-01-01

# Multiple pairs, 15m candles, saved as Parquet
python fetch_ohlcv.py -s BTC/USDT ETH/USDT SOL/USDT -t 15m --since 2023-06-01 -f parquet

# A different exchange with an explicit window
python fetch_ohlcv.py --exchange bybit -s BTC/USDT -t 4h --since 2022-01-01 --until 2023-01-01
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

The scripts fall into three groups. All take `--file data/<something>.csv` and default to
the 1h BTC/USDT set.

### Strategy comparison

| Script | What it does |
|--------|--------------|
| `run_backtest.py` | Runs the classic community strategies in `bt_strategies.py` (supertrend, EMA cross, ...) with realistic fees, ranked against Buy & Hold. |
| `bt_strategies.py` | Library of long-only community strategies for `backtesting.py`. No lookahead — orders fill at the next bar's open. |
| `maker_backtest.py` | Same strategies run **two ways** — TAKER (guaranteed fill, pays fee) vs MAKER (zero fee, but fills only if price comes to your limit). Exposes the hidden cost of maker-only execution. |

### Overfit-resistant search

| Script | What it does |
|--------|--------------|
| `strategy_lab.py` | Mass sweep of thousands of configs with a fast vectorized engine and a **train/test split** — the point is to separate real edge from multiple-testing luck. |
| `scalp_lab.py` | Frequent low-timeframe scalps for a zero-fee maker, modeling adverse-selection cost as a tunable per-fill "edge cost" in bps. |
| `scalp_optimize.py` | Grid search for a scalp with **out-of-sample surviving** edge at a realistic per-fill cost — no rigged reward:risk, no peeking at the test set. |
| `regime_lab.py` | Regime-aware, risk-managed maker strategy built as a cumulative ladder (trend filter → regime filter → vol targeting → circuit breaker) tested against a 0.03%/day target. |

### Reports & P&L

| Script | What it does |
|--------|--------------|
| `ema_report.py` | Detailed anti-overfit tearsheet for the fixed EMA 12/26 cross (per-year, train/test, neighbour sweep), maker and taker. Writes HTML + Markdown to `reports/`. |
| `combo_report.py` | Same protocol for the trend-filtered dip-buy combo (`SMA50` + `RSI14<45`) that survived the `strategy_lab` sweep. |
| `daily_pnl.py` | Turns a scalp equity curve into a **daily dollar P&L distribution** on a fixed capital base — "can I make $X/day?". |

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

`ccxt`, `pandas`, `pyarrow` (Parquet only). See `requirements.txt`. The backtesting
scripts additionally use `backtesting.py` / vectorized engines as noted in each file.
