"""Feature pool — what swarm bots are allowed to perceive.

All features are causal (computed from data up to and including bar t).
Rule thresholds are expressed as quantiles and resolved against the TRAIN
segment only, so the test period never leaks into a bot's genome.

Also home of the Binance Vision `metrics` archive downloader (open interest,
long/short ratios, taker long/short volume ratio — 5m granularity, Sep 2020+),
which bypasses the 30-day limit of the live API endpoints.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"

# Metrics archive columns -> short names used for merged dataframe columns.
METRICS_COLS = {
    "sum_open_interest": "oi",
    "sum_toptrader_long_short_ratio": "top_ls_pos",
    "count_toptrader_long_short_ratio": "top_ls_acct",
    "count_long_short_ratio": "global_ls",
    "sum_taker_long_short_vol_ratio": "taker_ls",
}


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(n, min_periods=1).mean().to_numpy()


def _roll_std(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(n, min_periods=2).std().to_numpy()


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def compute_features(df: pd.DataFrame, bars_per_day: int) -> tuple[np.ndarray, list[str]]:
    """Return (F [n_bars x n_feat] float32, feature names).

    NaNs in the warmup region are fine — the engine skips a warmup window.
    """
    c = df["close"].to_numpy(dtype=np.float64)
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    day = bars_per_day
    feats: dict[str, np.ndarray] = {}

    with np.errstate(divide="ignore", invalid="ignore"):
        # momentum / trend
        feats["mom_fast"] = pd.Series(c).pct_change(max(day // 24, 2)).to_numpy()   # ~1h
        feats["mom_slow"] = pd.Series(c).pct_change(day).to_numpy()                 # ~1 day
        feats["ema_spread"] = (_ema(c, 12) - _ema(c, 48)) / c
        # mean-reversion / location
        delta = np.diff(c, prepend=c[0])
        up = pd.Series(np.clip(delta, 0, None)).rolling(14).mean().to_numpy()
        dn = pd.Series(np.clip(-delta, 0, None)).rolling(14).mean().to_numpy()
        feats["rsi"] = 100.0 - 100.0 / (1.0 + up / np.where(dn == 0, np.nan, dn))
        feats["bb_pos"] = (c - _roll_mean(c, 20)) / (2.0 * _roll_std(c, 20))
        rmin = pd.Series(l).rolling(day).min().to_numpy()
        rmax = pd.Series(h).rolling(day).max().to_numpy()
        feats["range_pos"] = (c - rmin) / np.where(rmax - rmin == 0, np.nan, rmax - rmin)
        # volatility
        a = atr(df)
        feats["atr_pct"] = a / c
        r1 = pd.Series(c).pct_change().to_numpy()
        feats["vol_regime"] = _roll_std(r1, max(day // 4, 2)) / _roll_std(r1, day * 2)
        # participation
        feats["vol_z"] = (v - _roll_mean(v, day)) / _roll_std(v, day)
        rng = h - l
        feats["candle_body"] = (c - o) / np.where(rng == 0, np.nan, rng)
        # anchored VWAP distance (resets each UTC day — a different animal
        # from the rolling-window stats above)
        if "dt" in df.columns:
            bar_day = df["dt"].dt.floor("D")
            tp = (h + l + c) / 3.0
            pv = pd.Series(tp * v)
            cum_pv = pv.groupby(bar_day.values).cumsum().to_numpy()
            cum_v = pd.Series(v).groupby(bar_day.values).cumsum().to_numpy()
            vwap = cum_pv / np.where(cum_v == 0, np.nan, cum_v)
            feats["vwap_dist"] = (c - vwap) / vwap
        # order-flow (only if the dataset carries the extra kline columns)
        if "taker_buy_volume" in df.columns:
            tb = df["taker_buy_volume"].to_numpy(dtype=np.float64)
            imb = np.where(v > 0, 2.0 * tb / v - 1.0, np.nan)  # -1..+1
            feats["taker_imb"] = _roll_mean(imb, max(day // 24, 2))
        # perp funding rate (only if a funding file was merged)
        if "funding" in df.columns:
            feats["funding"] = df["funding"].to_numpy(dtype=np.float64)
        # derivatives sentiment (only if a metrics file was merged)
        if "oi" in df.columns:
            oi = df["oi"].to_numpy(dtype=np.float64)
            feats["oi_chg"] = pd.Series(oi).pct_change(day // 4).to_numpy()
        for col in ("top_ls_pos", "global_ls", "taker_ls"):
            if col in df.columns:
                feats[col] = df[col].to_numpy(dtype=np.float64)

    names = list(feats.keys())
    F = np.column_stack([feats[k] for k in names]).astype(np.float32)
    F[~np.isfinite(F)] = np.nan
    return F, names


def train_quantiles(F: np.ndarray, split_idx: int, qs: np.ndarray) -> np.ndarray:
    """Per-feature quantile values computed on the train segment only.

    Returns [n_feat x len(qs)]."""
    out = np.full((F.shape[1], len(qs)), np.nan, dtype=np.float64)
    for j in range(F.shape[1]):
        col = F[:split_idx, j]
        col = col[np.isfinite(col)]
        if len(col) > 100:
            out[j] = np.quantile(col, qs)
    return out


# ---------------------------------------------------------------- metrics ---

def fetch_metrics(symbol: str, since: str, until: str | None, out_dir: Path) -> Path:
    """Download daily `metrics` zips from data.binance.vision and merge to one CSV.

    symbol is the futures symbol, e.g. BTCUSDT. Files exist from 2020-09-01.
    """
    start = date.fromisoformat(since)
    end = date.fromisoformat(until) if until else date.today() - timedelta(days=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, missing = [], 0
    d = start
    while d <= end:
        url = f"{VISION_BASE}/{symbol}/{symbol}-metrics-{d.isoformat()}.zip"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                with zf.open(zf.namelist()[0]) as fh:
                    frames.append(pd.read_csv(fh))
        except Exception:
            missing += 1  # weekends never happen, but gaps/new listings do
        if (d - start).days % 50 == 0:
            print(f"  metrics {d} ({len(frames)} files, {missing} missing)")
        d += timedelta(days=1)
    if not frames:
        raise SystemExit(f"no metrics files found for {symbol} in range")
    m = pd.concat(frames, ignore_index=True)
    m["timestamp"] = pd.to_datetime(m["create_time"]).map(lambda t: int(t.timestamp() * 1000))
    keep = ["timestamp"] + [c for c in METRICS_COLS if c in m.columns]
    m = m[keep].rename(columns=METRICS_COLS).sort_values("timestamp")
    out = out_dir / f"{symbol}_metrics.csv"
    m.to_csv(out, index=False)
    print(f"wrote {out} ({len(m)} rows, {missing} days missing)")
    return out


def merge_metrics(df: pd.DataFrame, metrics_csv: Path) -> pd.DataFrame:
    """As-of merge (backward) so each candle only sees already-published metrics."""
    m = pd.read_csv(metrics_csv).sort_values("timestamp")
    df = df.sort_values("timestamp")
    merged = pd.merge_asof(df, m, on="timestamp", direction="backward",
                           tolerance=6 * 3600 * 1000)
    return merged


def fetch_funding(symbol: str, since: str, out_dir: Path) -> Path:
    """Full perp funding-rate history via ccxt (8h cadence, so it's tiny)."""
    import ccxt

    ex = ccxt.binanceusdm()
    ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= ms or len(batch) < 1000:
            break
        ms = nxt
    if not rows:
        raise SystemExit(f"no funding history for {symbol}")
    f = pd.DataFrame({"timestamp": [r["timestamp"] for r in rows],
                      "funding": [r["fundingRate"] for r in rows]})
    f = f.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{symbol.replace('/', '-').replace(':', '-')}_funding.csv"
    f.to_csv(out, index=False)
    print(f"wrote {out} ({len(f)} rows, {pd.Timestamp(f['timestamp'].iloc[0], unit='ms')} → "
          f"{pd.Timestamp(f['timestamp'].iloc[-1], unit='ms')})")
    return out


def level_drift(F: np.ndarray, names: list[str],
                bar_days: np.ndarray) -> list[tuple[str, float]]:
    """Stationarity guard: per feature, the range of its yearly medians in
    units of its full-sample IQR. Scores >~1 mean quantile thresholds anchored
    to long history can go permanently out of reach — discovered when ETH's
    top_ls_pos level tripled and a capitulation rule went dead for two years."""
    years = bar_days.astype("datetime64[Y]")
    out = []
    for j, name in enumerate(names):
        x = F[:, j].astype(np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 1000:
            out.append((name, float("nan")))
            continue
        q25, q75 = np.nanpercentile(x[ok], [25, 75])
        iqr = max(q75 - q25, 1e-12)
        meds = []
        for y in np.unique(years[ok]):
            m = np.nanmedian(x[ok & (years == y)])
            if np.isfinite(m):
                meds.append(m)
        score = (max(meds) - min(meds)) / iqr if len(meds) >= 2 else float("nan")
        out.append((name, float(score)))
    return out


def merge_funding(df: pd.DataFrame, funding_csv: Path) -> pd.DataFrame:
    """Backward as-of merge; each candle sees the last settled funding rate.

    Also writes `fund_pay`: the settled rate on the first bar at/after each
    settlement timestamp (0 elsewhere). The engine charges that rate to any
    position open on that bar — longs pay when funding is positive, shorts
    receive, and vice versa."""
    f = pd.read_csv(funding_csv).sort_values("timestamp")
    df = df.sort_values("timestamp")
    out = pd.merge_asof(df, f, on="timestamp", direction="backward",
                        tolerance=9 * 3600 * 1000)
    ts = out["timestamp"].to_numpy(np.int64)
    fts = f["timestamp"].to_numpy(np.int64)
    rates = f["funding"].to_numpy(np.float64)
    keep = fts >= ts[0]
    pos = np.searchsorted(ts, fts[keep])
    pay = np.zeros(len(out))
    ok = pos < len(out)
    pay[pos[ok]] = rates[keep][ok]
    out["fund_pay"] = pay
    return out
