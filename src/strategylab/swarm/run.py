"""sl-swarm — run a bot swarm, fetch derivatives metrics, rebuild recaps.

    sl-swarm run           --bots 5000 --since 2024-01-01
    sl-swarm fetch-metrics --symbol BTCUSDT --since 2024-01-01
    sl-swarm report        --run reports/swarm/<run_id>

Artifacts land in reports/swarm/<run_id>/:
    config.json  genomes.csv  results.csv  daily_equity.npz  recap.json  progress.json
The viewer's /swarm dashboard reads these; the server never runs a simulation.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, features, genome, recap


def _load(path: str, since: str | None, metrics: str | None,
          funding: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime"], utc=True, format="mixed")
    if since:
        df = df[df["dt"] >= pd.Timestamp(since, tz="UTC")].reset_index(drop=True)
    if metrics:
        df = features.merge_metrics(df, Path(metrics))
    if funding:
        df = features.merge_funding(df, Path(funding))
    return df.reset_index(drop=True)


def _market(df: pd.DataFrame, tf_code: int, bars_per_day: int, split_ts: int,
            days_arr: np.ndarray, qs: np.ndarray):
    F, names = features.compute_features(df, bars_per_day)
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    seg_b = ts >= split_ts
    split_idx = int(np.searchsorted(ts, split_ts))
    Q = features.train_quantiles(F, split_idx, qs)
    bar_days = df["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    return {
        "tf_code": tf_code,
        "o": df["open"].to_numpy(dtype=np.float64),
        "h": df["high"].to_numpy(dtype=np.float64),
        "l": df["low"].to_numpy(dtype=np.float64),
        "c": df["close"].to_numpy(dtype=np.float64),
        "atr": features.atr(df),
        "hour": df["dt"].dt.hour.to_numpy(dtype=np.int64),
        "day_pos": np.searchsorted(days_arr, bar_days).astype(np.int64),
        "seg_b": seg_b, "F": F, "Q": Q, "qs": qs,
    }, names


def cmd_run(args) -> None:
    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{args.seed}"
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def prog(stage: str, frac: float) -> None:
        (run_dir / "progress.json").write_text(json.dumps(
            {"stage": stage, "frac": round(frac, 4), "elapsed_s": round(time.time() - t0, 1)}))

    prog("loading", 0.0)
    df5 = _load(args.file5, args.since, args.metrics, args.funding)
    df15 = _load(args.file15, args.since, args.metrics, args.funding)

    ts_min = int(min(df5["timestamp"].iloc[0], df15["timestamp"].iloc[0]))
    ts_max = int(max(df5["timestamp"].iloc[-1], df15["timestamp"].iloc[-1]))
    split_ts = int(ts_min + args.split * (ts_max - ts_min))

    all_days = np.union1d(
        df5["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]"),
        df15["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]"))
    split_day = int(np.searchsorted(all_days,
                    np.datetime64(pd.Timestamp(split_ts, unit="ms"), "D")))
    qs = np.linspace(0.02, 0.98, 49)

    prog("features", 0.05)
    mkt5, names5 = _market(df5, 0, 288, split_ts, all_days, qs)
    mkt15, names15 = _market(df15, 1, 96, split_ts, all_days, qs)
    assert names5 == names15, "feature sets differ between timeframes"

    g = genome.sample(args.bots, len(names5), names5, args.control_frac, args.seed,
                      maker_only=args.maker_only)
    cfg = {
        "run_id": run_id, "seed": args.seed, "bots": args.bots,
        "control_frac": args.control_frac, "split": args.split,
        "split_ts": split_ts, "split_date": str(all_days[split_day]),
        "since": args.since, "file5": args.file5, "file15": args.file15,
        "metrics": args.metrics, "funding": args.funding,
        "maker_only": bool(args.maker_only), "feature_names": names5,
        "taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
        "start_capital": args.start_capital, "ruin_frac": args.ruin,
        "span": [str(all_days[0]), str(all_days[-1])], "n_days": int(len(all_days)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    gdf = genome.to_frame(g)
    gdf.to_csv(run_dir / "genomes.csv", index=False)

    n = args.bots
    out = {"daily": np.full((n, len(all_days)), np.nan, dtype=np.float32),
           "final_eq": np.zeros(n), "dead": np.zeros(n, dtype=bool),
           "death_day": np.full(n, -1, dtype=np.int64)}
    for k in ("trades_a", "trades_b", "wins_a", "wins_b", "expo_a", "expo_b",
              "bars_a", "bars_b"):
        out[k] = np.zeros(n, dtype=np.int64)

    idx5, idx15 = np.flatnonzero(g.tf == 0), np.flatnonzero(g.tf == 1)
    print(f"[{run_id}] {n} bots ({len(idx5)} on 5m, {len(idx15)} on 15m, "
          f"{int(g.is_control.sum())} control) | {cfg['span'][0]} → {cfg['span'][1]} "
          f"| split {cfg['split_date']}")
    engine.run_cohort(mkt5, g, idx5, cfg, out,
                      progress=lambda f: prog("sim 5m", 0.1 + 0.55 * f))
    print(f"  5m cohort done ({time.time() - t0:.0f}s)")
    engine.run_cohort(mkt15, g, idx15, cfg, out,
                      progress=lambda f: prog("sim 15m", 0.65 + 0.25 * f))
    print(f"  15m cohort done ({time.time() - t0:.0f}s)")

    prog("recap", 0.92)
    sm = recap.seg_metrics(out["daily"], split_day, args.start_capital)
    res = pd.DataFrame({
        "bot_id": np.arange(n), "is_control": g.is_control,
        "trades_a": out["trades_a"], "trades_b": out["trades_b"],
        "wins_a": out["wins_a"], "wins_b": out["wins_b"],
        "expo_a": out["expo_a"], "expo_b": out["expo_b"],
        "bars_a": out["bars_a"], "bars_b": out["bars_b"],
        "ret_a": sm["ret_a"], "ret_b": sm["ret_b"],
        "sharpe_a": sm["sharpe_a"], "sharpe_b": sm["sharpe_b"],
        "maxdd_b": sm["maxdd_b"], "maxdd_all": sm["maxdd_all"],
        "final_mult": sm["final_mult"],
        "dead": out["dead"], "death_day": out["death_day"],
    })
    res.to_csv(run_dir / "results.csv", index=False)
    np.savez_compressed(run_dir / "daily_equity.npz",
                        daily=sm["daily_filled"], days=all_days.astype(str),
                        split_day=split_day)

    c5 = mkt5["c"]
    bnh = float(c5[-1] / c5[engine.WARMUP])
    day_close = np.full(len(all_days), np.nan)
    day_close[mkt5["day_pos"]] = c5
    rec = recap.build_recap(res, gdf, sm["daily_filled"], list(all_days.astype(str)),
                            split_day, cfg, bnh, day_close)
    (run_dir / "recap.json").write_text(json.dumps(rec))
    prog("done", 1.0)

    t = rec["tiles"]
    print(f"  recap: alive {t['alive_pct']:.1f}% | above water {t['above_water_pct']:.1f}% "
          f"| median x{t['median_final_mult']:.3f} | B&H x{t['bnh_mult']:.3f}")
    print(f"  luck yardstick (control sharpe_B): p95={t['yardstick_sharpe_p95']} "
          f"p99={t['yardstick_sharpe_p99']}")
    print(f"  persistence rank-corr: pattern={t['rank_corr_pattern']} "
          f"control={t['rank_corr_control']} gap={t['rank_corr_gap']} "
          f"(control persistence = cost drag; skill = the gap)")
    print(f"  artifacts: {run_dir}  ({time.time() - t0:.0f}s total)")


def cmd_fetch_metrics(args) -> None:
    features.fetch_metrics(args.symbol, args.since, args.until, Path(args.out))


def cmd_report(args) -> None:
    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    res = pd.read_csv(run_dir / "results.csv")
    gdf = pd.read_csv(run_dir / "genomes.csv")
    z = np.load(run_dir / "daily_equity.npz", allow_pickle=False)
    daily, days, split_day = z["daily"], list(z["days"]), int(z["split_day"])
    df5 = _load(cfg["file5"], cfg.get("since"), None)
    c5 = df5["close"].to_numpy()
    bnh = float(c5[-1] / c5[engine.WARMUP])
    days_arr = np.array(days, dtype="datetime64[D]")
    bar_days = df5["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    day_close = np.full(len(days_arr), np.nan)
    day_close[np.searchsorted(days_arr, bar_days)] = c5
    rec = recap.build_recap(res, gdf, daily, days, split_day, cfg, bnh, day_close)
    (run_dir / "recap.json").write_text(json.dumps(rec))
    print(f"rebuilt {run_dir / 'recap.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="sl-swarm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate a swarm")
    r.add_argument("--file5", default="data/binance_BTC-USDT_5m.csv")
    r.add_argument("--file15", default="data/binance_BTC-USDT_15m.csv")
    r.add_argument("--metrics", default=None, help="metrics CSV from fetch-metrics")
    r.add_argument("--funding", default=None, help="funding CSV from fetch-funding")
    r.add_argument("--maker-only", action="store_true",
                   help="all entries rest as limits (stops still exit as taker)")
    r.add_argument("--bots", type=int, default=2000)
    r.add_argument("--control-frac", type=float, default=0.10)
    r.add_argument("--split", type=float, default=0.70, help="train fraction of the span")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--since", default=None, help="trim data start, YYYY-MM-DD")
    r.add_argument("--taker-bps", type=float, default=5.0)
    r.add_argument("--maker-bps", type=float, default=1.0, help="adverse-selection edge")
    r.add_argument("--start-capital", type=float, default=10_000.0)
    r.add_argument("--ruin", type=float, default=0.30)
    r.add_argument("--out", default="reports/swarm")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("fetch-metrics", help="download Binance Vision derivatives metrics")
    f.add_argument("--symbol", default="BTCUSDT")
    f.add_argument("--since", required=True)
    f.add_argument("--until", default=None)
    f.add_argument("--out", default="data/metrics")
    f.set_defaults(func=cmd_fetch_metrics)

    fu = sub.add_parser("fetch-funding", help="download perp funding-rate history (ccxt)")
    fu.add_argument("--symbol", default="BTC/USDT:USDT")
    fu.add_argument("--since", required=True)
    fu.add_argument("--out", default="data/metrics")
    fu.set_defaults(func=lambda a: features.fetch_funding(a.symbol, a.since, Path(a.out)))

    p = sub.add_parser("report", help="rebuild recap.json for an existing run")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_report)

    e = sub.add_parser("evolve", help="walk-forward generational evolution + placebo lineage")
    e.add_argument("--file5", default="data/binance_BTC-USDT_5m.csv")
    e.add_argument("--file15", default="data/binance_BTC-USDT_15m.csv")
    e.add_argument("--metrics", default=None)
    e.add_argument("--funding", default=None)
    e.add_argument("--maker-only", action="store_true")
    e.add_argument("--bots", type=int, default=1500, help="population per lineage")
    e.add_argument("--gens", type=int, default=6, help="fitness windows / generations")
    e.add_argument("--test-frac", type=float, default=0.20,
                   help="final span fraction reserved, untouched by selection")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--since", default=None)
    e.add_argument("--taker-bps", type=float, default=5.0)
    e.add_argument("--maker-bps", type=float, default=1.0)
    e.add_argument("--start-capital", type=float, default=10_000.0)
    e.add_argument("--ruin", type=float, default=0.30)
    e.add_argument("--out", default="reports/swarm")

    def _evolve(a):
        from . import evolve as _ev
        _ev.cmd_evolve(a)
    e.set_defaults(func=_evolve)

    pr = sub.add_parser("probe", help="standalone grid test of the top-trader-fade family")
    pr.add_argument("--file15", default="data/binance_BTC-USDT_15m.csv")
    pr.add_argument("--metrics", default="data/metrics/BTCUSDT_metrics.csv")
    pr.add_argument("--funding", default="data/metrics/BTC-USDT-USDT_funding.csv")
    pr.add_argument("--since", default="2021-01-06")
    pr.add_argument("--split", type=float, default=0.70)
    pr.add_argument("--taker-bps", type=float, default=5.0)
    pr.add_argument("--maker-bps", type=float, default=1.0)
    pr.add_argument("--start-capital", type=float, default=10_000.0)
    pr.add_argument("--ruin", type=float, default=0.30)
    pr.add_argument("--out", default="reports")

    def _probe(a):
        from . import probe as _pb
        _pb.cmd_probe(a)
    pr.set_defaults(func=_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
