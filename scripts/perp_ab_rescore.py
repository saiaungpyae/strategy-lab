#!/usr/bin/env python
"""A/B re-score: the paper roster's 40 HOF bots, spot tape vs perp tape.

Same procedure on both tapes (so any protocol drift cancels): features over
the full tape since 2021, rule thresholds from all history before the run's
reserved-test start, engine cohort replay, test-span Sharpe from daily
equity. The only variable is which candles the bots see — Binance spot
(what they were evolved on) vs the USDT-M perp (what a live account would
actually trade).

Run:  .venv/bin/python scripts/perp_ab_rescore.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from strategylab.data import paths as datapaths
from strategylab.swarm import engine, features, recap, stress
from strategylab.swarm.evolve import _alloc
from strategylab.swarm.run import _load

QS = np.linspace(0.02, 0.98, 49)


def rescore_tape(tape: Path, metrics: str, funding: str, bots: list[dict],
                 test_start_ms: int, t_end_ms: int) -> list[dict]:
    df = _load(str(tape), "2021-01-01", metrics, funding)
    df = df[df["timestamp"] <= t_end_ms].reset_index(drop=True)
    F, names = features.compute_features(df, 96)
    ts = df["timestamp"].to_numpy(np.int64)
    bar_days = df["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    all_days = np.unique(bar_days)
    day_pos = np.searchsorted(all_days, bar_days).astype(np.int64)
    i_test = int(np.searchsorted(ts, test_start_ms))
    Q = features.train_quantiles(F, i_test, QS)
    split_day = int(np.searchsorted(
        all_days, bar_days[min(i_test, len(bar_days) - 1)]))
    w = {"tf_code": 1,
         "o": df["open"].to_numpy(np.float64), "h": df["high"].to_numpy(np.float64),
         "l": df["low"].to_numpy(np.float64), "c": df["close"].to_numpy(np.float64),
         "atr": features.atr(df), "hour": df["dt"].dt.hour.to_numpy(np.int64),
         "day_pos": day_pos, "seg_b": ts >= test_start_ms, "F": F, "Q": Q,
         "qs": QS,
         "fund": (df["fund_pay"].to_numpy(np.float64)
                  if "fund_pay" in df.columns else np.zeros(len(df)))}
    out_rows = []
    for bot in bots:
        g = stress._genome_from_record(bot["rec"], names)
        cfg = dict(bot["cfg"])
        out = _alloc(1, len(all_days))
        engine.run_cohort(w, g, np.arange(1), cfg, out)
        sm = recap.seg_metrics(out["daily"], split_day, cfg["start_capital"])
        out_rows.append({
            "label": bot["label"],
            "sharpe": round(float(sm["sharpe_b"][0]), 3),
            "ret_pct": round(float(sm["ret_b"][0]) * 100, 2),
            "trades": int(out["trades_b"][0]),
            "dead": bool(out["dead"][0]),
        })
    return out_rows


def main() -> None:
    roster = json.loads(Path("reports/paper/roster.json").read_text())
    by_pair: dict[str, list[dict]] = {}
    for b in roster["bots"]:
        by_pair.setdefault(b["pair"], []).append(b)

    rows = []
    for pair in sorted(by_pair):
        bots = by_pair[pair]
        e = json.loads((Path("reports/swarm") / bots[0]["run_id"]
                        / "evolution.json").read_text())
        test_ms = int(np.datetime64(e["test_start"]).astype("datetime64[ms]")
                      .astype(np.int64))
        pdir = f"{pair}-USDT"
        spot = Path(datapaths.default_candles("15m", pdir))
        perp = datapaths.candles(f"{pair}/USDT:USDT", "15m", "binanceusdm")
        metrics = datapaths.default_metrics(pdir)
        funding = datapaths.default_funding(pdir)
        import pandas as pd
        t_end = min(int(pd.read_csv(p, usecols=["timestamp"])["timestamp"].iloc[-1])
                    for p in (spot, perp))
        a = rescore_tape(spot, metrics, funding, bots, test_ms, t_end)
        b = rescore_tape(perp, metrics, funding, bots, test_ms, t_end)
        for bot, ra, rb in zip(bots, a, b):
            rows.append({"pair": pair, "label": bot["label"],
                         "battery_S": bot.get("test_sharpe"),
                         "spot": ra, "perp": rb,
                         "dS": round(rb["sharpe"] - ra["sharpe"], 3)})
            print(f"  {bot['label']:10} spot S {ra['sharpe']:6.2f} "
                  f"({ra['trades']:4d} tr) | perp S {rb['sharpe']:6.2f} "
                  f"({rb['trades']:4d} tr) | dS {rb['sharpe']-ra['sharpe']:+.2f}")

    s = np.array([r["spot"]["sharpe"] for r in rows])
    p = np.array([r["perp"]["sharpe"] for r in rows])
    finite = np.isfinite(s) & np.isfinite(p)
    s, p = s[finite], p[finite]
    def _rank(x: np.ndarray) -> np.ndarray:
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return r

    rho = (float(np.corrcoef(_rank(s), _rank(p))[0, 1])
           if len(s) > 2 else float("nan"))
    flips = int(((s > 0.3) & (p < 0)).sum() + ((s < -0.3) & (p > 0)).sum())
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_bots": len(rows), "mean_spot_S": round(float(s.mean()), 3),
        "mean_perp_S": round(float(p.mean()), 3),
        "mean_dS": round(float((p - s).mean()), 3),
        "median_abs_dS": round(float(np.median(np.abs(p - s))), 3),
        "spearman_rank_corr": round(float(rho), 3),
        "sign_flips_meaningful": flips,
        "rows": rows,
    }
    out = Path("reports/perp-ab-rescore.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nmean S: spot {summary['mean_spot_S']} -> perp "
          f"{summary['mean_perp_S']} (mean dS {summary['mean_dS']:+})")
    print(f"median |dS| {summary['median_abs_dS']} | rank corr {rho:.3f} | "
          f"meaningful sign flips {flips}/{len(rows)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
