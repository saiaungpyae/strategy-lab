"""Standalone probe of the top-trader-fade rule family the swarm nominated.

Hypothesis (from the 2026-07-16 calibration wave): "short when the top-trader
long/short position ratio is high" (fading crowded longs) recurred in the lucky
tail of all three independent seeds. This probe tests the family properly:

- a fixed config grid (entry quantile x stop x take-profit x hold), SHORT fade
  plus the mirrored LONG variant as a symmetry control,
- run through the same engine as the swarm (maker entries, stops exit taker),
- judged by the lab's protocol: config selected on TRAIN only, headline is its
  TEST performance; family-wide OOS positivity and per-year returns reported
  so one lucky window can't hide.

Deliberately 15m-only (the wave's stable timeframe).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, features, genome, recap
from .evolve import _alloc

ENTRY_Q = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
STOP_ATR = [2.0, 3.0, 4.0]
TP_RR = [1.5, 3.0, float("nan")]
HOLD_DAYS = [1, 3]

TF = {"15m": {"file_arg": "file15", "bars_per_day": 96, "tf_code": 1},
      "1h": {"file_arg": "file1h", "bars_per_day": 24, "tf_code": 1}}


def build_grid(feature_names: list[str], bars_per_day: int,
               feature: str = "top_ls_pos"):
    fi = feature_names.index(feature)
    rows = []
    for q in ENTRY_Q:
        for stop in STOP_ATR:
            for rr in TP_RR:
                for hold in [d * bars_per_day for d in HOLD_DAYS]:
                    # SHORT fade: ratio above quantile -> crowded longs -> short
                    rows.append((q, 1, -1, stop, rr, hold, "fade_short"))
                    # LONG mirror: ratio below (1-q) -> crowded shorts -> long
                    rows.append((1.0 - q, -1, 1, stop, rr, hold, "mirror_long"))
    n = len(rows)
    g = genome.Genomes(seed=0, feature_names=feature_names)
    g.is_control = np.zeros(n, dtype=bool)
    g.n_rules = np.ones(n, dtype=np.int8)
    g.rule_feat = np.full((n, genome.MAX_RULES), fi, dtype=np.int16)
    g.rule_op = np.ones((n, genome.MAX_RULES), dtype=np.int8)
    g.rule_q = np.full((n, genome.MAX_RULES), 0.5, dtype=np.float32)
    g.rule_dir = np.ones((n, genome.MAX_RULES), dtype=np.int8)
    for i, (q, op, d, *_rest) in enumerate(rows):
        g.rule_q[i, 0] = q
        g.rule_op[i, 0] = op
        g.rule_dir[i, 0] = d
    g.ctrl_rate = np.full(n, 0.01, dtype=np.float32)
    g.tf = np.ones(n, dtype=np.int8)               # 15m
    g.dir_bias = np.zeros(n, dtype=np.int8)
    g.risk_pct = np.full(n, 0.005, dtype=np.float32)
    g.stop_atr = np.array([r[3] for r in rows], dtype=np.float32)
    g.tp_rr = np.array([r[4] for r in rows], dtype=np.float32)
    g.max_hold = np.array([r[5] for r in rows], dtype=np.int32)
    g.maker_off = np.full(n, 0.25, dtype=np.float32)
    g.order_ttl = np.full(n, 4, dtype=np.int16)
    g.session = np.zeros(n, dtype=np.int8)
    g.loss_react = np.zeros(n, dtype=np.int8)
    g.cooldown = np.zeros(n, dtype=np.int16)
    g.revenge = np.ones(n, dtype=np.float32)
    g.reentry_gap = np.full(n, 4, dtype=np.int16)
    labels = pd.DataFrame(rows, columns=["entry_q", "op", "dir", "stop_atr",
                                         "tp_rr", "hold_bars", "family"])
    return g, labels


def cmd_probe(args) -> None:
    from .run import _load, _market

    tf = TF[args.tf]
    df = _load(getattr(args, tf["file_arg"]), args.since, args.metrics, args.funding)
    all_days = df["dt"].dt.tz_convert(None).dt.floor("D") \
                       .to_numpy().astype("datetime64[D]")
    all_days = np.unique(all_days)
    ts = df["timestamp"].to_numpy(np.int64)
    split_ts = int(ts[0] + args.split * (ts[-1] - ts[0]))
    split_day = int(np.searchsorted(
        all_days, np.datetime64(pd.Timestamp(split_ts, unit="ms"), "D")))
    qs = np.linspace(0.02, 0.98, 49)
    mkt, names = _market(df, tf["tf_code"], tf["bars_per_day"], split_ts, all_days, qs)
    feature = getattr(args, "feature", "top_ls_pos")
    if feature not in names:
        raise SystemExit(f"{feature} missing — pass --metrics/--funding")

    bar_days = df["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    drift = dict(features.level_drift(mkt["F"], names, bar_days))[feature]
    drift_warn = ""
    if np.isfinite(drift) and drift > 1.0:
        drift_warn = (f"⚠ {feature} level-drift score {drift:.2f} — yearly medians span "
                      f">1 IQR, so quantile thresholds vs long history may sit "
                      f"permanently out of reach on parts of this span (ETH failure mode)")
        print(drift_warn)

    g, labels = build_grid(names, tf["bars_per_day"], feature)
    cfg = {"taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
           "start_capital": args.start_capital, "ruin_frac": args.ruin, "seed": 0}
    out = _alloc(g.n, len(all_days))
    engine.run_cohort(mkt, g, np.arange(g.n), cfg, out)
    sm = recap.seg_metrics(out["daily"], split_day, args.start_capital)

    res = labels.copy()
    for k in ("sharpe_a", "sharpe_b", "ret_a", "ret_b", "maxdd_b", "final_mult"):
        res[k] = np.round(sm[k], 4)
    res["trades"] = out["trades_a"] + out["trades_b"]
    res["trades_b"] = out["trades_b"]
    res["dead"] = out["dead"]

    # per-year returns from the daily equity matrix
    d = sm["daily_filled"].astype(np.float64)
    years = pd.Series(all_days).dt.year.to_numpy()
    yr_cols = {}
    for y in np.unique(years):
        ix = np.flatnonzero(years == y)
        yr_cols[int(y)] = d[:, ix[-1]] / d[:, ix[0]] - 1.0
    c = mkt["c"]
    bh_year = {}
    for y in np.unique(years):
        sel = df["dt"].dt.year.to_numpy() == y
        px = c[sel]
        bh_year[int(y)] = px[-1] / px[0] - 1.0

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(run_dir / f"{feature}_fade_grid_{args.tf}.csv", index=False)

    # honest selection: best on TRAIN (min activity), judged on TEST
    ok = (res["trades"] >= 30) & res["sharpe_a"].notna()
    lines = [f"# {feature}-fade probe ({args.tf}) — {datetime.now().date()}",
             "",
             f"Span {all_days[0]} → {all_days[-1]}, split {all_days[split_day]} "
             f"(train {args.split:.0%}). {args.tf} maker entries (0 fee + "
             f"{args.maker_bps} bp edge), stops/time exits pay taker "
             f"{args.taker_bps} bps. {len(res)} configs "
             f"({len(res)//2} per family), risk 0.5%/trade.", ""]
    if drift_warn:
        lines += [drift_warn, ""]
    for fam in ("fade_short", "mirror_long"):
        sub = res[(res["family"] == fam) & ok]
        lines.append(f"## {fam} ({len(sub)} active configs)")
        if not len(sub):
            lines.append("no active configs\n")
            continue
        best = sub.sort_values("sharpe_a", ascending=False).iloc[0]
        pos_b = (sub["sharpe_b"] > 0).mean() * 100
        lines += [
            f"- family median: train S {sub['sharpe_a'].median():.2f} → "
            f"test S {sub['sharpe_b'].median():.2f}; **{pos_b:.0f}% of configs "
            f"positive on test**",
            f"- train-selected config (q={best['entry_q']:.2f}, stop "
            f"{best['stop_atr']}xATR, tp {best['tp_rr']}, hold {best['hold_bars']} bars): "
            f"train S {best['sharpe_a']:.2f} / ret {best['ret_a']*100:.1f}% → "
            f"**test S {best['sharpe_b']:.2f} / ret {best['ret_b']*100:.1f}%**, "
            f"maxDD(test) {best['maxdd_b']*100:.1f}%, {int(best['trades'])} trades",
        ]
        bi = int(best.name)
        yr = " | ".join(f"{y}: {yr_cols[y][bi]*100:+.1f}% (B&H {bh_year[y]*100:+.1f}%)"
                        for y in sorted(yr_cols))
        lines += [f"- per-year (train-selected config): {yr}", ""]
    (run_dir / f"{feature}_fade_probe_{args.tf}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\ngrid csv + report -> {run_dir}/{feature}_fade_probe_{args.tf}.md")
