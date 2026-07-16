"""Evolution v2 — walk-forward generational selection with a placebo lineage.

Protocol (the anti-overfit rules are structural, not optional):
- The final `test_frac` of the span is RESERVED — no generation ever sees it.
- The remaining span is split into `gens` sequential fitness windows. Gen g is
  evaluated on window g only; its genomes were bred from parents selected on
  window g-1, so **every genome is scored exclusively on data that played no
  role in creating it**.
- Rule thresholds resolve against expanding pre-window history only.
- A PLACEBO lineage runs the identical breed/mutate/immigrate loop but selects
  parents at random. Whatever the placebo achieves on the reserved test span
  is what drift + mutation + multiple generations achieve without selection —
  evolution's claim to skill is only the amount it beats its placebo.
- A fresh random population on the test span provides the plain-luck baseline.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from . import engine, features, genome

ELITE_FRAC = 0.05
PARENT_FRAC = 0.25
FRESH_FRAC = 0.10
HOF_PER_GEN = 10


def _alloc(n: int, n_days: int) -> dict:
    out = {"daily": np.full((n, n_days), np.nan, dtype=np.float32),
           "final_eq": np.zeros(n), "dead": np.zeros(n, dtype=bool),
           "death_day": np.full(n, -1, dtype=np.int64)}
    for k in ("trades_a", "trades_b", "wins_a", "wins_b", "expo_a", "expo_b",
              "bars_a", "bars_b"):
        out[k] = np.zeros(n, dtype=np.int64)
    return out


def _window(mkt, ts, t0, t1, qs):
    """Slice a full-market dict to [t0, t1) with warmup margin; thresholds
    come from expanding history strictly before t0. The very first window has
    no pre-history, so it bootstraps thresholds from its own span — safe there
    because gen-0 genomes are random, not selected on that window."""
    i0 = int(np.searchsorted(ts, t0))
    i1 = int(np.searchsorted(ts, t1))
    m0 = max(0, i0 - engine.WARMUP)
    Q = features.train_quantiles(mkt["F"], i0 if i0 >= 5000 else i1, qs)
    w = {k: mkt[k][m0:i1] for k in ("o", "h", "l", "c", "atr", "hour", "day_pos", "F")}
    w.update({"seg_b": np.ones(i1 - m0, dtype=bool), "Q": Q, "qs": qs,
              "tf_code": mkt["tf_code"]})
    return w, i0, i1


def _eval_cohort(w, pop, idx, cfg, n_days):
    """Worker: one (population, timeframe) cohort. Identical inputs/seeds to
    the serial path, so results are bit-for-bit the same regardless of jobs."""
    out = _alloc(pop.n, n_days)
    engine.run_cohort(w, pop, idx, cfg, out)
    return out


def _evaluate_many(pops, w5, w15, cfg, n_days, pool):
    """Evaluate independent populations on the same window pair. Each pop
    splits into its 5m and 15m cohorts; all cohorts run concurrently when a
    process pool is given (pool=None falls back to serial)."""
    tasks = [(pop, np.flatnonzero(pop.tf == tf), w)
             for pop in pops for tf, w in ((0, w5), (1, w15))]
    if pool is None:
        rets = [_eval_cohort(w, pop, idx, cfg, n_days) for pop, idx, w in tasks]
    else:
        futs = [pool.submit(_eval_cohort, w, pop, idx, cfg, n_days)
                for pop, idx, w in tasks]
        rets = [f.result() for f in futs]
    outs = []
    for j, pop in enumerate(pops):
        out = _alloc(pop.n, n_days)
        for (_, idx, _), ret in zip(tasks[2 * j:2 * j + 2], rets[2 * j:2 * j + 2]):
            for k in out:
                out[k][idx] = ret[k][idx]
        outs.append(out)
    return outs


def _fill_eq(out, d0, d1, start_cap):
    eq = out["daily"][:, d0:d1].astype(np.float64)
    bad = ~np.isfinite(eq[:, 0])
    eq[bad, 0] = start_cap
    for j in range(1, eq.shape[1]):
        col = eq[:, j]
        col[~np.isfinite(col)] = eq[:, j - 1][~np.isfinite(col)]
    return eq


def _sharpe(out, d0, d1, start_cap):
    eq = _fill_eq(out, d0, d1, start_cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(eq, axis=1) / np.where(eq[:, :-1] == 0, np.nan, eq[:, :-1])
        mu, sd = np.nanmean(r, axis=1), np.nanstd(r, axis=1)
        return np.where(sd > 0, mu / sd * np.sqrt(365.0), np.nan)


def _fitness(out, d0, d1, start_cap, mode="sharpe", min_expo=0.15):
    """'sharpe': risk-adjusted (rewards sheltering when everything loses).
    'return': window return with a hard participation floor — a bot below
    `min_expo` time-in-market takes a penalty up to -25 return-points, so
    hiding in cash stops being a winning strategy."""
    sh = _sharpe(out, d0, d1, start_cap)
    trades = out["trades_a"] + out["trades_b"]
    if mode == "sharpe":
        fit = np.where(np.isfinite(sh), sh, -9.0)
        fit = fit - 2.0 * (trades < 10) - 4.0 * out["dead"]
    else:
        eq = _fill_eq(out, d0, d1, start_cap)
        ret = (eq[:, -1] / eq[:, 0] - 1.0) * 100.0
        bars = np.maximum(out["bars_a"] + out["bars_b"], 1)
        expo = (out["expo_a"] + out["expo_b"]) / bars
        shortfall = np.clip((min_expo - expo) / max(min_expo, 1e-9), 0.0, 1.0)
        fit = ret - 25.0 * shortfall - 40.0 * out["dead"]
    return fit, sh, trades


def _next_gen(pop, fit, n_feat, rng, maker_only, random_selection, fresh_seed):
    n = pop.n
    order = rng.permutation(n) if random_selection else np.argsort(-fit)
    elites = genome.subset(pop, order[:max(1, int(n * ELITE_FRAC))])
    parents = order[:max(2, int(n * PARENT_FRAC))]
    n_fresh = int(n * FRESH_FRAC)
    n_off = n - elites.n - n_fresh
    offspring = genome.breed(pop, parents, n_off, n_feat, rng, maker_only)
    fresh = genome.sample(n_fresh, n_feat, pop.feature_names, 0.0, fresh_seed,
                          maker_only=maker_only)
    return genome.concat([elites, offspring, fresh])


def cmd_evolve(args) -> None:
    from .run import _load, _market  # lazy: avoids circular import

    t_start = time.time()
    run_id = "evo-" + datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{args.seed}"
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df5 = _load(args.file5, args.since, args.metrics, args.funding)
    df15 = _load(args.file15, args.since, args.metrics, args.funding)
    ts5 = df5["timestamp"].to_numpy(np.int64)
    ts15 = df15["timestamp"].to_numpy(np.int64)
    ts_min, ts_max = int(ts5[0]), int(ts5[-1])

    all_days = np.union1d(
        df5["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]"),
        df15["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]"))
    qs = np.linspace(0.02, 0.98, 49)
    mkt5, names = _market(df5, 0, 288, ts_max + 1, all_days, qs)   # split unused here
    mkt15, _ = _market(df15, 1, 96, ts_max + 1, all_days, qs)
    n_feat = len(names)

    test_t0 = int(ts_min + (1.0 - args.test_frac) * (ts_max - ts_min))
    bounds = np.linspace(ts_min, test_t0, args.gens + 1).astype(np.int64)
    cfg = {"taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
           "start_capital": args.start_capital, "ruin_frac": args.ruin,
           "seed": args.seed}

    # 8 = max concurrent cohorts (final test: 4 pops x 2 timeframes);
    # threads suffice when the numba kernel (nogil) is active
    n_jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1)
    if n_jobs <= 1:
        pool = None
    elif engine.numba is not None:
        pool = ThreadPoolExecutor(max_workers=n_jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs)

    evolved = genome.sample(args.bots, n_feat, names, 0.0, args.seed,
                            maker_only=args.maker_only)
    placebo = genome.sample(args.bots, n_feat, names, 0.0, args.seed + 1000,
                            maker_only=args.maker_only)
    rng_e = np.random.default_rng(args.seed * 31 + 1)
    rng_p = np.random.default_rng(args.seed * 31 + 2)
    hof, gen_stats = [], []

    print(f"[{run_id}] {args.bots} bots/lineage, {args.gens} fitness windows, "
          f"{n_jobs} worker(s), test reserved from "
          f"{np.datetime64(int(test_t0), 'ms').astype('datetime64[D]')}")
    for gen in range(args.gens):
        t0, t1 = int(bounds[gen]), int(bounds[gen + 1])
        w5, i0_5, _ = _window(mkt5, ts5, t0, t1, qs)
        w15, _, _ = _window(mkt15, ts15, t0, t1, qs)
        d0 = int(mkt5["day_pos"][i0_5])
        d1 = int(w5["day_pos"][-1]) + 1
        row = {"gen": gen, "window": [str(np.datetime64(t0, 'ms').astype('datetime64[D]')),
                                      str(np.datetime64(t1, 'ms').astype('datetime64[D]'))]}
        fits = {}
        lineages = (("evolved", evolved), ("placebo", placebo))
        outs = _evaluate_many([p for _, p in lineages], w5, w15, cfg,
                              len(all_days), pool)
        for (name, _), out in zip(lineages, outs):
            fit, sh, trades = _fitness(out, d0, d1, args.start_capital,
                                       args.fitness, args.min_expo)
            fits[name] = fit
            row[name] = {"median_sharpe": round(float(np.nanmedian(sh)), 3),
                         "p90_sharpe": round(float(np.nanpercentile(sh, 90)), 3),
                         "dead_pct": round(float(out["dead"].mean() * 100), 1),
                         "median_trades": int(np.median(trades))}
        hof_idx = np.argsort(-fits["evolved"])[:HOF_PER_GEN]
        hof.append(genome.subset(evolved, hof_idx))
        gen_stats.append(row)
        print(f"  gen {gen} [{row['window'][0]}→{row['window'][1]}] "
              f"evolved S~{row['evolved']['median_sharpe']} "
              f"(p90 {row['evolved']['p90_sharpe']}) vs placebo "
              f"S~{row['placebo']['median_sharpe']} (p90 {row['placebo']['p90_sharpe']}) "
              f"({time.time() - t_start:.0f}s)")
        if gen < args.gens - 1:
            evolved = _next_gen(evolved, fits["evolved"], n_feat, rng_e,
                                args.maker_only, False, args.seed * 100 + gen)
            placebo = _next_gen(placebo, fits["placebo"], n_feat, rng_p,
                                args.maker_only, True, args.seed * 200 + gen)

    # ---- reserved test span ---------------------------------------------
    w5t, i0t, _ = _window(mkt5, ts5, test_t0, ts_max + 1, qs)
    w15t, _, _ = _window(mkt15, ts15, test_t0, ts_max + 1, qs)
    d0t, d1t = int(mkt5["day_pos"][i0t]), len(all_days)
    fresh = genome.sample(args.bots, n_feat, names, 0.0, args.seed + 5000,
                          maker_only=args.maker_only)
    hof_pop = genome.concat(hof)
    final = {}
    cohorts = (("evolved", evolved), ("placebo", placebo),
               ("fresh_random", fresh), ("hall_of_fame", hof_pop))
    test_outs = _evaluate_many([p for _, p in cohorts], w5t, w15t, cfg,
                               len(all_days), pool)
    if pool is not None:
        pool.shutdown()
    for (name, pop), out in zip(cohorts, test_outs):
        sh = _sharpe(out, d0t, d1t, args.start_capital)
        eqt = _fill_eq(out, d0t, d1t, args.start_capital)
        ret = (eqt[:, -1] / eqt[:, 0] - 1.0) * 100.0
        expo = (out["expo_a"] + out["expo_b"]) / np.maximum(out["bars_a"] + out["bars_b"], 1)
        trades = out["trades_a"] + out["trades_b"]
        okm = np.isfinite(sh) & (trades >= 8)
        final[name] = {
            "n": int(pop.n), "rankable": int(okm.sum()),
            "median_sharpe": round(float(np.nanmedian(sh[okm])), 3) if okm.any() else None,
            "p90_sharpe": round(float(np.nanpercentile(sh[okm], 90)), 3) if okm.any() else None,
            "max_sharpe": round(float(np.nanmax(sh[okm])), 3) if okm.any() else None,
            "pct_positive": round(float((sh[okm] > 0).mean() * 100), 1) if okm.any() else None,
            "median_ret_pct": round(float(np.median(ret)), 2),
            "median_expo_pct": round(float(np.median(expo) * 100), 1),
            "dead_pct": round(float(out["dead"].mean() * 100), 1),
        }
        if name == "hall_of_fame":
            gtab = genome.to_frame(pop)
            gtab["test_sharpe"] = np.round(sh, 3)
            gtab["test_trades"] = trades
            gtab["born_gen"] = np.repeat(np.arange(len(hof)), HOF_PER_GEN)
            gtab.sort_values("test_sharpe", ascending=False).to_csv(
                run_dir / "hof_test.csv", index=False)

    skill = None
    if final["evolved"]["median_sharpe"] is not None and \
       final["placebo"]["median_sharpe"] is not None:
        skill = round(final["evolved"]["median_sharpe"]
                      - final["placebo"]["median_sharpe"], 3)

    result = {"run_id": run_id, "seed": args.seed, "bots": args.bots,
              "fitness": args.fitness, "min_expo": args.min_expo,
              "gens": args.gens, "test_frac": args.test_frac,
              "test_start": str(np.datetime64(int(test_t0), 'ms')),
              "maker_only": bool(args.maker_only), "taker_bps": args.taker_bps,
              "maker_bps": args.maker_bps, "since": args.since,
              "feature_names": names, "gen_stats": gen_stats,
              "final_test": final, "skill_vs_placebo": skill,
              "elapsed_s": round(time.time() - t_start, 1)}
    (run_dir / "evolution.json").write_text(json.dumps(result, indent=2))

    print("\n  RESERVED TEST SPAN (never used for selection):")
    for name, r in final.items():
        print(f"    {name:14s} median S {r['median_sharpe']} | p90 {r['p90_sharpe']} "
              f"| max {r['max_sharpe']} | {r['pct_positive']}% positive "
              f"| median ret {r['median_ret_pct']}% | expo {r['median_expo_pct']}% "
              f"| dead {r['dead_pct']}%")
    print(f"  SKILL vs placebo (median-S difference): {skill}")
    print(f"  artifacts: {run_dir}  ({result['elapsed_s']}s)")
