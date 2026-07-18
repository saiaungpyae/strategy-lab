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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from . import engine, features, genome

ELITE_FRAC = 0.05
PARENT_FRAC = 0.25
FRESH_FRAC = 0.10
CHUNK = 128  # bots per parallel task; matches run.py so both saturate cores


def _alloc(n: int, n_days: int) -> dict:
    out = {"daily": np.full((n, n_days), np.nan, dtype=np.float32),
           "final_eq": np.zeros(n), "dead": np.zeros(n, dtype=bool),
           "death_day": np.full(n, -1, dtype=np.int64),
           "fees": np.zeros(n), "fund_paid": np.zeros(n)}
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
    w = {k: mkt[k][m0:i1] for k in ("o", "h", "l", "c", "atr", "hour", "day_pos",
                                    "F", "fund")}
    w.update({"seg_b": np.ones(i1 - m0, dtype=bool), "Q": Q, "qs": qs,
              "tf_code": mkt["tf_code"]})
    return w, i0, i1


def _eval_chunk(w, g_sub, cfg, n_days):
    """Worker: one CHUNK-sized slice of a (population, timeframe) cohort.
    Evolution lineages carry no control bots, so the control-RNG stream is
    never consumed and chunked results are bit-for-bit identical to the
    single-cohort path regardless of jobs or chunk boundaries."""
    out = _alloc(g_sub.n, n_days)
    engine.run_cohort(w, g_sub, np.arange(g_sub.n), cfg, out)
    return out


def _evaluate_many(pops, w5, w15, cfg, n_days, pool, on_progress=None):
    """Evaluate independent populations on the same window pair. Each pop's
    5m/15m cohorts are split into CHUNK-sized tasks so every core stays busy
    (pool=None falls back to serial). on_progress(frac_done) is called as
    tasks finish — this feeds the live progress.json for the viewer."""
    tasks = []
    for j, pop in enumerate(pops):
        for tf, w in ((0, w5), (1, w15)):
            idx = np.flatnonzero(pop.tf == tf)
            for i in range(0, len(idx), CHUNK):
                pos = idx[i:i + CHUNK]
                tasks.append((j, pos, genome.subset(pop, pos), w))
    rets = [None] * len(tasks)
    if pool is None:
        for ti, (_, _, g_sub, w) in enumerate(tasks):
            rets[ti] = _eval_chunk(w, g_sub, cfg, n_days)
            if on_progress:
                on_progress((ti + 1) / len(tasks))
    else:
        futs = {pool.submit(_eval_chunk, w, g_sub, cfg, n_days): ti
                for ti, (_, _, g_sub, w) in enumerate(tasks)}
        for done, f in enumerate(as_completed(futs), 1):
            rets[futs[f]] = f.result()
            if on_progress:
                on_progress(done / len(tasks))
    outs = [_alloc(pop.n, n_days) for pop in pops]
    for (j, pos, _, _), ret in zip(tasks, rets):
        for k in outs[j]:
            outs[j][k][pos] = ret[k]
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


def _fitness(out, d0, d1, start_cap, mode="sharpe", min_expo=0.15, day_up=None):
    """'sharpe': risk-adjusted (rewards sheltering when everything loses).
    'return': window return with a hard participation floor — a bot below
    `min_expo` time-in-market takes a penalty up to -25 return-points, so
    hiding in cash stops being a winning strategy.
    'balanced': mean of the bot's Sharpe on the window's up-days and down-days
    (underlying daily direction) — a bot must perform in BOTH regimes, so
    riding one regime stops being a winning strategy. Carries the same
    `min_expo` participation floor as 'return' (up to -6 Sharpe-points), so
    near-inactivity can't satisfy the both-regimes demand. Falls back to
    plain Sharpe when a window has <5 days of either regime."""
    sh = _sharpe(out, d0, d1, start_cap)
    trades = out["trades_a"] + out["trades_b"]
    if mode == "balanced":
        eq = _fill_eq(out, d0, d1, start_cap)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(eq, axis=1) / np.where(eq[:, :-1] == 0, np.nan, eq[:, :-1])
        du = day_up[d0 + 1:d1]

        def _side(mask):
            if mask.sum() < 5:
                return None
            sub = r[:, mask]
            with np.errstate(divide="ignore", invalid="ignore"):
                mu, sd = np.nanmean(sub, axis=1), np.nanstd(sub, axis=1)
                return np.where(sd > 0, mu / sd * np.sqrt(365.0), np.nan)

        s_up, s_dn = _side(du), _side(~du)
        if s_up is None or s_dn is None:
            bal = sh
        else:
            bal = np.where(np.isfinite(s_up) & np.isfinite(s_dn),
                           (s_up + s_dn) / 2.0, sh)
        fit = np.where(np.isfinite(bal), bal, -9.0)
        fit = fit - 2.0 * (trades < 10) - 4.0 * out["dead"]
        bars = np.maximum(out["bars_a"] + out["bars_b"], 1)
        expo = (out["expo_a"] + out["expo_b"]) / bars
        shortfall = np.clip((min_expo - expo) / max(min_expo, 1e-9), 0.0, 1.0)
        fit = fit - 6.0 * shortfall
    elif mode == "sharpe":
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

    hof_n = args.hof_per_gen if getattr(args, "hof_per_gen", 0) > 0 \
        else max(10, args.bots // 1000)
    hof_metric = getattr(args, "hof_metric", "fitness")

    # Live progress for the viewer's Evolution tab: rewritten after every
    # finished chunk, carrying the per-generation stats accumulated so far.
    meta = {"run_id": run_id, "seed": args.seed, "bots": args.bots,
            "gens": args.gens, "test_frac": args.test_frac,
            "fitness": args.fitness, "maker_only": bool(args.maker_only),
            "taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
            "since": args.since, "hof_per_gen": hof_n,
            "hof_metric": hof_metric,
            "metrics": args.metrics, "funding": args.funding}
    gen_stats = []

    def prog(stage: str, frac: float, gen: int = None) -> None:
        (run_dir / "progress.json").write_text(json.dumps(
            {**meta, "stage": stage, "gen": gen, "frac": round(frac, 4),
             "elapsed_s": round(time.time() - t_start, 1),
             "gen_stats": gen_stats}))

    prog("loading", 0.0)
    df5 = _load(args.file5, args.since, args.metrics, args.funding)
    df15 = _load(args.file15, args.since, args.metrics, args.funding)
    # provenance: data files are refreshed externally, so pin what this run saw
    meta["data"] = [
        {"path": p, "rows": len(d),
         "mtime": datetime.fromtimestamp(os.path.getmtime(p)).isoformat(
             timespec="seconds")}
        for p, d in ((args.file5, df5), (args.file15, df15))]
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

    # underlying daily direction (for 'balanced' fitness and the regime slice)
    day_close = np.full(len(all_days), np.nan)
    day_close[mkt5["day_pos"]] = mkt5["c"]
    _ix = np.where(np.isfinite(day_close), np.arange(len(day_close)), 0)
    np.maximum.accumulate(_ix, out=_ix)
    _dc = day_close[_ix]
    day_up = np.zeros(len(all_days), dtype=bool)
    day_up[1:] = _dc[1:] >= _dc[:-1]

    test_t0 = int(ts_min + (1.0 - args.test_frac) * (ts_max - ts_min))
    meta["test_start"] = str(np.datetime64(int(test_t0), "ms"))
    meta["span"] = [str(np.datetime64(ts_min, "ms").astype("datetime64[D]")),
                    str(np.datetime64(ts_max, "ms").astype("datetime64[D]"))]
    bounds = np.linspace(ts_min, test_t0, args.gens + 1).astype(np.int64)
    cfg = {"taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
           "start_capital": args.start_capital, "ruin_frac": args.ruin,
           "seed": args.seed}

    # cohorts are split into CHUNK-sized tasks (same scheme as run.py), so
    # workers scale with cores; threads suffice when the numba kernel (nogil)
    # is active
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
    hof = []

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
        outs = _evaluate_many(
            [p for _, p in lineages], w5, w15, cfg, len(all_days), pool,
            on_progress=lambda fr, g=gen: prog("gen", (g + fr) / (args.gens + 2), g))
        hof_fit = None
        for (name, _), out in zip(lineages, outs):
            fit, sh, trades = _fitness(out, d0, d1, args.start_capital,
                                       args.fitness, args.min_expo, day_up)
            fits[name] = fit
            row[name] = {"median_sharpe": round(float(np.nanmedian(sh)), 3),
                         "p90_sharpe": round(float(np.nanpercentile(sh, 90)), 3),
                         "dead_pct": round(float(out["dead"].mean() * 100), 1),
                         "median_trades": int(np.median(trades))}
            if name == "evolved":
                # hybrid mode: breed on args.fitness, admit HOF on Sharpe —
                # keeps e.g. return-fitness's population pressure without
                # letting max-exposure regime-riders monopolize the HOF
                if hof_metric == "sharpe" and args.fitness != "sharpe":
                    hof_fit, _, _ = _fitness(out, d0, d1, args.start_capital,
                                             "sharpe", args.min_expo)
                else:
                    hof_fit = fit
        hof_idx = np.argsort(-hof_fit)[:hof_n]
        hof.append(genome.subset(evolved, hof_idx))
        gen_stats.append(row)
        prog("gen", (gen + 1) / (args.gens + 2), gen)
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
    test_outs = _evaluate_many(
        [p for _, p in cohorts], w5t, w15t, cfg, len(all_days), pool,
        on_progress=lambda fr: prog("final_test", (args.gens + fr) / (args.gens + 2)))
    test_stats = {}
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
        test_stats[name] = {"sh": sh, "ret": ret, "trades": trades,
                            "dead": out["dead"], "eq": eqt, "okm": okm}
    hof_test = test_stats["hall_of_fame"]
    gtab = genome.to_frame(hof_pop)
    gtab["test_sharpe"] = np.round(hof_test["sh"], 3)
    gtab["test_trades"] = hof_test["trades"]
    gtab["born_gen"] = np.repeat(np.arange(len(hof)), hof_n)
    # hof_test.csv is written after the history pass, once oos_sharpe exists

    # Full evolved-population genomes + test outcomes: what the 20k-bot swarm
    # actually converged to, analyzable after the run (HOF alone is too small).
    pdf = genome.to_frame(evolved)
    est = test_stats["evolved"]
    pdf["test_sharpe"] = np.round(est["sh"], 3)
    pdf["test_ret_pct"] = np.round(est["ret"], 2)
    pdf["test_trades"] = est["trades"]
    pdf["test_dead"] = est["dead"]
    pdf.to_csv(run_dir / "final_population.csv", index=False)

    # Up-month vs down-month medians: same daily test returns, sliced by the
    # sign of the underlying's calendar-month move — separates "has edge in
    # normal months" from "sheltered well in the crash months".
    tdays = all_days[d0t:d1t]
    tmonths = tdays.astype("datetime64[M]")
    rmonths = tmonths[1:]  # month of each daily return
    up_mask = np.zeros(len(rmonths), dtype=bool)
    n_up = 0
    for mo in np.unique(tmonths):
        px = day_close[d0t:d1t][tmonths == mo]
        px = px[np.isfinite(px)]
        if len(px) >= 2 and px[-1] >= px[0]:
            up_mask[rmonths == mo] = True
            n_up += 1
    regime = {"up_months": n_up, "down_months": int(len(np.unique(tmonths))) - n_up}
    for name, st in test_stats.items():
        eqt = st["eq"]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(eqt, axis=1) / np.where(eqt[:, :-1] == 0, np.nan, eqt[:, :-1])
        for key, msk in (("up", up_mask), ("down", ~up_mask)):
            if not msk.any():
                final[name][f"median_sharpe_{key}_months"] = None
                continue
            sub = r[:, msk]
            with np.errstate(divide="ignore", invalid="ignore"):
                mu, sd = np.nanmean(sub, axis=1), np.nanstd(sub, axis=1)
                s = np.where(sd > 0, mu / sd * np.sqrt(365.0), np.nan)
            ok = st["okm"] & np.isfinite(s)
            final[name][f"median_sharpe_{key}_months"] = (
                round(float(np.median(s[ok])), 3) if ok.any() else None)

    # Bootstrap CI on the headline skill number (resample bots, not days).
    skill_ci = None
    e_sh = est["sh"][est["okm"]]
    p_sh = test_stats["placebo"]["sh"][test_stats["placebo"]["okm"]]
    if len(e_sh) > 10 and len(p_sh) > 10:
        brng = np.random.default_rng(args.seed + 424242)
        diffs = np.empty(1000)
        for b in range(1000):
            diffs[b] = (np.median(e_sh[brng.integers(0, len(e_sh), len(e_sh))])
                        - np.median(p_sh[brng.integers(0, len(p_sh), len(p_sh))]))
        skill_ci = [round(float(np.percentile(diffs, 2.5)), 3),
                    round(float(np.percentile(diffs, 97.5)), 3)]

    # ---- hall-of-fame history ------------------------------------------
    # Every HOF genome re-scored on every fitness window so the viewer can
    # show a bot's full walk-forward record. A bot's born window is where it
    # was selected (in-sample); every later window and the test span are
    # out-of-sample for it.
    hist_windows, per_win = [], []
    for gen in range(args.gens):
        t0, t1 = int(bounds[gen]), int(bounds[gen + 1])
        w5, i0_5, _ = _window(mkt5, ts5, t0, t1, qs)
        w15, _, _ = _window(mkt15, ts15, t0, t1, qs)
        d0 = int(mkt5["day_pos"][i0_5])
        d1 = int(w5["day_pos"][-1]) + 1
        out = _evaluate_many(
            [hof_pop], w5, w15, cfg, len(all_days), pool,
            on_progress=lambda fr, g=gen: prog(
                "hof_history",
                (args.gens + 1 + (g + fr) / args.gens) / (args.gens + 2)))[0]
        sh = _sharpe(out, d0, d1, args.start_capital)
        eq = _fill_eq(out, d0, d1, args.start_capital) / args.start_capital
        per_win.append((sh, (eq[:, -1] - 1.0) * 100.0,
                        out["trades_a"] + out["trades_b"], out["dead"], eq))
        hist_windows.append(
            {"gen": gen,
             "span": [str(np.datetime64(t0, "ms").astype("datetime64[D]")),
                      str(np.datetime64(t1, "ms").astype("datetime64[D]"))],
             "days": [str(x) for x in all_days[d0:d1]]})
    if pool is not None:
        pool.shutdown()

    # OOS consistency: a HOF bot's mean Sharpe on windows AFTER its birth —
    # its born-window fitness is an in-sample order statistic and measurably
    # uninformative about the test span; this is the honest ranking signal.
    born = gtab["born_gen"].to_numpy()
    win_sh = np.stack([pw[0] for pw in per_win])  # [gens, n_hof]
    oos = np.full(hof_pop.n, np.nan)
    for i in range(hof_pop.n):
        later = win_sh[born[i] + 1:, i]
        later = later[np.isfinite(later)]
        if len(later):
            oos[i] = later.mean()
    gtab["oos_sharpe"] = np.round(oos, 3)
    gtab.sort_values("test_sharpe", ascending=False).to_csv(
        run_dir / "hof_test.csv", index=False)

    def _corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3) \
            if ok.sum() > 2 else None
    born_sh = win_sh[born, np.arange(hof_pop.n)]
    hof_check = {"corr_born_fitness_vs_test": _corr(born_sh, hof_test["sh"]),
                 "corr_oos_consistency_vs_test": _corr(oos, hof_test["sh"])}

    def _f(x, nd=3):
        return round(float(x), nd) if np.isfinite(x) else None

    def _py(x):
        """numpy scalar → JSON-safe Python value (NaN → None)."""
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    hist_bots = []
    for i in range(hof_pop.n):
        rec = {k: _py(v) for k, v in gtab.iloc[i].items()}
        rec["gen_perf"] = [
            {"gen": g, "sharpe": _f(per_win[g][0][i]),
             "ret_pct": _f(per_win[g][1][i], 2),
             "trades": int(per_win[g][2][i]),
             "dead": bool(per_win[g][3][i])}
            for g in range(args.gens)]
        rec["eq"] = [np.round(per_win[g][4][i], 4).tolist()
                     for g in range(args.gens)]
        rec["test"] = {"sharpe": _f(hof_test["sh"][i]),
                       "ret_pct": _f(hof_test["ret"][i], 2),
                       "trades": int(hof_test["trades"][i]),
                       "dead": bool(hof_test["dead"][i])}
        rec["eq_test"] = np.round(
            hof_test["eq"][i] / args.start_capital, 4).tolist()
        hist_bots.append(rec)
    hist_bots.sort(key=lambda r: (r.get("test_sharpe") is None,
                                  -(r.get("test_sharpe") or 0.0)))
    (run_dir / "hof_history.json").write_text(json.dumps(
        {"run_id": run_id, "gens": args.gens,
         "start_capital": args.start_capital, "windows": hist_windows,
         "test": {"start": meta["test_start"],
                  "days": [str(x) for x in all_days[d0t:d1t]]},
         "bots": hist_bots}))

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
              "metrics": args.metrics, "funding": args.funding,
              "hof_per_gen": hof_n, "hof_metric": hof_metric,
              "data": meta["data"],
              "feature_names": names, "gen_stats": gen_stats,
              "final_test": final, "test_regime": regime,
              "skill_vs_placebo": skill, "skill_ci95": skill_ci,
              "hof_check": hof_check,
              "elapsed_s": round(time.time() - t_start, 1)}
    (run_dir / "evolution.json").write_text(json.dumps(result, indent=2))
    prog("done", 1.0)

    print("\n  RESERVED TEST SPAN (never used for selection):")
    for name, r in final.items():
        print(f"    {name:14s} median S {r['median_sharpe']} | p90 {r['p90_sharpe']} "
              f"| max {r['max_sharpe']} | {r['pct_positive']}% positive "
              f"| median ret {r['median_ret_pct']}% | expo {r['median_expo_pct']}% "
              f"| dead {r['dead_pct']}%")
    ci = f" (95% CI {skill_ci[0]}..{skill_ci[1]})" if skill_ci else ""
    print(f"  SKILL vs placebo (median-S difference): {skill}{ci}")
    print(f"  HOF predictiveness — born fitness vs test: "
          f"{hof_check['corr_born_fitness_vs_test']} | OOS consistency vs test: "
          f"{hof_check['corr_oos_consistency_vs_test']}")
    print(f"  artifacts: {run_dir}  ({result['elapsed_s']}s)")
