"""Freeze the paper-trading roster.

For each pair: pool the hall-of-fame bots of its battery evolution runs,
rank by reserved-test Sharpe, dedupe identical genomes, and keep the first
`--top` bots that PASS the hostile-futures stress battery — the same
criterion the viewer's stress page shows as "survived all hostile futures":
no hostile scenario may kill the bot, cost ≥25% of the account, or draw
down ≥40%.

Each roster entry freezes the full genome record plus its ABSOLUTE rule
thresholds, resolved once at selection from the pair's full real history
(exactly what the bot would use live — the stress module's convention).
The daemon never re-derives thresholds, so live data growth cannot silently
shift a bot's behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..swarm import engine, features, stress, trace
from ..swarm.run import _load, resolve_pair

QS = np.linspace(0.02, 0.98, 49)
BARS_PER_DAY = 96          # roster bots are 15m (the battery ran --tfs 15m)
TF_CODE = 1


def hostile_passed(scenarios: list[dict]) -> bool:
    """StressForecast.tsx's `crashed` rule, inverted, over hostile scenarios."""
    return all(not (s["dead"] or s["ret_pct"] <= -25 or s["maxdd_pct"] <= -40)
               for s in scenarios if s["key"] != "replay")


def _genome_sig(rec: dict) -> tuple:
    return tuple(rec.get(k) for k in (
        "rules", "tf", "dir_bias", "risk_pct", "stop_atr", "tp_rr",
        "max_hold_bars", "maker_off_atr", "order_ttl", "session",
        "loss_react", "cooldown_bars", "revenge_mult", "reentry_gap"))


def battery_runs(log_path: Path | None = None) -> dict[str, list[Path]]:
    """Pair -> battery run dirs, from the overnight log's artifacts lines."""
    log = log_path or sorted(Path("reports/overnight").glob("*.log"))[-1]
    runs: dict[str, list[Path]] = {}
    for m in re.finditer(r"artifacts: (reports/swarm/evo-\S+)", log.read_text()):
        d = Path(m.group(1))
        pair = d.name.rsplit("-", 1)[-1]
        runs.setdefault(pair, []).append(d)
    return runs


def select_pair(pair: str, run_dirs: list[Path], top: int, max_candidates: int,
                data_root: str, tape: str = "spot") -> tuple[list[dict], list[str]]:
    _f5, f15, metrics, funding = resolve_pair(pair, root=data_root, derivs=True,
                                              tape=tape)
    df = _load(f15, "2021-01-01", metrics, funding)
    F, names = features.compute_features(df, BARS_PER_DAY)
    Q_full = features.train_quantiles(F, len(F), QS)

    pool = []
    for d in run_dirs:
        h = json.loads((d / "hof_history.json").read_text())
        e = json.loads((d / "evolution.json").read_text())
        cfg = {"taker_bps": float(e.get("taker_bps") or 5.0),
               "maker_bps": float(e.get("maker_bps") or 1.0),
               "start_capital": float(h.get("start_capital", 10_000.0)),  # scaled at roster level below
               "ruin_frac": 0.30, "seed": int(e.get("seed") or 0)}
        for b in h.get("bots", []):
            if b.get("is_control") or b.get("test_sharpe") is None:
                continue
            pool.append({"rec": b, "run_id": e["run_id"], "seed": e.get("seed"),
                         "cfg": cfg})
    pool.sort(key=lambda x: -float(x["rec"]["test_sharpe"]))

    seen: set[tuple] = set()
    chosen, tried = [], 0
    for cand in pool:
        if len(chosen) >= top or tried >= max_candidates:
            break
        sig = _genome_sig(cand["rec"])
        if sig in seen:
            continue
        seen.add(sig)
        tried += 1
        rec = cand["rec"]
        try:
            scenarios = stress.stress_bot(rec, names, df, F, QS, cand["cfg"],
                                          BARS_PER_DAY, TF_CODE)
        except ValueError as ex:
            print(f"    {pair}#{rec['bot_id']} ({cand['run_id']}): skipped ({ex})")
            continue
        ok = hostile_passed(scenarios)
        worst = min((s for s in scenarios if s["key"] != "replay"),
                    key=lambda s: s["ret_pct"])
        print(f"    {pair}#{rec['bot_id']} testS={rec['test_sharpe']} "
              f"{'PASS' if ok else 'fail'} (worst: {worst['key']} "
              f"{worst['ret_pct']}%/dd{worst['maxdd_pct']}%)")
        if not ok:
            continue
        p = trace.parse_record(rec, names)
        thr = engine._interp_thresholds(p["rule_feat"], p["rule_q"],
                                        Q_full, QS)[0]
        # per-rule quantile ladders: lets the daemon place the LIVE feature
        # value on the same scale the threshold was drawn from ("how far
        # from triggering", in quantile space)
        trigger_meta = [
            {"feature": names[int(p["rule_feat"][0][r])],
             "op": ">" if p["rule_op_gt"][r] else "<",
             "q": float(p["rule_q"][0][r]),
             "thr": round(float(thr[r]), 8),
             "dir": "LONG" if p["rule_dir"][r] > 0 else "SHORT",
             "ladder": [round(float(x), 8)
                        for x in Q_full[int(p["rule_feat"][0][r])]]}
            for r in range(len(thr))]
        chosen.append({
            "pair": pair, "run_id": cand["run_id"], "seed": cand["seed"],
            "bot_id": int(rec["bot_id"]), "label": f"{pair}#{rec['bot_id']}",
            "test_sharpe": rec.get("test_sharpe"),
            "oos_sharpe": rec.get("oos_sharpe"), "born_gen": rec.get("born_gen"),
            "rec": {k: v for k, v in rec.items() if k not in ("eq", "eq_test",
                                                              "gen_perf")},
            "thresholds": [round(float(t), 8) for t in thr],
            "trigger_meta": trigger_meta,
            "cfg": cand["cfg"],
            "stress": [{k: s[k] for k in ("key", "label", "ret_pct",
                                          "maxdd_pct", "trades", "dead")}
                       for s in scenarios],
        })
    return chosen, names


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5, help="bots per pair")
    ap.add_argument("--max-candidates", type=int, default=20,
                    help="stress-test at most this many candidates per pair")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="restrict to these pairs (default: all battery pairs)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--tape", choices=["spot", "perp"], default="spot",
                    help="which candle tape to select/stress against")
    ap.add_argument("--log", default=None,
                    help="overnight log whose battery runs feed the pool "
                         "(default: newest reports/overnight/*.log)")
    ap.add_argument("--capital", type=float, default=100.0,
                    help="paper capital per bot (engine sizing is\n"
                         "percentage-based, so this is pure display scale)")
    ap.add_argument("--out", default="reports/paper")
    args = ap.parse_args(argv)

    runs = battery_runs(Path(args.log) if args.log else None)
    pairs = args.pairs or sorted(runs)
    bots, names_by_pair = [], {}
    for pair in pairs:
        print(f"  {pair}: pooling {len(runs[pair])} runs")
        chosen, names = select_pair(pair, runs[pair], args.top,
                                    args.max_candidates, args.data_root,
                                    tape=args.tape)
        if len(chosen) < args.top:
            print(f"    ⚠ only {len(chosen)}/{args.top} stress-passers "
                  f"within {args.max_candidates} candidates")
        bots += chosen
        names_by_pair[pair] = names

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # trades exist only at/after this instant — the paper epoch
        "paper_start_ms": int(time.time() * 1000),
        "start_capital": args.capital, "ruin_frac": 0.30, "tape": args.tape,
        "criteria": "top-{} per pair by test Sharpe among hostile-futures "
                    "passers (no hostile scenario dead / ret<=-25% / "
                    "dd<=-40%)".format(args.top),
        "feature_names": names_by_pair,
        "bots": bots,
    }
    (out_dir / "roster.json").write_text(json.dumps(roster, indent=2))
    print(f"\nroster: {len(bots)} bots across {len(pairs)} pairs "
          f"-> {out_dir / 'roster.json'}")


if __name__ == "__main__":
    main()
