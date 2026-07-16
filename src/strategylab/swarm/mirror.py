"""mirror_long tearsheet — promote the swarm's one survivor to a full grading.

Strategy family: go LONG when the top-trader long/short position ratio shows
crowded shorts (`top_ls_pos < quantile`). The probe showed a broad-and-thin
edge (85-96% of configs OOS-positive, point-optimized configs regress), so the
tradeable object is the EQUAL-WEIGHT ENSEMBLE of all 144 family configs, not a
single config.

This report:
- runs the family on 15m / 1h / 4h, under maker execution (resting limits,
  adverse-selection edge only; stops/time exits pay taker) AND pure taker
  execution (market entries) — is the edge fee-dependent?
- per-year ensemble returns vs buy & hold,
- correlation + 50/50 daily-rebalanced blend with the lab's best strategy
  (slow SMA 89/365 cross on 1h, long-only, re-implemented here vectorized),
- everything split-aware: train → 2024-11, test = 2024-11 → 2026-07.

Note: take-profits are modeled as resting limits in both execution modes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine
from .evolve import _alloc
from .probe import TF, build_grid

TFS = {"15m": ("file15", 96), "1h": ("file1h", 24), "4h": ("file4h", 6)}


def _stats(r: np.ndarray) -> dict:
    """Stats from a daily-return series (NaNs allowed)."""
    r = np.asarray(r, dtype=np.float64)
    ok = np.isfinite(r)
    n = ok.sum()
    if n < 30:
        return {k: float("nan") for k in ("total", "cagr", "sharpe", "maxdd")}
    eq = np.cumprod(1.0 + np.where(ok, r, 0.0))
    peak = np.maximum.accumulate(eq)
    return {"total": eq[-1] - 1.0,
            "cagr": eq[-1] ** (365.0 / n) - 1.0,
            "sharpe": float(np.nanmean(r) / np.nanstd(r) * np.sqrt(365.0))
                      if np.nanstd(r) > 0 else float("nan"),
            "maxdd": float((eq / peak - 1.0).min())}


def _per_year(days: np.ndarray, r: np.ndarray) -> dict:
    out = {}
    years = pd.Series(days).dt.year.to_numpy()
    for y in np.unique(years):
        m = (years == y) & np.isfinite(r)
        out[int(y)] = float(np.prod(1.0 + r[m]) - 1.0)
    return out


def family_daily_returns(df, tf_label, bars_per_day, exec_mode, args, all_days,
                         split_day):
    """Equal-weight ensemble daily returns of the mirror_long family."""
    from .run import _market
    ts = df["timestamp"].to_numpy(np.int64)
    split_ts = int(ts[0] + args.split * (ts[-1] - ts[0]))
    qs = np.linspace(0.02, 0.98, 49)
    mkt, names = _market(df, 1, bars_per_day, split_ts, all_days, qs)
    g, labels = build_grid(names, bars_per_day)
    keep = np.flatnonzero((labels["family"] == "mirror_long").to_numpy())
    from . import genome as _gn
    g = _gn.subset(g, keep)
    if exec_mode == "taker":
        g.maker_off = np.zeros(g.n, dtype=np.float32)  # market entries
    cfg = {"taker_bps": args.taker_bps, "maker_bps": args.maker_bps,
           "start_capital": 10_000.0, "ruin_frac": 0.0, "seed": 0}
    out = _alloc(g.n, len(all_days))
    engine.run_cohort(mkt, g, np.arange(g.n), cfg, out)
    d = out["daily"].astype(np.float64)
    d[:, 0][~np.isfinite(d[:, 0])] = 10_000.0
    for j in range(1, d.shape[1]):
        col = d[:, j]
        col[~np.isfinite(col)] = d[:, j - 1][~np.isfinite(col)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_cfg = np.diff(d, axis=1) / d[:, :-1]
    ens = np.nanmean(r_cfg, axis=0)                       # equal-weight, daily rebalance
    # per-config test sharpe -> % positive OOS
    rt = r_cfg[:, split_day:]
    with np.errstate(invalid="ignore"):
        sh = np.nanmean(rt, axis=1) / np.nanstd(rt, axis=1) * np.sqrt(365.0)
    pos = float(np.nanmean(sh > 0) * 100)
    expo = float(np.median((out["expo_a"] + out["expo_b"])
                           / np.maximum(out["bars_a"] + out["bars_b"], 1)) * 100)
    return ens, pos, expo


def sma_cross_daily_returns(df1h, all_days, fast=89, slow=365, taker_bps=5.0):
    """Slow SMA cross, long-only, next-bar execution, taker cost on flips."""
    c = df1h["close"].to_numpy(np.float64)
    f = pd.Series(c).rolling(fast).mean().to_numpy()
    s = pd.Series(c).rolling(slow).mean().to_numpy()
    pos = np.where(f > s, 1.0, 0.0)
    pos[np.isnan(s)] = 0.0
    pos = np.roll(pos, 1)          # decide at close, hold from next bar
    pos[0] = 0.0
    r_bar = np.diff(c, prepend=c[0]) / np.roll(c, 1)
    r_bar[0] = 0.0
    cost = np.abs(np.diff(pos, prepend=0.0)) * taker_bps / 1e4
    strat = pos * r_bar - cost
    day = df1h["dt"].dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    daily = pd.Series(1.0 + strat).groupby(day).prod().reindex(all_days) - 1.0
    return daily.to_numpy()


def cmd_mirror(args) -> None:
    from .run import _load

    df1h = _load(args.file1h, args.since, args.metrics, args.funding)
    all_days = np.unique(df1h["dt"].dt.tz_convert(None).dt.floor("D")
                         .to_numpy().astype("datetime64[D]"))
    days_dt = all_days.astype("datetime64[D]")
    ts = df1h["timestamp"].to_numpy(np.int64)
    split_ts = int(ts[0] + args.split * (ts[-1] - ts[0]))
    split_day = int(np.searchsorted(
        all_days, np.datetime64(pd.Timestamp(split_ts, unit="ms"), "D")))

    # buy & hold daily returns (1h closes)
    day_close = pd.Series(df1h["close"].to_numpy(),
                          index=df1h["dt"].dt.tz_convert(None).dt.floor("D")) \
                  .groupby(level=0).last().reindex(all_days).ffill().to_numpy()
    r_bh = np.diff(day_close) / day_close[:-1]
    r_bh = np.concatenate([[0.0], r_bh])

    rows = []
    ens_by_tf = {}
    for tf_label, (file_arg, bpd) in TFS.items():
        df = df1h if tf_label == "1h" else _load(getattr(args, file_arg),
                                                 args.since, args.metrics, args.funding)
        if tf_label != "1h":  # clamp to the 1h-derived day grid
            bar_day = df["dt"].dt.tz_convert(None).dt.floor("D") \
                              .to_numpy().astype("datetime64[D]")
            df = df[(bar_day >= all_days[0]) & (bar_day <= all_days[-1])] \
                .reset_index(drop=True)
        for mode in ("maker", "taker"):
            ens, pos, expo = family_daily_returns(df, tf_label, bpd, mode, args,
                                                  all_days, split_day)
            full = _stats(ens)
            test = _stats(ens[split_day:])
            rows.append({"tf": tf_label, "exec": mode, "pos_oos": pos,
                         "expo": expo, **{f"full_{k}": v for k, v in full.items()},
                         **{f"test_{k}": v for k, v in test.items()}})
            if mode == "maker":
                ens_by_tf[tf_label] = ens
            print(f"  {tf_label} {mode}: full CAGR {full['cagr']*100:+.1f}% "
                  f"S {full['sharpe']:.2f} DD {full['maxdd']*100:.0f}% | "
                  f"test S {test['sharpe']:.2f} | {pos:.0f}% cfg OOS+")

    r_sma = sma_cross_daily_returns(df1h, all_days, taker_bps=args.taker_bps)
    r_ens = ens_by_tf["1h"]
    n = min(len(r_sma), len(r_ens)) - 1
    r_sma, r_ens_a, r_bh_a = r_sma[1:n + 1], r_ens[:n], r_bh[1:n + 1]
    okc = np.isfinite(r_sma) & np.isfinite(r_ens_a)
    corr = float(np.corrcoef(r_sma[okc], r_ens_a[okc])[0, 1])
    r_blend = 0.5 * np.where(np.isfinite(r_sma), r_sma, 0.0) \
            + 0.5 * np.where(np.isfinite(r_ens_a), r_ens_a, 0.0)
    sd = max(split_day - 1, 0)

    def block(r):
        return {"full": _stats(r), "test": _stats(r[sd:]),
                "years": _per_year(days_dt[1:n + 1], r)}

    comps = {"mirror_long ensemble (1h maker)": block(r_ens_a),
             "slow SMA 89/365 (1h)": block(r_sma),
             "50/50 blend (daily rebal.)": block(r_blend),
             "buy & hold": block(r_bh_a)}

    # ---- report ----------------------------------------------------------
    L = [f"# mirror_long tearsheet — {datetime.now().date()}", "",
         f"Family: LONG when `top_ls_pos` (top-trader long/short position ratio) "
         f"is below its train quantile — 144 configs, traded as an equal-weight "
         f"ensemble. Span {all_days[0]} → {all_days[-1]}, split "
         f"{all_days[split_day]}. Maker exec: resting limits (0 fee + "
         f"{args.maker_bps} bp edge), stops/time exits taker {args.taker_bps} bps. "
         f"Taker exec: market entries. TPs rest as limits in both modes.", "",
         "## Ensemble by timeframe and execution", "",
         "| tf | exec | full CAGR | full Sharpe | maxDD | test Sharpe | % configs OOS+ | expo |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['tf']} | {r['exec']} | {r['full_cagr']*100:+.1f}% "
                 f"| {r['full_sharpe']:.2f} | {r['full_maxdd']*100:.0f}% "
                 f"| {r['test_sharpe']:.2f} | {r['pos_oos']:.0f}% | {r['expo']:.0f}% |")
    L += ["", "## Blend with the lab's best (slow SMA 89/365, 1h)", "",
          f"Daily-return correlation mirror_long ↔ SMA cross: **{corr:+.3f}**", "",
          "| series | full CAGR | full Sharpe | full maxDD | test CAGR | test Sharpe | test maxDD |",
          "|---|---|---|---|---|---|---|"]
    for name, b in comps.items():
        L.append(f"| {name} | {b['full']['cagr']*100:+.1f}% | {b['full']['sharpe']:.2f} "
                 f"| {b['full']['maxdd']*100:.0f}% | {b['test']['cagr']*100:+.1f}% "
                 f"| {b['test']['sharpe']:.2f} | {b['test']['maxdd']*100:.0f}% |")
    years = sorted(next(iter(comps.values()))["years"])
    L += ["", "## Per-year returns", "",
          "| year | " + " | ".join(comps.keys()) + " |",
          "|---|" + "---|" * len(comps)]
    for y in years:
        L.append(f"| {y} | " + " | ".join(f"{comps[k]['years'][y]*100:+.1f}%"
                                          for k in comps) + " |")
    # carry a hand-written verdict section across regenerations
    out_md = Path(args.out) / "mirror_long_tearsheet.md"
    verdict = ""
    if out_md.exists():
        old = out_md.read_text()
        i = old.find("## Verdict")
        if i >= 0:
            verdict = old[i:].split("_Generated by")[0].rstrip()
    if verdict:
        L += ["", verdict]
    L += ["", "_Generated by `sl-swarm mirror-report`._"]
    out_md.write_text("\n".join(L))
    print(f"\nwrote {out_md}")

    # paired HTML with equity curves (lab convention: .html + .md)
    def _eq(r):
        return np.cumprod(1.0 + np.where(np.isfinite(r), r, 0.0))

    import json as _json
    step = max(1, n // 1200)
    sel = list(range(0, n, step))
    if sel[-1] != n - 1:
        sel.append(n - 1)
    dsel = [str(days_dt[1:][i]) for i in sel]
    series = [("mirror_long ensemble (1h maker)", "#26a69a", _eq(r_ens_a)),
              ("slow SMA 89/365 (1h)", "#58a6ff", _eq(r_sma)),
              ("50/50 blend", "#f0b429", _eq(r_blend)),
              ("buy & hold", "#8b949e", _eq(r_bh_a))]
    payload = {"days": dsel, "split": str(all_days[split_day]),
               "series": [{"name": nm, "color": col,
                           "eq": [round(float(eq[i]), 5) for i in sel]}
                          for nm, col, eq in series]}

    tbl1 = "".join(f"<tr><td>{r['tf']}</td><td>{r['exec']}</td>"
                   f"<td>{r['full_cagr']*100:+.1f}%</td><td>{r['full_sharpe']:.2f}</td>"
                   f"<td>{r['full_maxdd']*100:.0f}%</td><td>{r['test_sharpe']:.2f}</td>"
                   f"<td>{r['pos_oos']:.0f}%</td><td>{r['expo']:.0f}%</td></tr>"
                   for r in rows)
    tbl2 = "".join(f"<tr><td>{k}</td><td>{b['full']['cagr']*100:+.1f}%</td>"
                   f"<td>{b['full']['sharpe']:.2f}</td><td>{b['full']['maxdd']*100:.0f}%</td>"
                   f"<td>{b['test']['cagr']*100:+.1f}%</td><td>{b['test']['sharpe']:.2f}</td>"
                   f"<td>{b['test']['maxdd']*100:.0f}%</td></tr>"
                   for k, b in comps.items())
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mirror_long tearsheet</title><style>
:root{{--bg:#0e1117;--panel:#161b22;--border:#2a2f38;--text:#e6edf3;--muted:#8b949e;--accent:#26a69a}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:28px 20px 80px}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:30px;border-bottom:1px solid var(--border);padding-bottom:6px}}
a{{color:var(--accent);text-decoration:none}}
table{{border-collapse:collapse;margin:14px 0;font-size:13px}}
th,td{{border:1px solid var(--border);padding:6px 10px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}th{{background:var(--panel);color:var(--muted)}}
canvas{{width:100%;display:block;background:var(--panel);border:1px solid var(--border);border-radius:8px}}
.legend{{display:flex;gap:16px;font-size:12px;color:var(--muted);margin:8px 0 4px;flex-wrap:wrap}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.muted{{color:var(--muted)}}.top{{padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--border)}}
</style></head><body>
<div class="top"><a href="/">← dashboard</a> · <a href="/swarm">bot swarm</a> ·
<a href="/reports/mirror_long_tearsheet.md">full report + verdict (md)</a></div><main>
<h1>mirror_long tearsheet — equity curves</h1>
<p class="muted">LONG when top-trader long/short ratio shows crowded shorts; 144-config
equal-weight ensemble. Span {all_days[0]} → {all_days[-1]}, dashed line = train|test split
({all_days[split_day]}). Generated by <code>sl-swarm mirror-report</code>.</p>
<h2>All series (log scale)</h2><div class="legend" id="lg1"></div><canvas id="c1" height="340"></canvas>
<h2>Ensemble alone (linear)</h2><canvas id="c2" height="240"></canvas>
<h2>Ensemble by timeframe and execution</h2>
<table><thead><tr><th>tf</th><th>exec</th><th>full CAGR</th><th>full Sharpe</th><th>maxDD</th>
<th>test Sharpe</th><th>% configs OOS+</th><th>expo</th></tr></thead><tbody>{tbl1}</tbody></table>
<h2>Blend comparison (corr {corr:+.3f})</h2>
<table><thead><tr><th>series</th><th>full CAGR</th><th>full Sharpe</th><th>full maxDD</th>
<th>test CAGR</th><th>test Sharpe</th><th>test maxDD</th></tr></thead><tbody>{tbl2}</tbody></table>
<script>
const D={_json.dumps(payload)};
function draw(id, series, log) {{
  const c=document.getElementById(id), dpr=window.devicePixelRatio||1;
  const W=c.clientWidth, H=+c.getAttribute('height');
  c.width=W*dpr; c.height=H*dpr; c.style.height=H+'px';
  const g=c.getContext('2d'); g.scale(dpr,dpr);
  const P={{l:56,r:12,t:14,b:26}}, n=D.days.length;
  const tr=v=>log?Math.log(v):v;
  let lo=1/0, hi=-1/0;
  series.forEach(s=>s.eq.forEach(v=>{{lo=Math.min(lo,tr(v));hi=Math.max(hi,tr(v));}}));
  const pad=(hi-lo)*.05; lo-=pad; hi+=pad;
  const X=i=>P.l+(W-P.l-P.r)*i/(n-1), Y=v=>H-P.b-(H-P.t-P.b)*(tr(v)-lo)/(hi-lo);
  g.strokeStyle='#2a2f38'; g.strokeRect(P.l+.5,P.t+.5,W-P.l-P.r,H-P.t-P.b);
  g.fillStyle='#8b949e'; g.font='10px sans-serif';
  for(let k=0;k<=4;k++){{const v=lo+(hi-lo)*k/4;
    const lab=log?'x'+Math.exp(v).toFixed(2):'x'+v.toFixed(2);
    g.textAlign='right'; g.fillText(lab,P.l-6,H-P.b-(H-P.t-P.b)*k/4+3);}}
  const si=D.days.findIndex(d=>d>=D.split);
  if(si>0){{g.strokeStyle='#f0b429'; g.setLineDash([5,4]); g.beginPath();
    g.moveTo(X(si),H-P.b); g.lineTo(X(si),P.t); g.stroke(); g.setLineDash([]);}}
  series.forEach(s=>{{g.strokeStyle=s.color; g.lineWidth=1.5; g.beginPath();
    s.eq.forEach((v,i)=>i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v))); g.stroke();}});
  g.fillStyle='#8b949e'; g.textAlign='left'; g.fillText(D.days[0],P.l,H-8);
  g.textAlign='right'; g.fillText(D.days[n-1],W-P.r,H-8);
}}
document.getElementById('lg1').innerHTML=D.series.map(s=>
  `<span><i style="background:${{s.color}}"></i>${{s.name}}</span>`).join('');
function paint(){{draw('c1',D.series,true);draw('c2',[D.series[0]],false);}}
paint(); addEventListener('resize',()=>{{clearTimeout(window.__r);window.__r=setTimeout(paint,150);}});
</script></main></body></html>"""
    out_html = Path(args.out) / "mirror_long_tearsheet.html"
    out_html.write_text(html)
    print(f"wrote {out_html}")


def add_parser(sub):
    m = sub.add_parser("mirror-report", help="mirror_long tearsheet + SMA blend")
    m.add_argument("--file15", default="data/binance_BTC-USDT_15m.csv")
    m.add_argument("--file1h", default="data/binance_BTC-USDT_1h.csv")
    m.add_argument("--file4h", default="data/binance_BTC-USDT_4h.csv")
    m.add_argument("--metrics", default="data/metrics/BTCUSDT_metrics.csv")
    m.add_argument("--funding", default="data/metrics/BTC-USDT-USDT_funding.csv")
    m.add_argument("--since", default="2021-01-06")
    m.add_argument("--split", type=float, default=0.70)
    m.add_argument("--taker-bps", type=float, default=5.0)
    m.add_argument("--maker-bps", type=float, default=1.0)
    m.add_argument("--out", default="reports")

    def _run(a):
        cmd_mirror(a)
    m.set_defaults(func=_run)
