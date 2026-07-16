"""Swarm engine — one pass over the tape, vectorized across bots.

Execution semantics (documented simplifications are v1 choices):
- Decisions happen at bar close t; nothing fills before t+1 (no lookahead).
- Taker (chase) entries fill at next bar's open and pay taker bps.
- Maker entries rest a limit at close ∓ offset*ATR for `order_ttl` bars; they
  fill only if price trades through, and pay an adverse-selection edge in bps
  (you get filled precisely when it's slightly bad for you).
- Stops exit as takers at the stop price; take-profits exit as maker limits;
  time exits (max hold) exit as takers at the close.
- If stop and TP are both touched within one bar, the stop wins (conservative).
- Exit checks start the bar AFTER entry (same-bar entry+stop not modeled).
- Sizing: fixed fractional — qty = equity * risk% * revenge_mult / stop_dist,
  capped at 3x notional leverage. Equity below ruin_frac * start => dead.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
except ImportError:  # pure-numpy fallback path below
    numba = None

MAX_LEV = 3.0
WARMUP = 300


def _interp_thresholds(rule_feat, rule_q, Q, qs):
    """Resolve per-rule quantile -> value against the train-only quantile grid."""
    Qsel = Q[rule_feat]                              # [n, R, len(qs)]
    pos = (rule_q - qs[0]) / (qs[-1] - qs[0]) * (len(qs) - 1)
    pos = np.clip(pos, 0, len(qs) - 1 - 1e-9)
    lo = pos.astype(np.int64)
    w = pos - lo
    n, R = rule_feat.shape
    ii = np.arange(n)[:, None]
    jj = np.arange(R)[None, :]
    v_lo = Qsel[ii, jj, lo]
    v_hi = Qsel[ii, jj, np.minimum(lo + 1, len(qs) - 1)]
    return (v_lo * (1 - w) + v_hi * w).astype(np.float64)


def _bar_kernel(o, h, l, c, A, hour, day_pos, seg_b, F,
                is_ctrl, ctrl_rate, rule_feat, rule_dir, rule_op_gt, rule_act,
                thr, dir_bias, risk, stop_atr, tp_rr, max_hold, maker_off,
                order_ttl, loss_react, cooldown_len, revenge, reentry_gap,
                sess_ok_bot, rng, taker, edge, start_cap, ruin, n_days):
    """JIT bar loop. Mirrors the vectorized numpy path op-for-op per element,
    so both paths (and any chunking of the cohort) are bit-identical — numba
    implements np.random.Generator with numpy's exact bitstream."""
    m = is_ctrl.shape[0]
    n_bars = c.shape[0]
    R = rule_feat.shape[1]

    pos = np.zeros(m, np.int64)
    entry = np.zeros(m, np.float64)
    qty = np.zeros(m, np.float64)
    stop_px = np.zeros(m, np.float64)
    tp_px = np.zeros(m, np.float64)
    held = np.zeros(m, np.int64)
    pk = np.zeros(m, np.int64)          # pending: 0 none / 1 market / 2 limit
    pdir = np.zeros(m, np.int64)
    ppx = np.zeros(m, np.float64)
    pttl = np.zeros(m, np.int64)
    cd = np.zeros(m, np.int64)
    rmult = np.ones(m, np.float64)
    gap_until = np.zeros(m, np.int64)
    eq = np.full(m, start_cap, np.float64)
    dead = np.zeros(m, np.bool_)
    elig = np.zeros(m, np.bool_)

    daily = np.full((m, n_days), np.nan, np.float32)
    trades = np.zeros((m, 2), np.int64)
    wins = np.zeros((m, 2), np.int64)
    expo = np.zeros((m, 2), np.int64)
    death_day = np.full(m, -1, np.int64)
    u = np.zeros(m, np.float64)
    r2 = np.zeros(m, np.float64)

    def open_pos(i, raw, eff, at):
        d = float(pdir[i])
        dist = stop_atr[i] * at
        # numpy semantics for ATR=0 bars: q -> inf, then the leverage cap wins
        q = eq[i] * risk[i] * rmult[i] / dist if dist > 0.0 else np.inf
        cap = MAX_LEV * eq[i] / raw
        if q > cap:
            q = cap
        pos[i] = pdir[i]
        entry[i] = eff
        qty[i] = q
        stop_px[i] = raw - d * dist
        rr = tp_rr[i]
        tp_px[i] = raw + d * rr * dist if np.isfinite(rr) else d * np.inf
        held[i] = 0
        pk[i] = 0

    def close_pos(i, exit_eff, t, seg):
        pnl = qty[i] * (exit_eff - entry[i]) * pos[i]
        eq[i] += pnl
        trades[i, seg] += 1
        if pnl > 0:
            wins[i, seg] += 1
            rmult[i] = 1.0
        else:
            if loss_react[i] == 1:
                cd[i] = cooldown_len[i]
            rmult[i] = revenge[i] if loss_react[i] == 2 else 1.0
        gap_until[i] = t + reentry_gap[i]
        pos[i] = 0
        qty[i] = 0.0
        held[i] = 0
        if eq[i] < ruin:  # ruin line
            dead[i] = True
            death_day[i] = day_pos[t]
            pk[i] = 0

    for t in range(WARMUP, n_bars):
        ot, ht, lt, ct, at = o[t], h[t], l[t], c[t], A[t]
        seg = 1 if seg_b[t] else 0
        hr = hour[t]
        any_ctl = False

        for i in range(m):
            # -- 1. fill pending orders placed on earlier bars ------------
            if pk[i] > 0 and not dead[i] and pos[i] == 0:
                if pk[i] == 1:                        # market @ open, taker
                    open_pos(i, ot, ot * (1.0 + pdir[i] * taker), at)
                elif (lt <= ppx[i]) if pdir[i] > 0 else (ht >= ppx[i]):
                    open_pos(i, ppx[i], ppx[i] * (1.0 + pdir[i] * edge), at)
                else:                                 # resting limit ages
                    pttl[i] -= 1
                    if pttl[i] <= 0:
                        pk[i] = 0

            # -- 2. exits (stop wins over TP within one bar) ---------------
            if pos[i] != 0 and held[i] >= 1:
                long = pos[i] > 0
                if (lt <= stop_px[i]) if long else (ht >= stop_px[i]):
                    close_pos(i, stop_px[i] * (1.0 - pos[i] * taker), t, seg)
                elif (ht >= tp_px[i]) if long else (lt <= tp_px[i]):
                    close_pos(i, tp_px[i] * (1.0 - pos[i] * edge), t, seg)
                elif held[i] >= max_hold[i]:
                    close_pos(i, ct * (1.0 - pos[i] * taker), t, seg)

            # -- eligibility for new entries -------------------------------
            e = (not dead[i] and pos[i] == 0 and pk[i] == 0 and cd[i] == 0
                 and t >= gap_until[i] and sess_ok_bot[i, hr])
            elig[i] = e
            if e and is_ctrl[i]:
                any_ctl = True

        # control draws: full-size arrays, only on bars where a control bot
        # is eligible — replicates the numpy path's RNG consumption exactly
        if any_ctl:
            u = rng.random(m)
            r2 = rng.random(m)

        write_day = t + 1 >= n_bars or day_pos[t + 1] != day_pos[t]
        for i in range(m):
            # -- 3. new entries (decided at close, fill from next bar) ----
            if elig[i]:
                des = 0
                if is_ctrl[i]:
                    if u[i] < ctrl_rate[i]:
                        des = 1 if r2[i] < 0.5 else -1
                else:
                    sig = 0
                    for r in range(R):
                        if not rule_act[i, r]:
                            continue
                        v = F[t, rule_feat[i, r]]
                        if not np.isfinite(v):
                            continue
                        if (v > thr[i, r]) if rule_op_gt[i, r] else (v < thr[i, r]):
                            sig += rule_dir[i, r]
                    des = 1 if sig > 0 else (-1 if sig < 0 else 0)
                if des > 0 and dir_bias[i] < 0:   # ideology direction filter
                    des = 0
                elif des < 0 and dir_bias[i] > 0:
                    des = 0
                if des != 0:
                    pdir[i] = des
                    if maker_off[i] == 0.0:       # taker chase
                        pk[i] = 1
                        pttl[i] = 1
                    else:                          # resting maker limit
                        pk[i] = 2
                        ppx[i] = ct - des * maker_off[i] * at
                        pttl[i] = order_ttl[i]

            # -- 4. bookkeeping -------------------------------------------
            if cd[i] > 0:
                cd[i] -= 1
            if pos[i] != 0:
                held[i] += 1
                expo[i, seg] += 1
            if write_day:
                daily[i, day_pos[t]] = eq[i] + qty[i] * (ct - entry[i]) * pos[i]

    # mark-to-market close of anything still open (taker at last close)
    for i in range(m):
        if pos[i] != 0:
            eq[i] += qty[i] * (c[n_bars - 1] * (1.0 - pos[i] * taker)
                               - entry[i]) * pos[i]

    return daily, trades, wins, expo, eq, dead, death_day


if numba is not None:
    # nogil: kernel runs outside the GIL, so cohort chunks parallelize on
    # plain threads — no process spawn, no market pickling, shared memory
    _bar_kernel = numba.njit(cache=True, nogil=True)(_bar_kernel)


def run_cohort(mkt, g, idx, cfg, out, progress=None, rng_salt=0):
    """Simulate the cohort of bots `idx` (indices into genome arrays) on one
    timeframe's market dict, writing per-bot results into `out`.

    rng_salt distinguishes the control-bot RNG stream when a cohort is split
    into parallel chunks; salt 0 preserves the original single-cohort stream.

    mkt keys: o,h,l,c,atr (float64), hour (int 0-23), day_pos (int index into
    the global day grid), seg_b (bool: bar in test segment), F (float32
    [bars x feats]), Q (train quantile grid [feats x len(qs)]), qs, tf_code.
    """
    m = len(idx)
    if m == 0:
        return
    o, h, l, c = mkt["o"], mkt["h"], mkt["l"], mkt["c"]
    A, hour, day_pos, seg_b, F = mkt["atr"], mkt["hour"], mkt["day_pos"], mkt["seg_b"], mkt["F"]

    taker = cfg["taker_bps"] / 1e4
    edge = cfg["maker_bps"] / 1e4
    start_cap = float(cfg["start_capital"])
    ruin = cfg["ruin_frac"] * start_cap

    # --- cohort genome slices -------------------------------------------
    is_ctrl = g.is_control[idx]
    ctrl_rate = g.ctrl_rate[idx].astype(np.float64)
    rule_feat = g.rule_feat[idx].astype(np.int64)
    rule_dir = g.rule_dir[idx].astype(np.int64)
    rule_op_gt = g.rule_op[idx] > 0
    rule_act = np.arange(g.rule_feat.shape[1])[None, :] < g.n_rules[idx][:, None]
    thr = _interp_thresholds(rule_feat, g.rule_q[idx].astype(np.float64), mkt["Q"], mkt["qs"])
    dir_bias = g.dir_bias[idx].astype(np.int64)
    risk = g.risk_pct[idx].astype(np.float64)
    stop_atr = g.stop_atr[idx].astype(np.float64)
    tp_rr = g.tp_rr[idx].astype(np.float64)
    max_hold = g.max_hold[idx].astype(np.int64)
    maker_off = g.maker_off[idx].astype(np.float64)
    order_ttl = g.order_ttl[idx].astype(np.int64)
    loss_react = g.loss_react[idx]
    cooldown_len = g.cooldown[idx].astype(np.int64)
    revenge = g.revenge[idx].astype(np.float64)
    reentry_gap = g.reentry_gap[idx].astype(np.int64)

    from .genome import SESSION_HOURS
    sess_tab = np.zeros((4, 24), dtype=bool)
    for k, (s, e) in SESSION_HOURS.items():
        sess_tab[k, s:e] = True
    sess_ok_bot = sess_tab[g.session[idx].astype(np.int64)]  # [m, 24]

    rng = np.random.default_rng(cfg["seed"] * 7919 + int(mkt["tf_code"]) + rng_salt)

    fn = _bar_kernel if numba is not None else _bar_loop_py
    daily, trades, wins, expo, eq, dead, death_day = fn(
        o, h, l, c, A, hour, day_pos, seg_b, F,
        is_ctrl, ctrl_rate, rule_feat, rule_dir, rule_op_gt, rule_act,
        thr, dir_bias, risk, stop_atr, tp_rr, max_hold, maker_off,
        order_ttl, loss_react.astype(np.int64), cooldown_len, revenge,
        reentry_gap, sess_ok_bot, rng, taker, edge, start_cap, ruin,
        out["daily"].shape[1])

    out["daily"][idx] = daily
    out["final_eq"][idx] = eq
    out["trades_a"][idx], out["trades_b"][idx] = trades[:, 0], trades[:, 1]
    out["wins_a"][idx], out["wins_b"][idx] = wins[:, 0], wins[:, 1]
    out["expo_a"][idx], out["expo_b"][idx] = expo[:, 0], expo[:, 1]
    out["bars_a"][idx] = int((~seg_b[WARMUP:]).sum())
    out["bars_b"][idx] = int(seg_b[WARMUP:].sum())
    out["death_day"][idx] = death_day
    out["dead"][idx] = dead


def _bar_loop_py(o, h, l, c, A, hour, day_pos, seg_b, F,
                 is_ctrl, ctrl_rate, rule_feat, rule_dir, rule_op_gt, rule_act,
                 thr, dir_bias, risk, stop_atr, tp_rr, max_hold, maker_off,
                 order_ttl, loss_react, cooldown_len, revenge, reentry_gap,
                 sess_ok_bot, rng, taker, edge, start_cap, ruin, n_days):
    """Pure-numpy fallback bar loop (used when numba is not installed).
    Bit-identical to _bar_kernel."""
    m = is_ctrl.shape[0]
    n_bars = len(c)

    # --- state -----------------------------------------------------------
    pos = np.zeros(m, dtype=np.int64)         # -1 / 0 / +1
    entry = np.zeros(m)                       # cost-adjusted entry price
    qty = np.zeros(m)
    stop_px = np.zeros(m)
    tp_px = np.zeros(m)
    held = np.zeros(m, dtype=np.int64)
    pk = np.zeros(m, dtype=np.int64)          # pending: 0 none / 1 market / 2 limit
    pdir = np.zeros(m, dtype=np.int64)
    ppx = np.zeros(m)
    pttl = np.zeros(m, dtype=np.int64)
    cd = np.zeros(m, dtype=np.int64)
    rmult = np.ones(m)
    gap_until = np.zeros(m, dtype=np.int64)
    eq = np.full(m, start_cap, dtype=np.float64)
    dead = np.zeros(m, dtype=bool)

    daily = np.full((m, n_days), np.nan, dtype=np.float32)
    trades = np.zeros((m, 2), dtype=np.int64)   # col 0=train, 1=test
    wins = np.zeros((m, 2), dtype=np.int64)
    expo = np.zeros((m, 2), dtype=np.int64)
    death_day = np.full(m, -1, dtype=np.int64)

    def open_position(who, raw_px, eff_px, at):
        """who: cohort indices. raw_px: intended price (stop math), eff_px:
        cost-adjusted fill. Direction comes from the pending order."""
        d = pdir[who].astype(np.float64)
        dist = stop_atr[who] * at
        q = eq[who] * risk[who] * rmult[who] / dist
        q = np.minimum(q, MAX_LEV * eq[who] / raw_px)
        pos[who] = pdir[who]
        entry[who] = eff_px
        qty[who] = q
        stop_px[who] = raw_px - d * dist
        rr = tp_rr[who]
        tp_px[who] = np.where(np.isfinite(rr), raw_px + d * rr * dist, d * np.inf)
        held[who] = 0

    def close_positions(who, exit_eff, t):
        seg = 1 if seg_b[t] else 0
        pnl = qty[who] * (exit_eff - entry[who]) * pos[who]
        eq[who] += pnl
        trades[who, seg] += 1
        won = pnl > 0
        wins[who, seg] += won
        w_idx, l_idx = who[won], who[~won]
        rmult[w_idx] = 1.0
        cd[l_idx] = np.where(loss_react[l_idx] == 1, cooldown_len[l_idx], cd[l_idx])
        rmult[l_idx] = np.where(loss_react[l_idx] == 2, revenge[l_idx], 1.0)
        gap_until[who] = t + reentry_gap[who]
        pos[who] = 0
        qty[who] = 0.0
        held[who] = 0
        d = who[eq[who] < ruin]  # ruin line
        dead[d] = True
        death_day[d] = day_pos[t]
        pk[d] = 0

    for t in range(WARMUP, n_bars):
        ot, ht, lt, ct, at = o[t], h[t], l[t], c[t], A[t]

        # -- 1. fill pending orders placed on earlier bars ----------------
        live = (pk > 0) & ~dead & (pos == 0)
        if live.any():
            mk = np.flatnonzero(live & (pk == 1))     # market @ open, taker
            if len(mk):
                open_position(mk, np.full(len(mk), ot),
                              ot * (1 + pdir[mk] * taker), at)
                pk[mk] = 0
            lm = np.flatnonzero(live & (pk == 2))     # resting limits
            if len(lm):
                fill = np.where(pdir[lm] > 0, lt <= ppx[lm], ht >= ppx[lm])
                fl = lm[fill]
                if len(fl):
                    open_position(fl, ppx[fl], ppx[fl] * (1 + pdir[fl] * edge), at)
                    pk[fl] = 0
                rest = lm[~fill]
                pttl[rest] -= 1
                pk[rest[pttl[rest] <= 0]] = 0

        # -- 2. exits ------------------------------------------------------
        act = np.flatnonzero((pos != 0) & (held >= 1))
        if len(act):
            long = pos[act] > 0
            stop_hit = np.where(long, lt <= stop_px[act], ht >= stop_px[act])
            tp_hit = np.where(long, ht >= tp_px[act], lt <= tp_px[act]) & ~stop_hit
            time_hit = (held[act] >= max_hold[act]) & ~stop_hit & ~tp_hit

            s_ = act[stop_hit]
            if len(s_):
                close_positions(s_, stop_px[s_] * (1 - pos[s_] * taker), t)
            p_ = act[tp_hit]
            if len(p_):
                close_positions(p_, tp_px[p_] * (1 - pos[p_] * edge), t)
            x_ = act[time_hit]
            if len(x_):
                close_positions(x_, ct * (1 - pos[x_] * taker), t)

        # -- 3. new entries (decided at close, fill from next bar) --------
        elig = (~dead & (pos == 0) & (pk == 0) & (cd == 0)
                & (t >= gap_until) & sess_ok_bot[:, hour[t]])
        if elig.any():
            des = np.zeros(m, dtype=np.int64)
            pat = elig & ~is_ctrl
            if pat.any():
                vals = F[t][rule_feat]                       # [m, R] float32
                cond = (np.where(rule_op_gt, vals > thr, vals < thr)
                        & np.isfinite(vals) & rule_act)
                sig = (cond * rule_dir).sum(axis=1)
                des[pat] = np.sign(sig[pat]).astype(np.int64)
            ctl = elig & is_ctrl
            if ctl.any():
                u = rng.random(m)
                d2 = np.where(rng.random(m) < 0.5, 1, -1)
                des[ctl] = np.where(u[ctl] < ctrl_rate[ctl], d2[ctl], 0)
            des[(des > 0) & (dir_bias < 0)] = 0   # ideology direction filter
            des[(des < 0) & (dir_bias > 0)] = 0
            go = np.flatnonzero(des != 0)
            if len(go):
                tk = go[maker_off[go] == 0]
                pk[tk], pdir[tk], pttl[tk] = 1, des[tk], 1
                mkr = go[maker_off[go] > 0]
                pk[mkr], pdir[mkr] = 2, des[mkr]
                ppx[mkr] = ct - des[mkr] * maker_off[mkr] * at
                pttl[mkr] = order_ttl[mkr]

        # -- 4. bookkeeping ------------------------------------------------
        np.maximum(cd - 1, 0, out=cd)
        in_pos = pos != 0
        held[in_pos] += 1
        expo[in_pos, 1 if seg_b[t] else 0] += 1

        if t + 1 >= n_bars or day_pos[t + 1] != day_pos[t]:
            daily[:, day_pos[t]] = eq + qty * (ct - entry) * pos

    # mark-to-market close of anything still open (taker at last close)
    open_ = np.flatnonzero(pos != 0)
    if len(open_):
        eq[open_] += qty[open_] * (c[-1] * (1 - pos[open_] * taker) - entry[open_]) * pos[open_]

    return daily, trades, wins, expo, eq, dead, death_day
