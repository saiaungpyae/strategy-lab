#!/bin/bash
# overnight-evolve-perp.sh — perp-native battery. Re-runs the 2026-07-19
# spot battery (same pairs, seeds, protocol, 15m gene pool) on the USDT-M
# perp tapes, so the tape is the ONLY variable:
#
#   phase 1  sl-update top-up of every tape (incl. perp)       ~3 min
#   phase 2  freeze data/snapshots/pin-YYYYMMDD-perp           seconds
#   phase 3  preflight: resolve every pair perp-side           seconds
#   phase 4  evolve battery, --tape perp                       ~2 h at 8x4
#
# Run with:      ./scripts/overnight-evolve-perp.sh
# (self-wraps in caffeinate; keep the laptop on AC power)
# Rerun after a partial failure:  SKIP_UPDATE=1 ./scripts/overnight-evolve-perp.sh

set -u
cd "$(dirname "$0")/.."

if [ -z "${_CAFF:-}" ] && command -v caffeinate >/dev/null; then
  export _CAFF=1
  exec caffeinate -is "$0" "$@"
fi

PAIRS=(BTC ETH BNB SOL XRP DOGE ADA LINK)
SEEDS=(63 64 65 66)
BOTS=100000
PIN=data/snapshots/pin-$(date +%Y%m%d)-perp

[ -x .venv/bin/sl-swarm ] || { echo "FATAL: run from the repo (no .venv/bin/sl-swarm)"; exit 1; }

mkdir -p reports/overnight
LOG=reports/overnight/perp-$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
die()   { echo "FATAL: $*"; exit 1; }
phase() { echo; echo "=== [$(( ($(date +%s) - t0) / 60 ))m] $*"; }

echo "perp battery: ${PAIRS[*]} x seeds ${SEEDS[*]} ($BOTS bots) -> $PIN"
echo "log: $LOG"

if [ -z "${SKIP_UPDATE:-}" ]; then
  phase "sl-update (top-up all tapes; snapshots stay frozen)"
  .venv/bin/sl-update || die "sl-update"
fi

if [ -d "$PIN" ]; then
  phase "pin $PIN already exists — reusing it untouched"
else
  phase "freezing $PIN"
  for P in "${PAIRS[@]}"; do
    mkdir -p "$PIN/ohlcv/$P-USDT" "$PIN/metrics/$P-USDT"
    cp "data/ohlcv/$P-USDT/binanceusdm_$P-USDT-USDT_5m.csv" \
       "data/ohlcv/$P-USDT/binanceusdm_$P-USDT-USDT_15m.csv" \
       "$PIN/ohlcv/$P-USDT/"                        || die "pin candles $P"
    cp data/metrics/"$P"-USDT/*.csv "$PIN/metrics/$P-USDT/" || die "pin metrics $P"
  done
fi

phase "preflight: resolve_pair (tape=perp) vs $PIN"
for P in "${PAIRS[@]}"; do
  .venv/bin/python -c "
from strategylab.swarm.run import resolve_pair
print('  ' + ' | '.join(p for p in resolve_pair('$P', root='$PIN', derivs=True, tape='perp') if p))" \
    || die "preflight $P"
done

FAILED=""
for P in "${PAIRS[@]}"; do
  for S in "${SEEDS[@]}"; do
    phase "evolve $P (perp) seed $S"
    .venv/bin/sl-swarm evolve --pair "$P" --tape perp --derivs --since auto \
        --tfs 15m --data-root "$PIN" --bots $BOTS --seed "$S" \
      || { echo "RUN FAILED: $P s$S (continuing)"; FAILED="$FAILED $P/s$S"; }
  done
done

phase "battery done"
if [ -n "$FAILED" ]; then
  echo "FAILED runs:$FAILED"
else
  echo "all $(( ${#PAIRS[@]} * ${#SEEDS[@]} )) perp runs completed"
fi
echo "results: reports/swarm/  (runs carry \"tape\": \"perp\" in evolution.json)"
