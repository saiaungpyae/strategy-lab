#!/bin/bash
# overnight-evolve.sh — fetch new pairs, freeze a data pin, run the 15m evolve
# battery across PAIRS x SEEDS. Designed to run unattended overnight.
#
#   phase 1  candles for NEW_PAIRS (5m/15m/1h/4h since 2021)   ~25-30 min
#   phase 2  metrics + funding for NEW_PAIRS                   ~1-1.5 h
#            (Binance Vision serves one zip per day — slow is normal)
#   phase 3  sl-update top-up of every existing tape           ~2 min
#   phase 4  freeze data/snapshots/pin-YYYYMMDD                seconds
#   phase 5  evolve battery vs the pin                         ~6 min/run
#                                                              = ~3.2 h at 8x4
#   total: ~4.5-5.5 h
#
# Run with:      caffeinate -is ./scripts/overnight-evolve.sh
# (self-wraps in caffeinate anyway, but keep the laptop on AC power — on
# battery, closing the lid still sleeps the machine and pauses everything)
#
# Rerun after a partial failure:  SKIP_FETCH=1 ./scripts/overnight-evolve.sh
# (jumps straight to the pin + battery; the pin is reused if it exists)
#
# Everything below runs against the frozen pin, so the viewer's background
# data refresh can't shift the train/test split mid-battery. Results land in
# reports/swarm/ — watch live progress on the viewer's evolution tab.

set -u
cd "$(dirname "$0")/.."

if [ -z "${_CAFF:-}" ] && command -v caffeinate >/dev/null; then
  export _CAFF=1
  exec caffeinate -is "$0" "$@"
fi

PAIRS=(BTC ETH BNB SOL XRP DOGE ADA LINK)   # battery = PAIRS x SEEDS
NEW_PAIRS=(SOL XRP DOGE ADA LINK)           # fetched from scratch tonight
SEEDS=(63 64 65 66)
BOTS=100000
SINCE=2021-01-01
PIN=data/snapshots/pin-$(date +%Y%m%d)

[ -x .venv/bin/sl-swarm ] || { echo "FATAL: run from the repo (no .venv/bin/sl-swarm)"; exit 1; }

mkdir -p reports/overnight
LOG=reports/overnight/$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
die()   { echo "FATAL: $*"; exit 1; }
phase() { echo; echo "=== [$(( ($(date +%s) - t0) / 60 ))m] $*"; }

echo "overnight battery: ${PAIRS[*]} x seeds ${SEEDS[*]} ($BOTS bots) -> $PIN"
echo "log: $LOG"

if [ -z "${SKIP_FETCH:-}" ]; then
  # --- 1: candles for new pairs --------------------------------------------
  if [ ${#NEW_PAIRS[@]} -gt 0 ]; then
    SYMS=(); for P in "${NEW_PAIRS[@]}"; do SYMS+=("$P/USDT"); done
    for TF in 5m 15m 1h 4h; do
      phase "fetch $TF candles: ${SYMS[*]}"
      .venv/bin/sl-fetch -s "${SYMS[@]}" -t "$TF" --since $SINCE || die "candle fetch $TF"
    done
  fi

  # --- 2: metrics + funding for new pairs ----------------------------------
  # Vision metrics only exist from ~2021-12 for most alts; earlier days are
  # skipped gracefully and `--since auto` starts each pair at its coverage.
  for P in "${NEW_PAIRS[@]}"; do
    phase "fetch-metrics $P (one zip per day — takes ~10-15 min)"
    .venv/bin/sl-swarm fetch-metrics --symbol "${P}USDT" --since $SINCE || die "metrics $P"
    phase "fetch-funding $P"
    .venv/bin/sl-swarm fetch-funding --symbol "$P/USDT:USDT" --since $SINCE || die "funding $P"
  done

  # --- 3: top-up every tape (snapshots stay frozen) ------------------------
  phase "sl-update"
  .venv/bin/sl-update || die "sl-update"
fi

# --- 4: freeze the pin (canonical per-pair layout) -------------------------
if [ -d "$PIN" ]; then
  phase "pin $PIN already exists — reusing it untouched"
else
  phase "freezing $PIN"
  for P in "${PAIRS[@]}"; do
    mkdir -p "$PIN/ohlcv/$P-USDT" "$PIN/metrics/$P-USDT"
    cp "data/ohlcv/$P-USDT/binance_$P-USDT_5m.csv" \
       "data/ohlcv/$P-USDT/binance_$P-USDT_15m.csv" \
       "$PIN/ohlcv/$P-USDT/"                        || die "pin candles $P"
    cp data/metrics/"$P"-USDT/*.csv "$PIN/metrics/$P-USDT/" || die "pin metrics $P"
  done
fi

# preflight: every pair must fully resolve against the pin before we burn hours
phase "preflight: resolve_pair vs $PIN"
for P in "${PAIRS[@]}"; do
  .venv/bin/python -c "
from strategylab.swarm.run import resolve_pair
print('  ' + ' | '.join(p for p in resolve_pair('$P', root='$PIN', derivs=True) if p))" \
    || die "preflight $P"
done

# --- 5: battery ------------------------------------------------------------
FAILED=""
for P in "${PAIRS[@]}"; do
  for S in "${SEEDS[@]}"; do
    phase "evolve $P seed $S"
    .venv/bin/sl-swarm evolve --pair "$P" --derivs --since auto --tfs 15m \
        --data-root "$PIN" --bots $BOTS --seed "$S" \
      || { echo "RUN FAILED: $P s$S (continuing)"; FAILED="$FAILED $P/s$S"; }
  done
done

phase "battery done"
if [ -n "$FAILED" ]; then
  echo "FAILED runs:$FAILED"
else
  echo "all $(( ${#PAIRS[@]} * ${#SEEDS[@]} )) runs completed"
fi
echo "results: reports/swarm/  (viewer -> evolution tab)"
