#!/bin/bash
# Launch the live paper-trading daemon from anywhere.
#   ./scripts/paper-daemon.sh                # 60s cycles
#   ./scripts/paper-daemon.sh --interval 30
# Stateless: safe to kill and restart at any time (positions and the trade
# log are re-derived from the roster epoch each cycle).
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -m strategylab.paper daemon "$@"
