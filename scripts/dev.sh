#!/bin/bash
# Run the full dev stack in one terminal:
#   - paper-trading daemon        (scripts/paper-daemon.sh)
#   - viewer API/report server    (viewer/server.py, port 8020)
#   - vite frontend dev server    (viewer/frontend, port 5173)
#
#   ./scripts/dev.sh                 # 60s daemon cycles
#   ./scripts/dev.sh --interval 30   # extra args go to the daemon
#
# Vite runs in the foreground; Ctrl+C stops all three.
cd "$(dirname "$0")/.." || exit 1

[ -d viewer/frontend/node_modules ] || (cd viewer/frontend && npm install) || exit 1

# Kill the whole process group (daemon + viewer) when this script exits.
trap 'kill 0' EXIT INT TERM

if [ -f reports/paper/roster.json ]; then
  ./scripts/paper-daemon.sh "$@" 2>&1 | sed 's/^/[daemon] /' &
else
  echo "[daemon] no reports/paper/roster.json — skipping daemon" \
       "(freeze one with: .venv/bin/python -m strategylab.paper select)"
fi
# Free the viewer + vite ports first — stray instances (crashed dev.sh,
# manual runs) would otherwise collide with "Address already in use".
lsof -ti tcp:"${PORT:-8020}" -ti tcp:5173 | xargs kill 2>/dev/null
.venv/bin/python viewer/server.py 2>&1 | sed 's/^/[viewer] /' &

cd viewer/frontend && npm run dev
