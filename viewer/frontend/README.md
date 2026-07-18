# viewer/frontend

Vite + React + TypeScript frontend for the strategy-lab viewer. Replaces the
legacy single-file pages in `viewer/static/` (kept as a fallback).

- `/` dashboard · `/chart` candle viewer · `/swarm` bot swarm
- Charts: `lightweight-charts` (npm, same 4.2.3 the legacy pages vendored)
- Tables: `@tanstack/react-table` (Bots tab: sort / filter / 250-row pages)
- All data comes from `viewer/server.py` — this app is UI only.

## Build (what the server serves)

```sh
cd viewer/frontend
npm install
npm run build     # → viewer/static/dist, served by server.py at / /chart /swarm
```

`server.py` serves `static/dist/index.html` for the three page routes when the
build exists, otherwise it falls back to the legacy static pages.

## Dev (hot reload)

```sh
./.venv/bin/python viewer/server.py   # API on :8020
cd viewer/frontend && npm run dev     # UI on :5173, proxies /api + /reports
```

`npm run check` runs the TypeScript typecheck.
