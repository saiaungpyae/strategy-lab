// Typed shapes of the JSON payloads served by viewer/server.py.

export interface DatasetInfo {
  file: string
  exchange: string
  symbol: string
  timeframe: string
  label: string
}

// stats are absent when the server could not read the file (error is set);
// `gaps` counts missing bars (expected-from-span minus actual rows)
export interface HealthDataset extends DatasetInfo {
  error?: string
  rows?: number
  first?: number
  last?: number
  age_seconds?: number
  bars_behind?: number | null
  gaps?: number
  size_bytes?: number
}

export interface OverviewPair {
  pair: string
  error?: string
  symbol?: string
  file?: string
  timeframe?: string
  last_close?: number
  change_24h_pct?: number | null
  spark?: number[]
  age_seconds?: number
  has_perp?: boolean
  n_datasets?: number
}

export interface RefreshResult {
  file: string
  added?: number
  error?: string
}

export interface RefreshStatus {
  state: 'idle' | 'running' | 'done'
  started: number | null
  finished: number | null
  results: RefreshResult[]
  triggered?: boolean
}

export interface HealthPayload {
  datasets: HealthDataset[]
  refresh: RefreshStatus
}

export interface SignalState {
  state: 'long' | 'flat'
  bars_since_flip: number
}

export interface SnapshotDataset extends DatasetInfo {
  error?: string
  last_close: number
  last_time: number
  signals: {
    ema_cross: SignalState
    sma_cross: SignalState
    supertrend: SignalState
    fvg_open: number
  }
  change_24h_pct: number | null
}

export interface ReportItem {
  file: string
  kind: string
  size_bytes: number
  mtime: number
}

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
}

export interface HistItem {
  time: number
  value: number
  color?: string
}

export interface CandlesPayload {
  error?: string
  file: string
  returned: number
  total: number
  candles: Candle[]
  volume: HistItem[]
  session: HistItem[]
}

export interface SignalLine {
  name: string
  color: string
  data: { time: number; value: number }[]
}

export interface ChartMarker {
  time: number
  position: 'aboveBar' | 'belowBar'
  color: string
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square'
  text: string
}

export interface SignalsPayload {
  error?: string
  label: string
  lines: SignalLine[]
  markers: ChartMarker[]
  entries: number
  exits: number
}

export interface FvgZone {
  from: number
  to: number
  top: number
  bottom: number
  dir: number
  outcome: 'win' | 'loss' | 'timeout' | 'open' | 'untouched'
}

export interface FvgSummary {
  gaps: number
  bull: number
  bear: number
  gaps_per_day: number
  touched: number
  touch_rate: number
  wins: number
  losses: number
  timeouts: number
  win_rate: number
  control_win_rate: number | null
  control_sd: number
  percentile_vs_random: number | null
  avg_net_pct: number
  avg_r: number
  grade: 'pass' | 'mixed' | 'weak' | 'fail' | 'none'
  verdict: string
}

export interface FvgPayload {
  error?: string
  zones: FvgZone[]
  markers: ChartMarker[]
  summary: FvgSummary
}

// ---------------------------------------------------------------- swarm ----

export interface SwarmRunInfo {
  run_id: string
  bots: number | null
  span: [string, string] | null
  split_date: string | null
  created: string | null
  stage: string | null
  frac: number | null
  has_recap: boolean
}

export interface SwarmRunsPayload {
  runs: SwarmRunInfo[]
  running: boolean
}

export interface SwarmProgress {
  stage?: string
  frac?: number
  elapsed_s?: number
}

export interface RecapTiles {
  n_bots: number
  n_control: number
  alive_pct: number
  above_water_pct: number
  median_final_mult: number
  bnh_mult: number
  yardstick_sharpe_p95: number
  yardstick_sharpe_p99: number
  rank_corr_gap: number | null
  rank_corr_pattern: number
  rank_corr_control: number
}

export interface RecapConfig {
  seed: number
  ruin_frac: number
  span: [string, string]
  split_date: string
  start_capital: number
  taker_bps: number
  maker_bps: number
}

export interface RecapHistogram {
  bins: number[]
  pattern: number[]
  control: number[]
  refs: {
    start?: number | null
    bnh?: number | null
    ctl_p95?: number | null
    ctl_p99?: number | null
  }
}

export interface RecapSurvival {
  days: string[]
  split_day: string
  pattern: { above_water: (number | null)[] }
  control: { above_water: (number | null)[] }
}

export interface TraitGroup {
  value: string
  n: number
  median_sharpe_b: number
}

export interface RecapTraits {
  n_used: number
  numeric: { trait: string; rho: number }[]
  features: { feature: string; delta_vs_all: number }[]
  categorical: { trait: string; groups: TraitGroup[] }[]
}

export interface RecapPersistence {
  rho_pattern: number | null
  rho_control: number | null
  scatter: [number, number, number][]
  top_decile_dest: number[] | null
}

export interface LeaderboardRow {
  bot_id: number
  control: boolean
  dead: boolean
  rules: string
  tf: string
  session: string
  risk_pct: number
  sharpe_a: number | null
  sharpe_b: number | null
  final_usd: number | null
  ret_a: number | null
  ret_b: number | null
  maxdd_b: number | null
  trades: number
  expo_b_pct: number | null
}

export interface RegimeRow {
  regime: string
  days: number
  bnh_bps_day: number
  pattern_bps_day: number
  control_bps_day: number
}

export interface Recap {
  tiles: RecapTiles
  config: RecapConfig
  histogram: RecapHistogram
  survival: RecapSurvival
  traits: RecapTraits
  persistence: RecapPersistence
  leaderboard: LeaderboardRow[]
  regimes?: { basis: string; rows: RegimeRow[] }
}

export interface SwarmBotsPayload {
  error?: string
  start_capital: number
  n: number
  bots: {
    bot_id: number[]
    control: boolean[]
    tf: string[]
    session: string[]
    rules: string[]
    final_usd: (number | null)[]
    ret_a: (number | null)[]
    ret_b: (number | null)[]
    sharpe_b: (number | null)[]
    trades: number[]
    dead: boolean[]
  }
}

export interface BotRow {
  bot_id: number
  control: boolean
  tf: string
  session: string
  rules: string
  final_usd: number | null
  ret_a: number | null
  ret_b: number | null
  sharpe_b: number | null
  trades: number
  dead: boolean
}

export interface BotYearly {
  year: string
  ret_pct: number
  end_usd: number
}

export interface BotDetail {
  error?: string
  genome: Record<string, string | number | boolean | null>
  result: Record<string, string | number | boolean | null>
  days: string[]
  equity: number[]
  split_day: string
  yearly: BotYearly[]
}

export interface EvoGroupStats {
  median_sharpe?: number | null
  p90_sharpe?: number | null
  pct_positive?: number | null
  median_ret_pct?: number | null
  dead_pct?: number | null
  median_trades?: number
}

export interface EvoGenStat {
  gen: number
  window: [string, string]
  evolved: EvoGroupStats
  placebo: EvoGroupStats
}

export interface HofRow {
  bot_id: number | string
  born_gen: number | string
  rules: string
  tf: string
  session: string
  risk_pct: number | string
  oos_sharpe?: number | string
  test_sharpe: number | string
  test_trades: number | string
}

export interface EvoRun {
  run_id: string
  done: boolean
  bots?: number
  gens?: number
  gen?: number
  stage?: string
  frac?: number
  fitness?: string
  seed?: number
  span?: [string, string]
  test_start?: string
  maker_only?: boolean
  maker_bps?: number
  taker_bps?: number
  elapsed_s?: number
  age_s?: number
  gen_stats?: EvoGenStat[]
  skill_vs_placebo?: number | null
  skill_ci95?: [number, number] | null
  hof_check?: {
    corr_born_fitness_vs_test: number | null
    corr_oos_consistency_vs_test: number | null
  }
  funding?: string | null
  pair?: string | null
  tfs?: string[] | null
  metrics_coverage?: string | null
  final_test?: Record<string, EvoGroupStats>
  hof_top?: HofRow[]
}

export interface EvosPayload {
  evos: EvoRun[]
  running: boolean
  pairs?: { pair: string; has_metrics: boolean }[]
}

export interface StartResult {
  started: boolean
  error?: string
  cmd?: string
}

// ------------------------------------------------- evolution bot pages ----

export interface EvoGenPerf {
  gen: number
  sharpe: number | null
  ret_pct: number | null
  trades: number
  dead: boolean
}

export interface EvoWindow {
  gen: number
  span: [string, string]
  days: string[]
}

export interface EvoBotSummary {
  bot_id: number
  born_gen: number
  rules: string
  tf: string
  session: string
  dir_bias?: string
  risk_pct: number | null
  oos_sharpe?: number | null
  test_sharpe: number | null
  test_trades: number | null
  test_ret_pct?: number | null
  final_usd?: number | null
  gen_sharpes?: (number | null)[]
}

export interface EvoBotsPayload {
  error?: string
  run_id: string
  fitness?: string
  gens?: number
  test_start?: string
  seed?: number
  bots_per_lineage?: number
  has_history: boolean
  n_hof: number
  start_capital?: number
  windows: [string, string][]
  bots: EvoBotSummary[]
}

export interface EvoBotRecord extends EvoBotSummary {
  n_rules?: number
  stop_atr?: number | null
  tp_rr?: number | null
  max_hold_bars?: number | null
  maker_off_atr?: number | null
  order_ttl?: number | null
  loss_react?: string
  cooldown_bars?: number | null
  revenge_mult?: number | null
  reentry_gap?: number | null
  gen_perf?: EvoGenPerf[]
  eq?: number[][]
  test?: { sharpe: number | null; ret_pct: number | null; trades: number; dead: boolean }
  eq_test?: number[]
}

export interface EvoBotPayload {
  error?: string
  run_id: string
  has_history: boolean
  start_capital?: number
  windows: EvoWindow[]
  test: { start?: string; days?: string[] }
  // run-level context for the worst-case panel: generation count, pair and
  // the null cohorts' best test Sharpe (the run's luck ceilings)
  run_ctx?: {
    gens?: number | null
    pair?: string | null
    placebo_max?: number | null
    fresh_max?: number | null
  }
  bot: EvoBotRecord
}

// One engineered 90-day future from the stress forecast battery.
export interface StressScenario {
  key: string
  label: string
  desc: string
  eq: number[] // daily equity multiplier, ×1.00 start
  px: number[] // daily scenario price, normalized to 1.0
  ret_pct: number
  maxdd_pct: number
  trades: number
  dead: boolean
  fees: number
}

export interface EvoStressPayload {
  error?: string
  run_id: string
  bot_id: number
  tf: string
  days: number
  scenarios: StressScenario[]
  note?: string
}

// One replayed position of a hall-of-fame bot (times are epoch seconds).
export interface BotTrade {
  seg: string // 'g0'…'gN' fitness window, or 'test'
  side: 1 | -1
  et: number
  ep: number
  xt: number
  xp: number
  qty: number
  pnl: number
  why: 'stop' | 'tp' | 'time' | 'eof'
  hold: number
  ruin?: boolean
}

export interface EvoTradesPayload {
  error?: string
  run_id: string
  bot_id: number
  tf: string
  rules: string
  born_gen: number
  file: string | null
  windows: { seg: string; span: [string, string] }[]
  test_start: string
  n_trades: number
  trades: BotTrade[]
  note?: string
}

// ---- paper trading (reports/paper via /api/paper) --------------------------

export interface PaperPosition {
  side: 'LONG' | 'SHORT'
  entry: number
  entry_eff: number
  qty: number
  opened_sec: number | null
  held_bars: number
  stop_px: number
  tp_px: number | null
  liq_px: number | null
  leverage: number
  entry_cost: number
  funding: number
}

export interface PaperPendingOrder {
  side: 'LONG' | 'SHORT'
  type: 'market' | 'limit'
  px: number | null
  ttl_bars: number
}

export interface PaperBot {
  pair: string
  label: string
  bot_id: number
  run_id: string
  rules: string
  tf: string
  test_sharpe: number | null
  maker_off_atr: number | null
  order_ttl: number | null
  triggers?: {
    feature: string
    op: '>' | '<'
    dir: 'LONG' | 'SHORT'
    need_q: number
    thr: number
    value: number | null
    cur_q: number | null
    fired: boolean
  }[]
  status: 'idle' | 'pending' | 'position' | 'dead'
  equity: number
  n_trades: number
  realized_pnl: number
  unrealized_pnl: number | null
  mark: number | null
  position?: PaperPosition
  pending_order?: PaperPendingOrder
  last_bar_sec?: number
}

export interface PaperTrade {
  bot: string
  pair: string
  side: number
  et: number
  ep: number
  xt: number
  xp: number
  qty: number
  pnl: number
  fund?: number
  why: string
  hold: number
}

export interface PaperState {
  generated_ms: number
  generated: string
  paper_start_ms: number
  interval_s: number
  marks: Record<string, number>
  errors: string[]
  totals: {
    bots: number
    open_positions: number
    equity: number
    realized_pnl: number
    unrealized_pnl: number
  }
  bots: PaperBot[]
  trades: PaperTrade[]
}

export interface PaperRosterEntry {
  pair: string
  label: string
  bot_id: number
  run_id: string
  seed: number | null
  test_sharpe: number | null
  born_gen: number | null
  rules: string
  stress: { key: string; label: string; ret_pct: number; maxdd_pct: number; trades: number; dead: boolean }[]
}

export interface PaperPayload {
  error?: string
  created?: string
  tape?: 'spot' | 'perp'
  paper_start_ms?: number
  criteria?: string
  start_capital?: number
  roster?: PaperRosterEntry[]
  state?: PaperState | null
  stale_s?: number
}

// ---- Binance account (read-only) -------------------------------------------

export interface BinanceAssetRow {
  asset: string
  earn: boolean
  free: number
  locked: number
  total: number
  price_usd: number | null
  value_usd: number | null
}

export interface BinanceWallet {
  assets: BinanceAssetRow[]
  value_usd: number
  dust_hidden: number
}

export interface BinancePosition {
  symbol: string
  market?: string
  side: string
  contracts: number
  notional: number
  leverage: number
  entry_price: number
  mark_price: number
  liq_price: number | null
  unrealized_pnl: number
  margin_mode: string | null
}

export interface BinanceFutures {
  wallet_usd: number
  unrealized_pnl: number
  margin_balance_usd: number
  available_usd: number
  assets: { asset: string; wallet: number; unrealized_pnl: number; margin_balance: number }[]
  positions: BinancePosition[]
}

export interface BinanceOrder {
  symbol: string
  market?: string
  side: string
  type: string
  price: number | null
  trigger?: number | null
  amount: number
  filled: number
  reduce_only?: boolean
  created_ms: number | null
}

export interface BinancePM {
  assets: { asset: string; wallet: number; um_upnl: number; cm_upnl: number; value_usd: number | null }[]
  value_usd: number
  equity_usd: number
  actual_equity_usd: number
  uni_mmr: number | null
  status: string | null
  positions: BinancePosition[]
  open_orders: BinanceOrder[]
}

export interface BinanceRestrictions {
  ip_restrict: boolean
  enable_reading: boolean
  enable_spot_trading: boolean
  enable_margin: boolean
  enable_futures: boolean
  enable_portfolio_margin: boolean
  enable_withdrawals: boolean
  enable_internal_transfer: boolean
  permits_universal_transfer: boolean
  created_ms: number
}

export interface BinancePayload {
  configured: boolean
  error?: string
  generated_ms?: number
  errors?: string[]
  public_ip?: string | null
  futures_off?: boolean
  restrictions?: BinanceRestrictions | null
  spot?: BinanceWallet | null
  funding?: BinanceWallet | null
  futures?: BinanceFutures | null
  portfolio_margin?: BinancePM | null
  open_orders?: BinanceOrder[] | null
  total_value_usd?: number
}
