// Typed shapes of the JSON payloads served by viewer/server.py.

export interface DatasetInfo {
  file: string
  exchange: string
  symbol: string
  timeframe: string
  label: string
}

export interface HealthDataset extends DatasetInfo {
  rows: number
  first: number
  last: number
  age_seconds: number
  bars_behind: number | null
  gaps: number
  size_bytes: number
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
  final_test?: Record<string, EvoGroupStats>
  hof_top?: HofRow[]
}

export interface EvosPayload {
  evos: EvoRun[]
  running: boolean
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
  bot: EvoBotRecord
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
