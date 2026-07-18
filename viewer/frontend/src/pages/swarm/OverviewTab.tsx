import CanvasBox from '../../components/CanvasBox'
import { histogram, lines } from '../../lib/canvas'
import { C } from '../../lib/colors'
import { fmt } from '../../lib/format'
import type { Recap } from '../../types'

function Tile({
  label,
  value,
  note,
  cls,
}: {
  label: string
  value: string
  note?: string
  cls?: string
}) {
  return (
    <div className="card tile" style={{ gridColumn: 'span 2' }}>
      <h3>{label}</h3>
      <div className={`big ${cls || ''}`}>{value}</div>
      <div className="note">{note || ''}</div>
    </div>
  )
}

export default function OverviewTab({ recap }: { recap: Recap }) {
  const t = recap.tiles
  const cfg = recap.config
  const gap = t.rank_corr_gap
  const sv = recap.survival
  let splitIdx = sv.days.indexOf(sv.split_day)
  if (splitIdx < 0) splitIdx = sv.days.findIndex((d) => d >= sv.split_day)

  return (
    <div className="grid">
      <Tile
        label="Bots"
        value={t.n_bots.toLocaleString()}
        note={`${t.n_control} random control · seed ${cfg.seed}`}
      />
      <Tile
        label="Alive"
        value={fmt(t.alive_pct, 1) + '%'}
        note={`ruin line at ${cfg.ruin_frac * 100}% of capital`}
        cls={t.alive_pct > 60 ? 'pos' : 'neg'}
      />
      <Tile
        label="Above water"
        value={fmt(t.above_water_pct, 1) + '%'}
        note={`median x${fmt(t.median_final_mult, 3)} · B&H x${fmt(t.bnh_mult, 3)}`}
        cls={t.above_water_pct > 40 ? 'pos' : 'neg'}
      />
      <Tile
        label="Luck yardstick"
        value={'S ' + fmt(t.yardstick_sharpe_p95, 2)}
        note={`test-Sharpe p95 of RANDOM bots · p99 = ${fmt(t.yardstick_sharpe_p99, 2)} — beat this or it's luck`}
        cls="warn"
      />
      <Tile
        label="Persistence gap"
        value={gap == null ? '—' : fmt(gap, 3)}
        note={`pattern ρ ${fmt(t.rank_corr_pattern, 2)} − control ρ ${fmt(t.rank_corr_control, 2)} · skill must exceed cost-drag persistence`}
        cls={gap != null && gap > 0.05 ? 'pos' : ''}
      />
      <Tile
        label="Span"
        value={cfg.span[0].slice(0, 10)}
        note={`→ ${cfg.span[1].slice(0, 10)} · split ${cfg.split_date.slice(0, 10)}`}
      />
      <div className="card" style={{ gridColumn: 'span 8' }}>
        <h3>Final equity distribution</h3>
        <div className="sub">
          multiplier of starting capital (${(+cfg.start_capital).toLocaleString()}) · costs: taker{' '}
          {cfg.taker_bps}bps / maker edge {cfg.maker_bps}bps
        </div>
        <CanvasBox draw={(el) => histogram(el, recap.histogram)} deps={[recap]} />
        <div className="legend" style={{ marginTop: 6 }}>
          <span>
            <i style={{ background: 'rgba(38,166,154,.6)' }} />
            pattern bots
          </span>
          <span>
            <i style={{ background: C.muted }} />
            random control (outline)
          </span>
          <span>
            <i style={{ background: C.warn }} />
            buy &amp; hold
          </span>
        </div>
      </div>
      <div className="card" style={{ gridColumn: 'span 4' }}>
        <h3>Survival</h3>
        <div className="sub">% of bots above starting capital · shaded: pattern IQR of equity</div>
        <CanvasBox
          draw={(el) =>
            lines(
              el,
              sv.days,
              [
                { v: sv.pattern.above_water, c: C.up },
                { v: sv.control.above_water, c: C.muted },
              ],
              220,
              (v) => fmt(v, 0) + '%',
              undefined,
              splitIdx,
            )
          }
          deps={[recap]}
        />
        <div className="legend" style={{ marginTop: 6 }}>
          <span>
            <i style={{ background: C.up }} />
            pattern
          </span>
          <span>
            <i style={{ background: C.muted }} />
            control
          </span>
        </div>
      </div>
      {recap.regimes && recap.regimes.rows.length > 0 && (
        <div className="card" style={{ gridColumn: 'span 6' }}>
          <h3>Regime breakdown (test period)</h3>
          <div className="sub">{recap.regimes.basis} · median bot daily return</div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th className="l">regime</th>
                  <th>days</th>
                  <th>B&amp;H bps/day</th>
                  <th>pattern bps/day</th>
                  <th>control bps/day</th>
                </tr>
              </thead>
              <tbody>
                {recap.regimes.rows.map((r) => (
                  <tr key={r.regime}>
                    <td className="l">{r.regime}</td>
                    <td>{r.days}</td>
                    <td className={r.bnh_bps_day >= 0 ? 'pos' : 'neg'}>{r.bnh_bps_day}</td>
                    <td className={r.pattern_bps_day >= 0 ? 'pos' : 'neg'}>{r.pattern_bps_day}</td>
                    <td className="muted">{r.control_bps_day}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
