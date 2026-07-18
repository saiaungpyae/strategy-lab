import CanvasBox from '../../components/CanvasBox'
import { scatter, vbars } from '../../lib/canvas'
import { fmt } from '../../lib/format'
import type { Recap } from '../../types'

export default function PersistenceTab({ recap }: { recap: Recap }) {
  const p = recap.persistence
  return (
    <div className="grid">
      <div className="card" style={{ gridColumn: 'span 8' }}>
        <h3>Does the train-period ranking survive the test period?</h3>
        <div className="sub">
          each dot is one bot with ≥8 trades in both periods · pattern ρ ={' '}
          <b>{fmt(p.rho_pattern, 3)}</b>, control ρ = <b>{fmt(p.rho_control, 3)}</b>. Control
          persistence is cost-drag (frequent traders bleed in both periods) — skill is the amount
          pattern exceeds control.
        </div>
        <CanvasBox draw={(el) => scatter(el, p.scatter)} deps={[recap]} />
      </div>
      <div className="card" style={{ gridColumn: 'span 4' }}>
        <h3>Where did train's top decile land?</h3>
        <div className="sub">
          test-period decile of the bots that ranked top-10% in training · flat = luck, right-heavy
          = something persisted
        </div>
        {p.top_decile_dest ? (
          <CanvasBox
            draw={(el) =>
              vbars(
                el,
                p.top_decile_dest!,
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                'rgba(38,166,154,.7)',
              )
            }
            deps={[recap]}
          />
        ) : (
          <div className="muted">not enough rankable bots</div>
        )}
      </div>
    </div>
  )
}
