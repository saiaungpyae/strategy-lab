import CanvasBox from '../../components/CanvasBox'
import { hbars } from '../../lib/canvas'
import { fmt } from '../../lib/format'
import type { Recap } from '../../types'

export default function TraitsTab({ recap }: { recap: Recap }) {
  const tr = recap.traits
  return (
    <div className="grid">
      <div className="card" style={{ gridColumn: 'span 6' }}>
        <h3>Trait → test-period Sharpe (rank correlation)</h3>
        <div className="sub">
          computed on UNSEEN test period only · {tr.n_used} pattern bots with ≥5 test trades
        </div>
        <CanvasBox
          draw={(el) => hbars(el, tr.numeric, 'rho', 'trait', 'spearman ρ vs test Sharpe')}
          deps={[recap]}
        />
      </div>
      <div className="card" style={{ gridColumn: 'span 6' }}>
        <h3>Feature usage → test-period Sharpe</h3>
        <div className="sub">
          does an ideology that watches this feature do better out-of-sample?
        </div>
        {tr.features.length ? (
          <CanvasBox
            draw={(el) =>
              hbars(el, tr.features, 'delta_vs_all', 'feature', 'median test Sharpe minus swarm median')
            }
            deps={[recap]}
          />
        ) : (
          <div className="muted">not enough bots per feature</div>
        )}
      </div>
      {tr.categorical.map((c) => (
        <div className="card" style={{ gridColumn: 'span 3' }} key={c.trait}>
          <h3>{c.trait}</h3>
          <table>
            <tbody>
              {c.groups.map((g) => (
                <tr key={g.value}>
                  <td className="l">{g.value}</td>
                  <td className="muted">{g.n}</td>
                  <td className={g.median_sharpe_b >= 0 ? 'pos' : 'neg'}>
                    {fmt(g.median_sharpe_b, 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}
