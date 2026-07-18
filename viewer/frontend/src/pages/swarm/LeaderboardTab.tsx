import { fmt, money } from '../../lib/format'
import type { Recap } from '../../types'

export default function LeaderboardTab({
  recap,
  openBot,
}: {
  recap: Recap
  openBot: (id: number) => void
}) {
  return (
    <>
      <div className="banner">
        ⚠ Entertainment only — among {recap.tiles.n_bots.toLocaleString()} bots the best single
        Sharpe is expected to be extreme by pure luck. The Traits and Persistence tabs are the
        science. Test-period columns are emphasized; train is grayed.
      </div>
      <div className="card scroll" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>bot</th>
              <th className="l">ideology (sampled rules)</th>
              <th>tf</th>
              <th>session</th>
              <th>risk</th>
              <th>S(train)</th>
              <th>S(test)</th>
              <th>final $</th>
              <th>ret(train)</th>
              <th>ret(test)</th>
              <th>maxDD(test)</th>
              <th>trades</th>
              <th>expo(test)</th>
            </tr>
          </thead>
          <tbody>
            {recap.leaderboard.map((r, i) => (
              <tr className="click" key={r.bot_id} onClick={() => openBot(r.bot_id)}>
                <td>{i + 1}</td>
                <td>
                  #{r.bot_id}
                  {r.control && <span className="chip gray">CONTROL</span>}
                  {r.dead && <span className="chip red">DEAD</span>}
                </td>
                <td className="l rules" title={r.rules}>
                  {r.rules}
                </td>
                <td>{r.tf}</td>
                <td>{r.session}</td>
                <td>{fmt(r.risk_pct, 2)}%</td>
                <td className="muted">{fmt(r.sharpe_a, 2)}</td>
                <td>
                  <b className={(r.sharpe_b ?? 0) >= 0 ? 'pos' : 'neg'}>{fmt(r.sharpe_b, 2)}</b>
                </td>
                <td className={(r.final_usd ?? 0) >= recap.config.start_capital ? 'pos' : 'neg'}>
                  {r.final_usd == null ? '—' : money(r.final_usd)}
                </td>
                <td className="muted">{fmt(r.ret_a, 1)}%</td>
                <td className={(r.ret_b ?? 0) >= 0 ? 'pos' : 'neg'}>{fmt(r.ret_b, 1)}%</td>
                <td>{fmt(r.maxdd_b, 1)}%</td>
                <td>{r.trades}</td>
                <td>{fmt(r.expo_b_pct, 0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
