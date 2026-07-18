import { useEffect, useRef, useState } from 'react'
import CanvasBox from '../../components/CanvasBox'
import { lines } from '../../lib/canvas'
import { C } from '../../lib/colors'
import { getJSON } from '../../lib/api'
import { fmt, money } from '../../lib/format'
import type { BotDetail } from '../../types'

function KvRows({ o, keys }: { o: Record<string, unknown>; keys: string[] }) {
  return (
    <>
      {keys.map((k) => {
        let v = o[k]
        if (typeof v === 'number' && !Number.isInteger(v)) v = fmt(v, 3)
        return (
          <div style={{ display: 'contents' }} key={k}>
            <b>{k}</b>
            <span>{v == null ? '—' : String(v)}</span>
          </div>
        )
      })}
    </>
  )
}

const GENOME_KEYS = [
  'tf', 'dir_bias', 'risk_pct', 'stop_atr', 'tp_rr', 'max_hold_bars',
  'maker_off_atr', 'order_ttl', 'session', 'loss_react', 'cooldown_bars',
  'revenge_mult', 'reentry_gap',
]
const RESULT_KEYS = ['trades_a', 'trades_b', 'sharpe_a', 'sharpe_b', 'maxdd_b', 'death_day']

export default function BotDialog({
  runId,
  botId,
  startCapital,
  onClose,
}: {
  runId: string
  botId: number
  startCapital: number
  onClose: () => void
}) {
  const dlgRef = useRef<HTMLDialogElement>(null)
  const [bot, setBot] = useState<BotDetail | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    dlgRef.current?.showModal()
  }, [])

  useEffect(() => {
    let alive = true
    setBot(null)
    setError('')
    getJSON<BotDetail>(`/api/swarm/bot?id=${encodeURIComponent(runId)}&bot=${botId}`).then((b) => {
      if (!alive) return
      if (b.error) setError(b.error)
      else setBot(b)
    })
    return () => {
      alive = false
    }
  }, [runId, botId])

  const g = bot?.genome
  const res = bot?.result
  const finalMult = res?.final_mult == null ? null : Number(res.final_mult)
  const finalUsd = finalMult == null ? null : finalMult * startCapital
  const splitIdx = bot ? bot.days.findIndex((d) => d >= bot.split_day) : -1

  return (
    <dialog ref={dlgRef} onClose={onClose}>
      {error && <div>{error}</div>}
      {!error && !bot && <div className="muted">loading bot…</div>}
      {bot && g && res && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <h2 style={{ margin: 0, fontSize: 16 }}>
              Bot #{botId}
              {Boolean(g.is_control) && <span className="chip gray">CONTROL</span>}
              {Boolean(res.dead) && <span className="chip red">DEAD</span>}
            </h2>
            <span>
              <span
                className={(finalUsd ?? 0) >= startCapital ? 'pos' : 'neg'}
                style={{ fontSize: 17, fontWeight: 650 }}
              >
                {money(finalUsd)}
              </span>
              <span className="muted">
                {' '}
                of {money(startCapital)} (x{fmt(finalMult, 3)})
              </span>
              <button style={{ marginLeft: 12 }} onClick={() => dlgRef.current?.close()}>
                close
              </button>
            </span>
          </div>
          <p className="rules" style={{ maxWidth: 'none', whiteSpace: 'normal' }}>
            <b>Ideology:</b> {String(g.rules ?? '')}
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 18 }}>
            <div className="kv">
              <KvRows o={g} keys={GENOME_KEYS} />
              <b>—</b>
              <span></span>
              <KvRows o={res} keys={RESULT_KEYS} />
            </div>
            <div>
              <h3 style={{ margin: '0 0 6px', fontSize: 12, color: 'var(--muted)' }}>
                EQUITY (train | test)
              </h3>
              <CanvasBox
                draw={(el) =>
                  lines(
                    el,
                    bot.days,
                    [{ v: bot.equity, c: C.up }],
                    260,
                    (v) => 'x' + fmt(v / startCapital, 2),
                    undefined,
                    splitIdx,
                  )
                }
                deps={[bot, startCapital]}
              />
              <h3 style={{ margin: '12px 0 6px', fontSize: 12, color: 'var(--muted)' }}>
                YEAR BY YEAR
              </h3>
              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      {(bot.yearly || []).map((y) => (
                        <th key={y.year}>{y.year}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      {(bot.yearly || []).map((y) => (
                        <td key={y.year} className={y.ret_pct >= 0 ? 'pos' : 'neg'}>
                          {fmt(y.ret_pct, 1)}%
                        </td>
                      ))}
                    </tr>
                    <tr>
                      {(bot.yearly || []).map((y) => (
                        <td key={y.year} className="muted">
                          {money(y.end_usd)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}
    </dialog>
  )
}
