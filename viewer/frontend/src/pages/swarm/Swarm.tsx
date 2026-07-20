import { useCallback, useEffect, useRef, useState } from 'react'
import TopNav from '../../components/TopNav'
import { getJSON, postJSON } from '../../lib/api'
import type { Recap, StartResult, SwarmProgress, SwarmRunInfo, SwarmRunsPayload } from '../../types'
import OverviewTab from './OverviewTab'
import TraitsTab from './TraitsTab'
import PersistenceTab from './PersistenceTab'
import LeaderboardTab from './LeaderboardTab'
import BotsTab from './BotsTab'
import BotDialog from './BotDialog'

const TABS = [
  ['overview', 'Overview'],
  ['bots', 'Bots'],
  ['traits', 'Traits'],
  ['persistence', 'Persistence'],
  ['leaderboard', 'Leaderboard'],
] as const

type TabKey = (typeof TABS)[number][0]

export default function Swarm() {
  const [runs, setRuns] = useState<SwarmRunInfo[]>([])
  const [runId, setRunId] = useState('')
  const [recap, setRecap] = useState<Recap | null>(null)
  const [progress, setProgress] = useState<SwarmProgress | null>(null)
  const [tab, setTab] = useState<TabKey>('overview')
  const [newRunOpen, setNewRunOpen] = useState(false)
  const [dlgBot, setDlgBot] = useState<number | null>(null)
  const [loadedRuns, setLoadedRuns] = useState(false)

  // new-run form
  const [nrBots, setNrBots] = useState('5000')
  const [nrSince, setNrSince] = useState('')
  const [nrSplit, setNrSplit] = useState('0.7')
  const [nrCtrl, setNrCtrl] = useState('10')
  const [nrSeed, setNrSeed] = useState('42')

  useEffect(() => {
    document.title = 'strategy-lab · Bot Swarm'
  }, [])

  const loadRuns = useCallback(async (selectFirst = false): Promise<SwarmRunInfo[]> => {
    const data = await getJSON<SwarmRunsPayload>('/api/swarm/runs')
    setRuns(data.runs)
    setLoadedRuns(true)
    if (data.runs.length) {
      setRunId((cur) => (selectFirst || !cur ? data.runs[0].run_id : cur))
    }
    return data.runs
  }, [])

  useEffect(() => {
    loadRuns()
  }, [loadRuns])

  // load the recap for the selected run; poll every 2s while it is simulating
  useEffect(() => {
    if (!runId) return
    let alive = true
    let timer: number | undefined
    setRecap(null)
    setProgress(null)
    const poll = async () => {
      const r = await fetch(`/api/swarm/run?id=${encodeURIComponent(runId)}`)
      if (!alive) return
      if (r.status === 202) {
        const j = await r.json()
        setProgress(j.progress || {})
        timer = window.setTimeout(async () => {
          await loadRuns() // pick up stage changes in the run selector
          if (alive) poll()
        }, 2000)
        return
      }
      const j = await r.json()
      if (!alive) return
      setRecap(j)
      setProgress(null)
    }
    poll()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [runId, loadRuns])

  const startRun = async () => {
    const body = {
      bots: +nrBots,
      since: nrSince.trim(),
      split: +nrSplit,
      seed: +nrSeed,
      control_frac: +nrCtrl / 100,
    }
    const j = await postJSON<StartResult>('/api/swarm/start', body)
    if (!j.started) {
      alert(j.error || 'failed')
      return
    }
    setNewRunOpen(false)
    setTimeout(() => loadRuns(true), 1500)
  }

  const progressText = progress
    ? `⏳ ${progress.stage || 'starting'} ${progress.frac ? Math.round(progress.frac * 100) + '%' : ''} (${progress.elapsed_s || 0}s)`
    : ''

  const startCapital = recap ? +recap.config.start_capital : 10000

  return (
    <div className="swarm">
      <TopNav>
        <span className="refresh-status">{progressText}</span>
      </TopNav>
      <div className="toolbar">
        <span className="crumb">Bot Swarm</span>
        <select
          title="run"
          value={runId}
          onChange={(e) => {
            setRunId(e.target.value)
          }}
        >
          {runs.length ? (
            runs.map((x) => (
              <option key={x.run_id} value={x.run_id}>
                {x.run_id} · {x.bots ?? '?'} bots
                {x.has_recap ? '' : ' · ' + (x.stage || 'running')}
              </option>
            ))
          ) : (
            <option value="">no runs yet</option>
          )}
        </select>
        <button className="primary" onClick={() => setNewRunOpen((o) => !o)}>
          New run…
        </button>
      </div>
      <div className={'top newrun' + (newRunOpen ? ' open' : '')}>
        <label>
          bots{' '}
          <input type="number" min={100} max={20000} value={nrBots} onChange={(e) => setNrBots(e.target.value)} />
        </label>
        <label>
          since{' '}
          <input placeholder="2024-01-01" style={{ width: 110 }} value={nrSince} onChange={(e) => setNrSince(e.target.value)} />
        </label>
        <label>
          train split{' '}
          <input type="number" step={0.05} min={0.5} max={0.9} value={nrSplit} onChange={(e) => setNrSplit(e.target.value)} />
        </label>
        <label>
          control %{' '}
          <input type="number" min={0} max={50} value={nrCtrl} onChange={(e) => setNrCtrl(e.target.value)} />
        </label>
        <label>
          seed <input type="number" value={nrSeed} onChange={(e) => setNrSeed(e.target.value)} />
        </label>
        <button className="primary" onClick={startRun}>
          Run swarm
        </button>
        <span className="muted">runs as its own process; artifacts land in reports/swarm/</span>
      </div>
      <div className="tabs">
        {TABS.map(([key, label]) => (
          <button
            key={key}
            className={tab === key ? 'active' : ''}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="wrap">
        <SwarmBody
          tab={tab}
          runId={runId}
          recap={recap}
          progress={progress}
          loadedRuns={loadedRuns}
          openBot={setDlgBot}
        />
      </div>
      {dlgBot != null && (
        <BotDialog
          runId={runId}
          botId={dlgBot}
          startCapital={startCapital}
          onClose={() => setDlgBot(null)}
        />
      )}
    </div>
  )
}

function SwarmBody({
  tab,
  runId,
  recap,
  progress,
  loadedRuns,
  openBot,
}: {
  tab: TabKey
  runId: string
  recap: Recap | null
  progress: SwarmProgress | null
  loadedRuns: boolean
  openBot: (id: number) => void
}) {
  if (loadedRuns && !runId)
    return (
      <div className="empty">
        No swarm runs yet — hit <b>New run…</b> or run <code>sl-swarm run</code> from the terminal.
      </div>
    )
  if (progress)
    return (
      <div className="empty">
        Simulating… {progress.stage || ''}{' '}
        {progress.frac ? Math.round(progress.frac * 100) + '%' : ''}
      </div>
    )
  if (!recap) return <div className="empty">loading…</div>
  switch (tab) {
    case 'overview':
      return <OverviewTab recap={recap} />
    case 'bots':
      return <BotsTab runId={runId} openBot={openBot} />
    case 'traits':
      return <TraitsTab recap={recap} />
    case 'persistence':
      return <PersistenceTab recap={recap} />
    case 'leaderboard':
      return <LeaderboardTab recap={recap} openBot={openBot} />
  }
}
