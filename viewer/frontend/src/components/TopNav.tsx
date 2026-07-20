import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'

const LINKS = [
  ['/', 'Dashboard'],
  ['/chart', 'Chart'],
  ['/swarm', 'Swarm'],
  ['/evolution', 'Evolution'],
  ['/paper', 'Paper'],
  ['/binance', 'Binance'],
] as const

export default function TopNav({ children }: { children?: ReactNode }) {
  const { pathname } = useLocation()
  const active = (to: string) => (to === '/' ? pathname === '/' : pathname.startsWith(to))
  return (
    <header className="topnav">
      <Link to="/" className="brand">
        <span className="logo">▲</span>
        strategy-lab
      </Link>
      <nav>
        {LINKS.map(([to, label]) => (
          <Link key={to} to={to} className={active(to) ? 'active' : ''}>
            {label}
          </Link>
        ))}
      </nav>
      <span className="spacer" />
      {children}
    </header>
  )
}
