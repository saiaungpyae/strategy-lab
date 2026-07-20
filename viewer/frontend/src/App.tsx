import { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/dashboard/Dashboard'

// every non-landing page is code-split so the dashboard bundle stays small
const ChartView = lazy(() => import('./pages/chart/ChartView'))
const Swarm = lazy(() => import('./pages/swarm/Swarm'))
const Evolution = lazy(() => import('./pages/evolution/Evolution'))
const EvoBots = lazy(() => import('./pages/evolution/EvoBots'))
const EvoBot = lazy(() => import('./pages/evolution/EvoBot'))
const Paper = lazy(() => import('./pages/paper/Paper'))
const Binance = lazy(() => import('./pages/binance/Binance'))

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chart" element={<ChartView />} />
          <Route path="/swarm" element={<Swarm />} />
          <Route path="/evolution" element={<Evolution />} />
          <Route path="/evolution/bots" element={<EvoBots />} />
          <Route path="/evolution/bot" element={<EvoBot />} />
          <Route path="/paper" element={<Paper />} />
          <Route path="/binance" element={<Binance />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}
