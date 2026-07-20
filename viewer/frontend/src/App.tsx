import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/dashboard/Dashboard'
import ChartView from './pages/chart/ChartView'
import Swarm from './pages/swarm/Swarm'
import Evolution from './pages/evolution/Evolution'
import EvoBots from './pages/evolution/EvoBots'
import EvoBot from './pages/evolution/EvoBot'
import Paper from './pages/paper/Paper'
import Binance from './pages/binance/Binance'

export default function App() {
  return (
    <BrowserRouter>
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
    </BrowserRouter>
  )
}
