import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { LoadingSpinner } from './components/ui/LoadingSpinner'

const ExecutiveOverview = lazy(() => import('./pages/ExecutiveOverview'))
const ForecastLab = lazy(() => import('./pages/ForecastLab'))
const InventoryRisk = lazy(() => import('./pages/InventoryRisk'))
const ProductDrilldown = lazy(() => import('./pages/ProductDrilldown'))
const ModelQuality = lazy(() => import('./pages/ModelQuality'))
const ScenarioLab = lazy(() => import('./pages/ScenarioLab'))
const About = lazy(() => import('./pages/About'))

function App() {
  return (
    <Layout>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<ExecutiveOverview />} />
          <Route path="/forecast" element={<ForecastLab />} />
          <Route path="/inventory" element={<InventoryRisk />} />
          <Route path="/product" element={<ProductDrilldown />} />
          <Route path="/quality" element={<ModelQuality />} />
          <Route path="/scenarios" element={<ScenarioLab />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}

export default App
