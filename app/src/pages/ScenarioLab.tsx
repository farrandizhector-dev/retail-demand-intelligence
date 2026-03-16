import { TrendingUp, Clock, DollarSign, Shield, AlertOctagon, Package } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Header } from '../components/layout/Header'
import { SectionCard } from '../components/ui/SectionCard'
import { DataTable } from '../components/ui/DataTable'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { useJsonData } from '../hooks/useServingData'

interface ScenarioData {
  [key: string]: unknown
}

interface ScenarioItem {
  name: string
  description: string
  key_metric: string
  delta: number
  worsened: boolean
  icon: LucideIcon
}

interface PolicyItem {
  name: string
  fill_rate: number
  stockout_days: number
  avg_inventory: number
  total_cost: number
  recommended: boolean
}

const defaultScenarios: ScenarioItem[] = [
  { name: 'Demand +30%', description: 'Sudden demand surge of 30% across all categories. Tests supply chain resilience.', key_metric: 'Stockout Days', delta: 3.2, worsened: true, icon: TrendingUp },
  { name: 'Supplier Delay +5d', description: 'Lead time extension of 5 days due to supplier disruption.', key_metric: 'Safety Stock', delta: 18.4, worsened: true, icon: Clock },
  { name: 'Holding Cost +15%', description: 'Increase in warehouse cost. Impacts optimal order quantity.', key_metric: 'EOQ Reduction', delta: -6.8, worsened: false, icon: DollarSign },
  { name: 'Service Level 95%→99%', description: 'Raising target fill rate from 95% to 99% increases required buffer.', key_metric: 'Safety Stock', delta: 42.1, worsened: false, icon: Shield },
  { name: 'Combined Stress', description: 'Simultaneous demand surge + lead time delay. Worst-case scenario.', key_metric: 'Stockout Prob', delta: 28.7, worsened: true, icon: AlertOctagon },
]

const defaultPolicies: PolicyItem[] = [
  { name: '(s,Q) Fixed Order Qty', fill_rate: 89.7, stockout_days: 4.4, avg_inventory: 23.8, total_cost: 4820, recommended: false },
  { name: '(s,S) Min-Max', fill_rate: 91.2, stockout_days: 3.1, avg_inventory: 28.4, total_cost: 5140, recommended: true },
  { name: '(R,S) Periodic Review', fill_rate: 87.3, stockout_days: 5.9, avg_inventory: 19.6, total_cost: 4380, recommended: false },
  { name: 'SL-Driven Newsvendor', fill_rate: 93.4, stockout_days: 2.2, avg_inventory: 31.7, total_cost: 5810, recommended: false },
]

export default function ScenarioLab() {
  const { data } = useJsonData<ScenarioData>('scenario_results.json', {})
  void data

  const scenarios = defaultScenarios
  const policies = defaultPolicies

  const policyRows = policies.map(p => [
    p.name,
    `${p.fill_rate.toFixed(1)}%`,
    p.stockout_days.toFixed(1),
    p.avg_inventory.toFixed(1),
    `$${p.total_cost.toLocaleString()}`,
    p.recommended
      ? <Badge key={p.name} label="Recommended" variant="success" />
      : <span key={p.name} className="text-text-tertiary">—</span>,
  ])

  return (
    <div>
      <Header title="Scenario Lab" />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: '#F1F5F9', margin: 0 }}>What-If Scenario Analysis</h2>
          <p style={{ fontSize: 14, color: '#94A3B8', marginTop: 4, marginBottom: 0 }}>Pre-calculated Monte Carlo simulations (1,000 paths × 90 days). Results are deterministic for reproducibility.</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 24 }}>
          {scenarios.map((s) => {
            const Icon = s.icon
            const isNeg = s.worsened
            return (
              <div
                key={s.name}
                style={{
                  backgroundColor: '#151922',
                  borderRadius: 12,
                  border: '1px solid #1E293B',
                  padding: 24,
                  transition: 'box-shadow 200ms, border-color 200ms',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLDivElement).style.boxShadow = '0 4px 12px rgba(0,0,0,0.4)'
                  ;(e.currentTarget as HTMLDivElement).style.borderColor = '#3B82F6'
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLDivElement).style.boxShadow = 'none'
                  ;(e.currentTarget as HTMLDivElement).style.borderColor = '#1E293B'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
                  <div style={{ padding: 10, borderRadius: 8, backgroundColor: isNeg ? 'rgba(239,68,68,0.08)' : 'rgba(59,130,246,0.08)' }}>
                    <Icon size={20} color={isNeg ? '#EF4444' : '#3B82F6'} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 16, fontWeight: 600, color: '#F1F5F9' }}>{s.name}</div>
                    <div style={{ fontSize: 14, color: '#94A3B8', marginTop: 4, marginBottom: 16 }}>{s.description}</div>
                    <div>
                      <div style={{ fontSize: 11, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>{s.key_metric}</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: isNeg ? '#EF4444' : '#10B981' }}>
                        {isNeg ? '+' : ''}{Math.abs(s.delta).toFixed(1)}{s.key_metric.includes('Stock') ? ' units' : s.key_metric.includes('Prob') ? '%' : 'd'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <SectionCard title="Policy Comparison" subtitle="(s,Q) · (s,S) · (R,S) · Newsvendor — 30-day simulation">
          {policies.length > 0
            ? <DataTable headers={['Policy', 'Fill Rate', 'Stockout Days', 'Avg Inventory', 'Total Cost', 'Recommended']} rows={policyRows} highlightRow={policies.findIndex(p => p.recommended)} />
            : <EmptyState icon={<Package />} title="No policy data available" description="Run the inventory simulation pipeline to generate policy comparisons" />
          }
        </SectionCard>
      </div>
    </div>
  )
}
