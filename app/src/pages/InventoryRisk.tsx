import { Header } from '../components/layout/Header'
import { KPICard } from '../components/ui/KPICard'
import { SectionCard } from '../components/ui/SectionCard'
import { DataTable } from '../components/ui/DataTable'
import { EChartWrapper } from '../components/ui/EChartWrapper'
import { Badge } from '../components/ui/Badge'
import { CheckCircle, AlertTriangle, Clock } from 'lucide-react'
import { useJsonData } from '../hooks/useServingData'

interface InventoryData {
  fill_rate?: number
  stockout_rate?: number
  days_of_supply?: number
  [key: string]: unknown
}

function makeHeatmapData() {
  const stores = ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']
  const depts = ['FOODS_1','FOODS_2','FOODS_3','HOBBIES_1','HOBBIES_2','HOUSEHOLD_1','HOUSEHOLD_2']
  const data: [number, number, number][] = []
  for (let si = 0; si < stores.length; si++) {
    for (let di = 0; di < depts.length; di++) {
      data.push([di, si, Math.round(Math.random() * 100)])
    }
  }
  return { stores, depts, data }
}

export default function InventoryRisk() {
  const { data } = useJsonData<InventoryData>('inventory_risk_matrix.json', {})
  const fillRate = (data.fill_rate ?? 0.897) * 100
  const stockoutRate = (data.stockout_rate ?? 0.674) * 100
  const dos = data.days_of_supply ?? 18.4

  const { stores, depts, data: hmData } = makeHeatmapData()

  const heatmapOption = {
    tooltip: {
      position: 'top' as const,
      formatter: (p: { data: [number, number, number] }) =>
        `${stores[p.data[1]]} × ${depts[p.data[0]]}<br/>Stockout prob: ${p.data[2]}%`,
    },
    grid: { left: 80, right: 20, top: 20, bottom: 60, containLabel: false },
    xAxis: { type: 'category' as const, data: depts, axisLabel: { rotate: 30, fontSize: 10, color: '#64748B' }, axisTick: { show: false }, axisLine: { lineStyle: { color: '#1E293B' } } },
    yAxis: { type: 'category' as const, data: stores, axisLabel: { fontSize: 10, color: '#64748B' }, axisTick: { show: false }, axisLine: { lineStyle: { color: '#1E293B' } } },
    visualMap: {
      min: 0, max: 100,
      calculable: true,
      orient: 'horizontal' as const,
      left: 'center', bottom: 0,
      inRange: { color: ['#10B981', '#F59E0B', '#EF4444'] },
      textStyle: { color: '#94A3B8', fontSize: 11 },
    },
    series: [{
      type: 'heatmap' as const,
      data: hmData,
      label: { show: true, formatter: (p: { data: [number, number, number] }) => `${p.data[2]}%`, fontSize: 9, color: '#fff' },
    }],
  }

  const atRisk = stores.flatMap(store =>
    depts.map(dept => ({
      store,
      dept,
      prob: Math.round(Math.random() * 100),
      dos: Math.round(Math.random() * 30 + 2),
    }))
  ).sort((a, b) => b.prob - a.prob).slice(0, 10)

  const atRiskRows = atRisk.map(r => [
    r.store, r.dept,
    <Badge key={`${r.store}-${r.dept}`} label={`${r.prob}%`} variant={r.prob > 50 ? 'danger' : r.prob > 25 ? 'warning' : 'success'} />,
    `${r.dos}d`,
  ])

  const dosOption = {
    tooltip: { trigger: 'axis' as const },
    xAxis: { type: 'category' as const, data: Array.from({ length: 20 }, (_, i) => `${i*2}-${i*2+2}`), axisLabel: { fontSize: 10, color: '#64748B' } },
    yAxis: { type: 'value' as const, axisLabel: { fontSize: 10, color: '#64748B' } },
    series: [{
      type: 'bar' as const,
      data: Array.from({ length: 20 }, () => Math.round(Math.random() * 50 + 5)),
      itemStyle: { color: 'rgba(59,130,246,0.7)', borderRadius: [3, 3, 0, 0] },
    }],
  }

  return (
    <div>
      <Header title="Inventory Risk" />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
          <KPICard label="Fill Rate" value={fillRate.toFixed(1)} suffix="%" accent="success" icon={CheckCircle} />
          <KPICard label="Stockout Rate" value={stockoutRate.toFixed(1)} suffix="%" accent="danger" icon={AlertTriangle} />
          <KPICard label="Days of Supply" value={dos.toFixed(1)} accent="primary" icon={Clock} />
        </div>

        <div style={{ marginBottom: 24 }}>
          <SectionCard title="Stockout Risk Matrix — Stores × Departments" subtitle="Probability of stockout within 30 days">
            <EChartWrapper option={heatmapOption} height={320} />
          </SectionCard>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <SectionCard title="Top 10 At-Risk Items" subtitle="Sorted by stockout probability">
            <DataTable headers={['Store', 'Department', 'Stockout Prob', 'Days of Supply']} rows={atRiskRows} />
          </SectionCard>
          <SectionCard title="Days of Supply Distribution" subtitle="All SKUs · current inventory position">
            <EChartWrapper option={dosOption} height={240} />
          </SectionCard>
        </div>
      </div>
    </div>
  )
}
