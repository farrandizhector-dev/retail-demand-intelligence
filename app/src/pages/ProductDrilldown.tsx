import { useState } from 'react'
import type React from 'react'
import { Package } from 'lucide-react'
import { Header } from '../components/layout/Header'
import { SectionCard } from '../components/ui/SectionCard'
import { EmptyState } from '../components/ui/EmptyState'
import { EChartWrapper } from '../components/ui/EChartWrapper'
import { Badge } from '../components/ui/Badge'
import { COLORS } from '../utils/echartTheme'

const STORES = ['CA_1','CA_2','CA_3','TX_1','TX_2','WI_1']
const ITEMS = ['FOODS_3_001','FOODS_3_002','HOBBIES_1_001','HOUSEHOLD_1_001']

function makeSkuData(item: string, store: string) {
  void item
  void store
  const dates: string[] = []
  const actual: (number | null)[] = []
  const forecast: number[] = []
  const p10: number[] = []
  const p90: number[] = []
  const inventory: number[] = []
  let base = 5 + Math.random() * 5
  let inv = 50
  const start = new Date('2016-04-01')
  for (let i = 0; i < 56; i++) {
    const d = new Date(start); d.setDate(d.getDate() + i)
    dates.push(d.toISOString().slice(0, 10))
    const v = Math.max(0, base + (Math.random() - 0.5) * 3)
    actual.push(i < 28 ? Math.round(v * 10) / 10 : null)
    const pred = Math.max(0, base + (Math.random() - 0.5) * 2)
    forecast.push(Math.round(pred * 10) / 10)
    p10.push(Math.max(0, pred - 2))
    p90.push(pred + 3)
    inv = Math.max(0, inv - v + (i % 14 === 0 ? 30 : 0))
    inventory.push(Math.round(inv))
    base += (Math.random() - 0.5) * 0.3
  }
  return { dates, actual, forecast, p10, p90, inventory }
}

export default function ProductDrilldown() {
  const [store, setStore] = useState('')
  const [item, setItem] = useState('')
  const hasData = store && item
  const sku = hasData ? makeSkuData(item, store) : null

  const ropVal = 15
  const ssVal = 8

  const forecastOption = sku ? {
    tooltip: { trigger: 'axis' as const },
    legend: { data: ['Actual', 'Forecast'] },
    xAxis: { type: 'category' as const, data: sku.dates, axisLabel: { rotate: 30, fontSize: 9 }, boundaryGap: false },
    yAxis: { type: 'value' as const, name: 'Units' },
    series: [
      { name: 'Interval', type: 'line' as const, data: sku.p90, lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(59,130,246,0.08)' }, stack: 'ci', symbol: 'none' },
      { name: 'p10', type: 'line' as const, data: sku.p10, lineStyle: { opacity: 0 }, areaStyle: { color: '#0B0E14' }, stack: 'ci', symbol: 'none', legendHoverLink: false },
      { name: 'Actual', type: 'line' as const, data: sku.actual, lineStyle: { color: COLORS.slate, width: 2 }, itemStyle: { color: COLORS.slate }, symbol: 'none' },
      { name: 'Forecast', type: 'line' as const, data: sku.forecast, lineStyle: { color: COLORS.blue, width: 2 }, itemStyle: { color: COLORS.blue }, symbol: 'none' },
    ],
  } : {}

  const invOption = sku ? {
    tooltip: { trigger: 'axis' as const },
    xAxis: { type: 'category' as const, data: sku.dates.slice(0, 28), axisLabel: { rotate: 30, fontSize: 9 } },
    yAxis: { type: 'value' as const, name: 'Units' },
    series: [
      {
        name: 'Inventory',
        type: 'line' as const,
        data: sku.inventory.slice(0, 28),
        smooth: true,
        lineStyle: { color: COLORS.blue, width: 2 },
        itemStyle: { color: COLORS.blue },
        areaStyle: { color: 'rgba(59,130,246,0.06)' },
        symbol: 'none',
        markLine: {
          data: [
            { yAxis: ropVal, name: 'Reorder Point', lineStyle: { color: COLORS.yellow, type: 'dashed' as const }, label: { formatter: 'ROP', color: COLORS.yellow } },
            { yAxis: ssVal, name: 'Safety Stock', lineStyle: { color: COLORS.red, type: 'dashed' as const }, label: { formatter: 'SS', color: COLORS.red } },
          ],
          symbol: 'none',
        },
      },
    ],
  } : {}

  const selectStyle: React.CSSProperties = {
    height: 36,
    backgroundColor: '#1E2432',
    border: '1px solid #1E293B',
    borderRadius: 8,
    fontSize: 14,
    color: '#94A3B8',
    padding: '0 12px',
    outline: 'none',
  }

  return (
    <div>
      <Header title="Product Drilldown" />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
          <select value={store} onChange={e => setStore(e.target.value)} style={selectStyle}>
            <option value="">Select Store</option>
            {STORES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={item} onChange={e => setItem(e.target.value)} style={selectStyle}>
            <option value="">Select Item</option>
            {ITEMS.map(i => <option key={i} value={i}>{i}</option>)}
          </select>
        </div>

        {!hasData ? (
          <div style={{ backgroundColor: '#151922', borderRadius: 12, border: '1px solid #1E293B' }}>
            <EmptyState icon={<Package />} title="Select a store and item to view SKU details" description="Use the dropdowns above to drill into a specific product" />
          </div>
        ) : (
          <>
            <div style={{ marginBottom: 24 }}>
              <SectionCard title={`SKU Demand — ${item} at ${store}`} subtitle="Actual (historical) + Forecast (28d) with 80% prediction interval">
                <EChartWrapper option={forecastOption} height={280} />
              </SectionCard>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
              <SectionCard title="Inventory Trajectory" subtitle="Stock on hand · Reorder point · Safety stock">
                <EChartWrapper option={invOption} height={240} />
              </SectionCard>
              <SectionCard title="Item Profile" subtitle="Classification & inventory parameters">
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {[
                    ['Demand Class', <Badge key="dc" label="Intermittent" variant="warning" />],
                    ['ABC Class', <Badge key="abc" label="B" variant="neutral" />],
                    ['XYZ Class', <Badge key="xyz" label="Z" variant="danger" />],
                    ['Avg Daily Demand', '4.8 units'],
                    ['% Zero Days', '42%'],
                    ['Safety Stock', `${ssVal} units`],
                    ['Reorder Point', `${ropVal} units`],
                    ['Lead Time', '7 days'],
                  ].map(([label, val]) => (
                    <div key={String(label)} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: '#64748B' }}>{label}</span>
                      <span style={{ fontSize: 14, color: '#F1F5F9' }}>{val}</span>
                    </div>
                  ))}
                </div>
              </SectionCard>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
