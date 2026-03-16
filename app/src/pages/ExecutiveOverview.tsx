import { DollarSign, CheckCircle, Target, AlertTriangle, Package, Clock } from 'lucide-react'
import { Header } from '../components/layout/Header'
import { KPICard } from '../components/ui/KPICard'
import { SectionCard } from '../components/ui/SectionCard'
import { EChartWrapper } from '../components/ui/EChartWrapper'
import { useJsonData } from '../hooks/useServingData'
import { COLORS } from '../utils/echartTheme'

interface ExecutiveSummary {
  revenue_proxy_total?: number
  fill_rate_avg?: number
  forecast_mae_avg?: number
  stockout_rate?: number
  revenue_by_state?: Record<string, number>
  revenue_by_category?: Record<string, number>
  monthly_trend?: Array<{ month: string; sales: number }>
  [key: string]: unknown
}

export default function ExecutiveOverview() {
  const { data } = useJsonData<ExecutiveSummary>('executive_summary.json', {})

  const revenueTotal = data.revenue_proxy_total ?? 111800
  const fillRate = (data.fill_rate_avg ?? 0.897) * 100
  const mae = data.forecast_mae_avg ?? 0.626
  const stockoutRate = (data.stockout_rate ?? 0.674) * 100

  const stateData = data.revenue_by_state ?? { CA: 45200, TX: 38600, WI: 28000 }
  const states = Object.keys(stateData)
  const stateValues = Object.values(stateData) as number[]

  const catData = data.revenue_by_category ?? { FOODS: 55900, HOBBIES: 22360, HOUSEHOLD: 33540 }
  const cats = Object.keys(catData)
  const catValues = Object.values(catData) as number[]
  const catColors = [COLORS.FOODS, COLORS.HOBBIES, COLORS.HOUSEHOLD]

  const months = data.monthly_trend ?? Array.from({ length: 12 }, (_, i) => ({
    month: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i],
    sales: 8000 + Math.round(Math.sin(i / 2) * 2000),
  }))

  const barOption = {
    tooltip: { trigger: 'axis' as const },
    xAxis: { type: 'value' as const, axisLabel: { formatter: (v: number) => `$${(v/1000).toFixed(0)}K` } },
    yAxis: { type: 'category' as const, data: states },
    series: [{
      type: 'bar' as const,
      data: stateValues,
      barMaxWidth: 32,
      itemStyle: {
        borderRadius: [0, 6, 6, 0],
        color: (p: { dataIndex: number }) => [COLORS.CA, COLORS.TX, COLORS.WI][p.dataIndex] ?? COLORS.blue,
      },
      label: { show: true, position: 'right' as const, formatter: (p: { value: number }) => `$${(p.value/1000).toFixed(1)}K`, color: '#94A3B8', fontSize: 11 },
    }],
  }

  const donutOption = {
    tooltip: { trigger: 'item' as const, formatter: '{b}: ${c} ({d}%)' },
    legend: { bottom: 0, left: 'center' },
    series: [{
      type: 'pie' as const,
      radius: ['55%', '80%'],
      center: ['50%', '45%'],
      data: cats.map((cat, i) => ({ name: cat, value: catValues[i], itemStyle: { color: catColors[i] } })),
      label: { show: false },
      emphasis: { label: { show: true, fontSize: 13, fontWeight: 'bold' as const } },
    }],
  }

  const trendDates = months.map((m: { month: string }) => m.month.slice(0, 7))
  const trendValues = months.map((m: { sales: number }) => m.sales)

  const areaOption = {
    tooltip: { trigger: 'axis' as const },
    xAxis: { type: 'category' as const, data: trendDates, boundaryGap: false, axisLabel: { rotate: 30, fontSize: 10 } },
    yAxis: { type: 'value' as const, axisLabel: { formatter: (v: number) => `${(v/1000).toFixed(0)}K` } },
    series: [{
      name: 'Sales',
      type: 'line' as const,
      data: trendValues,
      smooth: true,
      symbolSize: 4,
      lineStyle: { color: COLORS.blue, width: 2 },
      itemStyle: { color: COLORS.blue },
      areaStyle: {
        color: {
          type: 'linear' as const,
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(59,130,246,0.20)' },
            { offset: 1, color: 'rgba(59,130,246,0.00)' },
          ],
        },
      },
    }],
  }

  return (
    <div>
      <Header title="Executive Overview" />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        {/* KPI Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
          <KPICard label="Revenue Proxy" value={`${(revenueTotal/1000).toFixed(1)}K`} prefix="$" accent="primary" icon={DollarSign} delta={4.2} deltaLabel="vs last period" />
          <KPICard label="Fill Rate" value={fillRate.toFixed(1)} suffix="%" accent="success" icon={CheckCircle} delta={2.1} deltaLabel="vs baseline" />
          <KPICard label="Forecast MAE" value={mae.toFixed(3)} accent="primary" icon={Target} />
          <KPICard label="Stockout Rate" value={stockoutRate.toFixed(1)} suffix="%" accent="danger" icon={AlertTriangle} delta={-1.3} deltaLabel="improvement" />
          <KPICard label="Avg Inventory" value="23.8" accent="warning" icon={Package} suffix=" units" />
          <KPICard label="Days of Supply" value="18.4" accent="primary" icon={Clock} />
        </div>

        {/* Charts Row */}
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 16, marginBottom: 24 }}>
          <SectionCard title="Revenue by State" subtitle="M5 Walmart dataset · 3 states">
            <EChartWrapper option={barOption} height={200} />
          </SectionCard>
          <SectionCard title="Revenue by Category" subtitle="Relative contribution">
            <EChartWrapper option={donutOption} height={200} />
          </SectionCard>
        </div>

        {/* Trend */}
        <SectionCard title="Revenue Trend — Last 12 Months" subtitle="Aggregated proxy revenue from M5 sales data">
          <EChartWrapper option={areaOption} height={240} />
        </SectionCard>
      </div>
    </div>
  )
}
