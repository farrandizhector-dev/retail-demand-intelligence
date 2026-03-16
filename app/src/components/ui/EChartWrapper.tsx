import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'

interface EChartWrapperProps {
  option: EChartsOption
  height?: number | string
  className?: string
}

export function EChartWrapper({ option, height = 300, className = '' }: EChartWrapperProps) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!ref.current) return
    chartRef.current = echarts.init(ref.current, 'dark-premium')
    return () => { chartRef.current?.dispose() }
  }, [])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [option])

  useEffect(() => {
    const observer = new ResizeObserver(() => chartRef.current?.resize())
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])

  return <div ref={ref} style={{ height }} className={`w-full ${className}`} />
}
