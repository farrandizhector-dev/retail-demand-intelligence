export const darkTheme = {
  backgroundColor: 'transparent',
  textStyle: { color: '#94A3B8', fontFamily: 'Inter, system-ui, sans-serif' },
  title: { textStyle: { color: '#F1F5F9', fontSize: 14, fontWeight: '600' } },
  legend: {
    textStyle: { color: '#94A3B8', fontSize: 12 },
    icon: 'roundRect',
    itemWidth: 12,
    itemHeight: 8,
    itemGap: 16,
  },
  categoryAxis: {
    axisLine: { lineStyle: { color: '#1E293B' } },
    axisTick: { show: false },
    axisLabel: { color: '#64748B', fontSize: 11 },
    splitLine: { lineStyle: { color: '#1E293B', type: 'dashed' as const } },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: '#64748B', fontSize: 11 },
    splitLine: { lineStyle: { color: '#1E293B', type: 'dashed' as const } },
  },
  tooltip: {
    backgroundColor: '#1C2231',
    borderColor: '#1E293B',
    borderWidth: 1,
    textStyle: { color: '#F1F5F9', fontSize: 12 },
    extraCssText: 'border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.4);padding:10px 14px;',
  },
  grid: { left: 48, right: 24, top: 40, bottom: 32, containLabel: true },
}

export const COLORS = {
  blue: '#3B82F6',
  green: '#10B981',
  yellow: '#F59E0B',
  red: '#EF4444',
  purple: '#8B5CF6',
  slate: '#94A3B8',
  CA: '#3B82F6',
  TX: '#10B981',
  WI: '#F59E0B',
  FOODS: '#3B82F6',
  HOBBIES: '#10B981',
  HOUSEHOLD: '#F59E0B',
}
