import React, { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import {
  BarChart, LineChart, PieChart, HeatmapChart, ScatterChart,
} from "echarts/charts";
import {
  TitleComponent, TooltipComponent, GridComponent, LegendComponent,
  VisualMapComponent, DataZoomComponent, MarkLineComponent, MarkAreaComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart, LineChart, PieChart, HeatmapChart, ScatterChart,
  TitleComponent, TooltipComponent, GridComponent, LegendComponent,
  VisualMapComponent, DataZoomComponent, MarkLineComponent, MarkAreaComponent,
  CanvasRenderer,
]);

// ─── Shared dark theme defaults ───────────────────────────────────────────────
export const CHART_THEME = {
  backgroundColor: "transparent",
  textStyle: { color: "#9CA3AF", fontFamily: "Inter, system-ui, sans-serif", fontSize: 12 },
  grid: { left: 16, right: 16, top: 16, bottom: 32, containLabel: true },
  tooltip: {
    backgroundColor: "#242736",
    borderColor: "#2D3348",
    borderWidth: 1,
    textStyle: { color: "#E5E7EB", fontSize: 12 },
    confine: true,
  },
  legend: { textStyle: { color: "#9CA3AF" }, itemHeight: 8 },
  axisLabel: { color: "#6B7280", fontSize: 11 },
  axisLine: { lineStyle: { color: "#2D3348" } },
  splitLine: { lineStyle: { color: "#1F2337", type: "dashed" as const } },
};

export const PALETTE = ["#4F8EF7", "#34D399", "#FBBF24", "#EF4444", "#60A5FA", "#A78BFA", "#FB923C"];

interface Props {
  option: echarts.EChartsOption;
  height?: number;
  className?: string;
}

export const BaseEChart: React.FC<Props> = ({ option, height = 260, className = "" }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    chart.setOption(option, true);

    const handleResize = () => chart.resize();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []); // init once

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.setOption(option, true);
    }
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height }} className={className} />;
};
