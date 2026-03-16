// ─── Serving JSON types ──────────────────────────────────────────────────────

export interface MonthlyPoint { month: string; sales: number }

export interface ExecutiveSummary {
  data_label: string;
  revenue_proxy_total: number;
  revenue_by_state: Record<string, number>;
  revenue_by_category: Record<string, number>;
  fill_rate_avg: number;
  stockout_rate: number;
  forecast_mae_avg: number;
  inventory_value_total: number;
  days_of_supply_avg: number;
  monthly_trend: MonthlyPoint[];
  n_skus: number;
  n_stores: number;
}

export interface HistoryPoint { date: string; actual: number }
export interface ForecastPoint { date: string; forecast_p10: number; forecast_p50: number; forecast_p90: number }

export interface ForecastSeries {
  state_id: string;
  cat_id: string;
  history: HistoryPoint[];
  forecast: ForecastPoint[];
}

export interface DeptRisk {
  fill_rate: number;
  avg_stockout_days: number;
  days_of_supply: number;
  n_items: number;
  n_items_at_risk: number;
  overstock_flag: boolean;
  stockout_probability: number;
}

export interface StoreRisk {
  store_id: string;
  departments: Record<string, DeptRisk>;
}

export interface InventoryRiskMatrix {
  stores: StoreRisk[];
  synthetic_tag: string;
  note?: string;
}

export interface ModelMetrics {
  synthetic_tag: string;
  aggregate?: Record<string, number>;
  segments?: Record<string, unknown>[];
}

export interface ShapFeature {
  rank: number;
  feature: string;
  mean_abs_shap: number;
}

export interface ShapSummary {
  top_features: ShapFeature[];
  n_samples: number;
  n_features: number;
  all_features?: Record<string, number>;
}

export interface ScenarioData {
  description: string;
  fill_rate_mean: number;
  stockout_probability: number;
  expected_stockout_days: number;
  avg_inventory_mean: number;
  total_cost_mean: number;
  delta_fill_rate: number;
  delta_stockout_prob: number;
  delta_stockout_days: number;
  delta_total_cost: number;
}

export interface ScenarioSeries {
  series_id: string;
  baseline: {
    fill_rate_mean: number;
    stockout_probability: number;
    expected_stockout_days: number;
    avg_inventory_mean: number;
    total_cost_mean: number;
  };
  scenarios: Record<string, ScenarioData>;
  worst_scenario: string;
  worst_cost_delta: number;
}

export interface ScenarioResults {
  n_series: number;
  scenario_names: string[];
  scenario_descriptions: Record<string, string>;
  series: ScenarioSeries[];
  synthetic_tag: string;
}

export interface PolicyData {
  fill_rate: number;
  stockout_prob: number;
  avg_inventory: number;
  total_cost: number;
}

export interface PolicyComparisonSeries {
  series_id: string;
  best_policy: string;
  best_cost: number;
  policies: Record<string, PolicyData>;
}

export interface PolicyComparison {
  n_series: number;
  policies: string[];
  best_policy_distribution: Record<string, number>;
  series: PolicyComparisonSeries[];
  synthetic_tag: string;
}

// ─── Filter store ─────────────────────────────────────────────────────────────

export interface FilterState {
  selectedState: string;
  selectedCategory: string;
  selectedDepartment: string;
  setSelectedState: (s: string) => void;
  setSelectedCategory: (c: string) => void;
  setSelectedDepartment: (d: string) => void;
}
