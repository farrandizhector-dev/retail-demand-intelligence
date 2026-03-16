import { AlertTriangle, Database, Cpu, Globe, Monitor } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Header } from '../components/layout/Header'
import { SectionCard } from '../components/ui/SectionCard'
import { Badge } from '../components/ui/Badge'

interface Layer {
  icon: LucideIcon
  name: string
  desc: string
  tech: string
  color: string
}

const LAYERS: Layer[] = [
  { icon: Database, name: 'Raw Layer', desc: 'M5 Walmart sales, Open-Meteo weather, FRED macro', tech: 'CSV / JSON', color: '#64748B' },
  { icon: Database, name: 'Bronze Layer', desc: 'Schema-validated Parquet. Pandera contracts enforced.', tech: 'Polars + Pandera', color: '#F59E0B' },
  { icon: Database, name: 'Silver Layer', desc: 'Long format sales (58M rows), daily prices, calendar joins, 18 Hive partitions.', tech: 'Polars', color: '#3B82F6' },
  { icon: Database, name: 'Gold / Features', desc: '85 features, ADI/CV² classification, ABC/XYZ. Leakage-proof via rolling window.', tech: 'Polars + custom leakage guard', color: '#10B981' },
  { icon: Cpu, name: 'ML Engine', desc: 'LightGBM global model, 5-fold backtesting, conformal calibration, MinT reconciliation, Monte Carlo.', tech: 'LightGBM · statsforecast · MLflow', color: '#3B82F6' },
  { icon: Monitor, name: 'Frontend', desc: '7-page React dashboard, pre-calculated JSON serving, HF Static Space deploy.', tech: 'React 18 · TypeScript · ECharts', color: '#10B981' },
]

const TECH = [
  'Python 3.11','Polars','LightGBM','statsforecast','hierarchicalforecast',
  'Pandera','MLflow','React 18','TypeScript','Tailwind CSS','ECharts','Vite',
  'Framer Motion','dbt-core','PostgreSQL',
]

export default function About() {
  return (
    <div>
      <Header title="About" />
      <div style={{ padding: 24, maxWidth: 800, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>

        {/* Architecture */}
        <SectionCard title="Data Architecture" subtitle="6-layer medallion architecture">
          <div>
            {LAYERS.map((layer, i) => {
              const Icon = layer.icon
              return (
                <div key={layer.name} style={{ display: 'flex', gap: 16, position: 'relative' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <div style={{ padding: 8, backgroundColor: '#1E2432', borderRadius: 8, color: layer.color }}>
                      <Icon size={16} color={layer.color} />
                    </div>
                    {i < LAYERS.length - 1 && <div style={{ width: 1, flex: 1, backgroundColor: '#1E293B', margin: '4px 0' }} />}
                  </div>
                  <div style={{ paddingBottom: 16, flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                      <span style={{ fontSize: 14, fontWeight: 600, color: '#F1F5F9' }}>{layer.name}</span>
                      <span style={{ fontSize: 12, color: '#64748B' }}>{layer.tech}</span>
                    </div>
                    <div style={{ fontSize: 12, color: '#94A3B8', marginTop: 2 }}>{layer.desc}</div>
                  </div>
                </div>
              )
            })}
          </div>
        </SectionCard>

        {/* Data Sources */}
        <SectionCard title="Data Sources" subtitle="Real datasets + synthetic inventory layer">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              { name: 'M5 Forecasting', desc: '42,840 hierarchical time series, 58.3M observations, Walmart US 2011–2016', badge: 'REAL', bv: 'success' as const },
              { name: 'Open-Meteo Weather', desc: 'Historical weather for CA, TX, WI — free REST API, 3 states × 1,969 days', badge: 'REAL', bv: 'success' as const },
              { name: 'FRED Macro', desc: 'CPI, unemployment, consumer sentiment — US macroeconomic indicators', badge: 'REAL', bv: 'success' as const },
              { name: 'Inventory Parameters', desc: 'Lead time, safety stock, ROP, holding costs — synthetically generated (seed=42)', badge: 'SYNTHETIC', bv: 'danger' as const },
            ].map((s, idx, arr) => (
              <div key={s.name} style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, paddingBottom: 8, borderBottom: idx < arr.length - 1 ? '1px solid #162032' : 'none' }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: '#F1F5F9' }}>{s.name}</div>
                  <div style={{ fontSize: 12, color: '#94A3B8', marginTop: 2 }}>{s.desc}</div>
                </div>
                <Badge label={s.badge} variant={s.bv} />
              </div>
            ))}
          </div>
        </SectionCard>

        {/* Methodology */}
        <SectionCard title="Methodology" subtitle="Technical approach">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {[
              { title: 'Demand Classification (ADI/CV²)', desc: 'Before any modeling, each SKU is classified as Smooth, Erratic, Intermittent, or Lumpy using Average Demand Interval and Coefficient of Variation squared. This determines which baseline is appropriate and conditions feature engineering.' },
              { title: 'Feature Engineering (85 features)', desc: '7 families: time lags (7/14/28/56/365d), rolling statistics (7/28/90d), calendar effects, price features (level, change, relative), weather, SNAP indicators, and interaction terms. All computed with strict leakage controls.' },
              { title: 'LightGBM Global Model', desc: 'One model trained on all 30,490 series simultaneously using item/store/category embeddings. More robust than per-series models given high intermittency. Quantile variants (p10, p50, p90) for uncertainty.' },
              { title: 'Conformal Calibration', desc: 'Post-hoc split conformal prediction calibrates quantile intervals to guarantee ≥80% empirical coverage without retraining. Coverage verified at 90.0% on holdout.' },
              { title: 'Hierarchical Reconciliation (MinT-Shrink)', desc: '42,840 series across 12 hierarchy levels reconciled using Minimum Trace (MinT) with shrinkage estimator. S-matrix constructed from M5 hierarchy.' },
              { title: 'Monte Carlo Inventory Simulation', desc: '1,000 stochastic paths × 90 days per SKU using NumPy vectorized simulation. Demand sampled from empirical distribution, lead times from log-normal. 4 policies compared: (s,Q), (s,S), (R,S), Newsvendor.' },
            ].map(m => (
              <div key={m.title}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#F1F5F9', marginBottom: 4 }}>{m.title}</div>
                <div style={{ fontSize: 14, color: '#94A3B8' }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </SectionCard>

        {/* Synthetic disclaimer */}
        <div style={{ borderLeft: '4px solid #F59E0B', backgroundColor: 'rgba(245,158,11,0.08)', borderRadius: '0 8px 8px 0', padding: 20 }}>
          <div style={{ display: 'flex', gap: 12 }}>
            <AlertTriangle size={20} color="#F59E0B" style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#F59E0B', marginBottom: 8 }}>Synthetic Data Disclaimer</div>
              <div style={{ fontSize: 14, color: '#94A3B8', display: 'flex', flexDirection: 'column', gap: 8 }}>
                <p style={{ margin: 0 }}>The <strong style={{ color: '#F1F5F9' }}>demand data (M5/Walmart)</strong> is <Badge label="REAL" variant="success" /> — 58.3M actual retail sales observations.</p>
                <p style={{ margin: 0 }}>The <strong style={{ color: '#F1F5F9' }}>inventory layer</strong> (stock levels, order quantities, lead times, stockout events) is <Badge label="SYNTHETIC" variant="danger" /> — generated from calibrated distributions (seed=42) because real inventory records are proprietary.</p>
                <p style={{ margin: 0 }}>This is standard practice in portfolio projects. All synthetic parameters are labeled in code and outputs.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Tech stack */}
        <SectionCard title="Tech Stack" subtitle="Full-stack ML system">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {TECH.map(t => (
              <span key={t} style={{ display: 'inline-flex', alignItems: 'center', padding: '6px 12px', backgroundColor: '#1E2432', borderRadius: 6, fontSize: 12, fontWeight: 500, color: '#94A3B8', border: '1px solid #1E293B' }}>
                {t}
              </span>
            ))}
          </div>
        </SectionCard>

        {/* Author */}
        <SectionCard title="Author" subtitle="Portfolio project">
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ width: 48, height: 48, backgroundColor: 'rgba(59,130,246,0.08)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Globe size={24} color="#3B82F6" />
            </div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#F1F5F9' }}>Héctor Ferrándiz Sanchis</div>
              <div style={{ fontSize: 12, color: '#64748B', marginTop: 2 }}>Principal/Staff-grade · Data Engineering + ML + Operations Research</div>
              <div style={{ fontSize: 12, color: '#3B82F6', marginTop: 4, cursor: 'pointer' }}>github.com/hferrandiz · LinkedIn</div>
            </div>
          </div>
        </SectionCard>

      </div>
    </div>
  )
}
