import React, { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";

const ExecutiveOverview = lazy(() => import("./pages/ExecutiveOverview").then(m => ({ default: m.ExecutiveOverview })));
const ForecastLab       = lazy(() => import("./pages/ForecastLab").then(m => ({ default: m.ForecastLab })));
const InventoryRisk     = lazy(() => import("./pages/InventoryRisk").then(m => ({ default: m.InventoryRisk })));
const ProductDrilldown  = lazy(() => import("./pages/ProductDrilldown").then(m => ({ default: m.ProductDrilldown })));
const ModelQuality      = lazy(() => import("./pages/ModelQuality").then(m => ({ default: m.ModelQuality })));
const ScenarioLab       = lazy(() => import("./pages/ScenarioLab").then(m => ({ default: m.ScenarioLab })));
const About             = lazy(() => import("./pages/About").then(m => ({ default: m.About })));

const Loading = () => (
  <div className="flex items-center justify-center h-64 text-text-secondary text-sm">
    Loading…
  </div>
);

export const AppRouter: React.FC = () => (
  <Suspense fallback={<Loading />}>
    <Routes>
      <Route path="/"          element={<Navigate to="/overview" replace />} />
      <Route path="/overview"  element={<ExecutiveOverview />} />
      <Route path="/forecast"  element={<ForecastLab />} />
      <Route path="/risk"      element={<InventoryRisk />} />
      <Route path="/drilldown" element={<ProductDrilldown />} />
      <Route path="/quality"   element={<ModelQuality />} />
      <Route path="/scenarios" element={<ScenarioLab />} />
      <Route path="/about"     element={<About />} />
    </Routes>
  </Suspense>
);
