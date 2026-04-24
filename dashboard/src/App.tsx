import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './components/layout/DashboardLayout';
import Portfolio from './pages/Portfolio';
import RlAgent from './pages/RlAgent';
import StressTesting from './pages/StressTesting';
import Federated from './pages/Federated';
import Sentiment from './pages/Sentiment';
import GraphVisualization from './pages/GraphVisualization';
import WorkflowViz from './pages/WorkflowViz';
import FuturePrediction from './pages/FuturePrediction';
import ToastContainer from './components/ui/Toast';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route index element={<Portfolio />} />
          <Route path="rl" element={<RlAgent />} />
          <Route path="stress" element={<StressTesting />} />
          <Route path="fl" element={<Federated />} />
          <Route path="sentiment" element={<Sentiment />} />
          <Route path="graph" element={<GraphVisualization />} />
          <Route path="workflow" element={<WorkflowViz />} />
          <Route path="future" element={<FuturePrediction />} />
          {/* Catch-all: redirect unknown routes to Portfolio */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      <ToastContainer />
    </BrowserRouter>
  );
}
