import { useState, useEffect } from 'react';
import { FlaskConical, Layers, Cpu, Zap, Loader2, AlertTriangle } from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { NASLabResponse } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import { staggerContainer } from '../lib/animations';
import { motion } from 'framer-motion';

const PAGE_INFO = {
  title: 'NAS Lab — What Does This Page Show?',
  sections: [
    { heading: 'What is NAS?', text: 'Neural Architecture Search (DARTS) automatically finds the best neural network design for financial time series, instead of manually designing it through trial and error.' },
    { heading: 'How DARTS works', text: 'A "supernet" with all possible operations (Linear, Conv1D, Attention, Skip, Zero) is trained. Each operation has a learnable weight (alpha). Best operations get higher alpha over time.' },
    { heading: 'Alpha Convergence chart', text: 'Shows how operation weights evolve during search. Attention dominates because financial data has strong cross-asset dependencies. Linear is second (basic transformations always useful).' },
    { heading: 'NAS vs Hand-Designed', text: 'Compares the NAS-found architecture against a manually designed baseline on the same validation data. Metrics: Sharpe, Sortino, Return, Max Drawdown.' },
    { heading: 'Why Attention wins', text: 'Financial time series have complex inter-stock relationships (when TCS moves, Infosys follows). Attention mechanisms capture these long-range dependencies better than local operations.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'Search Epochs': {
    what: 'Number of epochs the DARTS search ran. Each epoch updates both the supernet weights and the architecture parameters (alphas).',
    why: 'More epochs allow better convergence of architecture selection. Too few = premature selection, too many = diminishing returns.',
    how: 'Bilevel optimization: outer loop updates alphas on validation loss, inner loop trains weights on training loss.',
    good: '30-50 epochs is standard for DARTS. Our 50 epochs ensure full convergence of alpha parameters.',
  },
  'NAS Sharpe': {
    what: 'Sharpe Ratio achieved by the NAS-optimized architecture on validation data (2022-2023). This proves the found architecture generalizes.',
    why: 'If NAS Sharpe > baseline Sharpe, the architecture search was successful — AI found a better design than human experts.',
    how: 'NAS-found model\'s portfolio returns on validation period. Sharpe = (Return - 7%) / Volatility.',
    good: '> baseline = success | Improvement > 5% = meaningful | > 10% = strong result',
  },
  'Improvement': {
    what: 'Percentage improvement in Sharpe Ratio of NAS-found architecture over the hand-designed baseline.',
    why: 'Quantifies the value of automatic architecture search. Shows that AI can design better models than manual engineering.',
    how: '((NAS_Sharpe - Baseline_Sharpe) / |Baseline_Sharpe|) × 100%.',
    good: '> 5% = meaningful improvement | > 10% = significant | > 20% = exceptional',
  },
};

export default function NasLab() {
  const [data, setData] = useState<NASLabResponse | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.nasSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load NAS data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <Loader2 size={32} className="animate-spin text-primary" />
        <p className="text-text-secondary text-sm">Loading DARTS architecture search results...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <AlertTriangle size={32} className="text-loss" />
        <p className="text-loss text-sm font-medium">{error || 'No data available'}</p>
        <p className="text-text-muted text-xs">Make sure the backend is running: uvicorn src.api.main:app --port 8000</p>
      </div>
    );
  }

  const alphaData = data.alpha_convergence.map(a => ({
    epoch: a.epoch,
    Linear: a.linear,
    Conv1D: a.conv1d,
    Attention: a.attention,
    Skip: a.skip,
    Zero: a.zero,
  }));

  const compareData = data.comparison.map(c => ({
    metric: c.metric,
    'NAS-Found': c.nas_value,
    'Hand-Designed': c.handcraft_value,
  }));

  const OPS = ['Linear', 'Conv1D', 'Attention', 'Skip', 'Zero'];
  const opColors = ['#C15F3C', '#6366F1', '#0D9488', '#F59E0B', '#D1D5DB'];

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="NAS Lab"
          subtitle={`DARTS architecture search — ${data.search_epochs} epochs — best op: ${data.best_op} — ${data.improvement_pct}% improvement`}
          icon={<FlaskConical size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
        <MetricCard title="Search Epochs" value={data.search_epochs} decimals={0} icon={<Layers size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Search Epochs' ? null : 'Search Epochs')} active={expandedMetric === 'Search Epochs'} />
        <MetricCard title="Best Op" value={0} decimals={0} prefix={data.best_op} icon={<Cpu size={18} />} />
        <MetricCard title="NAS Sharpe" value={data.nas_sharpe} decimals={4} icon={<Zap size={18} />}
          onClick={() => setExpandedMetric(m => m === 'NAS Sharpe' ? null : 'NAS Sharpe')} active={expandedMetric === 'NAS Sharpe'} />
        <MetricCard title="Improvement" value={data.improvement_pct} decimals={1} suffix="%"
          onClick={() => setExpandedMetric(m => m === 'Improvement' ? null : 'Improvement')} active={expandedMetric === 'Improvement'} />
      </motion.div>

      <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />

      {/* Architecture Diagram */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">Best Architecture Found</h2>
        <div className="flex items-center gap-3 overflow-x-auto py-4">
          <div className="shrink-0 px-4 py-3 bg-bg-card rounded-xl border border-border text-sm font-mono text-text-secondary">
            Input (21 feat)
          </div>
          {data.best_architecture.map((op, i) => (
            <div key={i} className="flex items-center gap-3">
              <svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#D1D5DB" strokeWidth="2" /></svg>
              <div className={`shrink-0 px-4 py-3 rounded-xl border-2 text-sm font-semibold ${
                op === 'Linear' ? 'border-primary bg-primary-subtle text-primary' :
                op === 'Attention' ? 'border-accent-indigo bg-indigo-50 text-accent-indigo' :
                op === 'Skip' ? 'border-warning bg-amber-50 text-warning' :
                'border-border bg-bg-card text-text-secondary'
              }`}>
                Layer {i + 1}: {op}
              </div>
            </div>
          ))}
          <svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#D1D5DB" strokeWidth="2" /></svg>
          <div className="shrink-0 px-4 py-3 bg-bg-card rounded-xl border border-border text-sm font-mono text-text-secondary">
            GRU → Output (64)
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alpha Convergence */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Alpha Convergence</h2>
          <ResponsiveContainer width="100%" height={300} minHeight={1}>
            <LineChart data={alphaData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="epoch" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              {OPS.map((op, i) => (
                <Line key={op} type="monotone" dataKey={op} stroke={opColors[i]}
                  strokeWidth={op === 'Attention' || op === 'Linear' ? 2.5 : 1.5}
                  dot={false} animationDuration={1200} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Performance Comparison */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">NAS vs Hand-Designed</h2>
          <ResponsiveContainer width="100%" height={300} minHeight={1}>
            <BarChart data={compareData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="metric" tick={{ fontSize: 12, fill: '#6B7280' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="NAS-Found" name="NAS-Found" radius={[6, 6, 0, 0]} fill="#C15F3C" opacity={0.85} />
              <Bar dataKey="Hand-Designed" name="Hand-Designed" radius={[6, 6, 0, 0]} fill="#D1D5DB" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <p className="text-center text-xs text-text-muted mt-6 mb-2">
        Alpha convergence driven by real stock autocorrelation and cross-correlation statistics — Comparison on validation period (2022-2023)
      </p>
    </div>
  );
}
