import { useState, useEffect } from 'react';
import { Users, Shield, Lock, Activity, AlertTriangle } from 'lucide-react';
import { MetricCardSkeleton, Skeleton } from '../components/ui/Skeleton';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar,
} from 'recharts';
import { api } from '../lib/api';
import type { FLSummaryResponse } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import type { MetricBadge } from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import { staggerContainer, staggerFast, fadeSlideUp } from '../lib/animations';
import { motion } from 'framer-motion';

const CLIENT_COLORS = ['#C15F3C', '#6366F1', '#0D9488', '#F59E0B'];

function getPrivacyBadge(epsilon: number): MetricBadge {
  if (epsilon < 1)  return { label: 'STRONG PRIVACY', variant: 'profit' };
  if (epsilon <= 10) return { label: 'MODERATE', variant: 'warning' };
  return { label: 'WEAK PRIVACY', variant: 'loss' };
}

function getSharpeBadge(sharpe: number): MetricBadge {
  if (sharpe >= 1.0) return { label: 'EXCELLENT', variant: 'profit' };
  if (sharpe >= 0.5) return { label: 'GOOD', variant: 'warning' };
  return { label: 'POOR', variant: 'loss' };
}

const PAGE_INFO = {
  title: 'Federated Learning — What Does This Page Show?',
  sections: [
    { heading: 'What is Federated Learning?', text: 'Multiple institutions (banks) train a shared model WITHOUT sharing their private data. Only model weights are exchanged — raw trading data stays local.' },
    { heading: 'Why is it needed?', text: 'Banks cannot share trading data (RBI regulations, GDPR). But collaborative learning benefits everyone. FL enables learning from distributed data while preserving privacy.' },
    { heading: 'FedProx vs FedAvg', text: 'FedAvg simply averages client weights. FedProx adds a "proximal term" that prevents clients from drifting too far from the global model — better for Non-IID data (each client has different sectors).' },
    { heading: 'Convergence chart', text: 'Shows loss decreasing over 50 FL rounds. FedProx (solid) converges faster than FedAvg (dashed). Individual client lines show per-sector training progress.' },
    { heading: 'Differential Privacy', text: 'DP-SGD adds controlled noise to gradients. Privacy budget ε=8.0, δ=10⁻⁵. Mathematically guarantees that no individual stock\'s data can be reverse-engineered from model weights.' },
    { heading: 'Fairness comparison', text: 'With FL, even small clients (IT: 6 stocks) benefit from global knowledge. Without FL, clients with less data perform worse. FL improves fairness across all participants.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'FL Rounds': {
    what: 'Number of federated learning communication rounds. Each round: clients train locally → send weights → server aggregates → broadcast updated model.',
    why: 'More rounds = better convergence. But each round has communication cost. 50 rounds is a sweet spot for convergence vs efficiency.',
    how: 'Each round: (1) broadcast global model, (2) clients train 5 local epochs, (3) collect weights, (4) FedProx aggregate.',
    good: '20-30 rounds = basic convergence | 50 rounds = good convergence | 100+ = diminishing returns',
  },
  'Privacy ε': {
    what: 'Epsilon (ε) is the privacy budget. Lower = more private. ε=8.0 means there is a mathematical guarantee that individual data points cannot be identified from the model.',
    why: 'Without DP, an adversary could potentially reconstruct training data from model weights. ε quantifies how much information leaks.',
    how: 'DP-SGD clips gradients to max norm 1.0, then adds calibrated Gaussian noise. ε is computed via the moments accountant.',
    good: 'ε < 1 = very strong privacy | ε = 1-10 = moderate (practical for finance) | ε > 10 = weak privacy',
  },
  'Global Sharpe': {
    what: 'Sharpe Ratio of the global (federated) model evaluated on the full validation dataset with all stocks.',
    why: 'Shows the quality of the collaboratively trained model. Should be better than any single client\'s local-only model.',
    how: 'Global model weights → equal-weight portfolio on all 44 stocks → Sharpe on 2022-2023 validation data.',
    good: '> 0.5 = decent global model | > 0.8 = good | > 1.0 = FL is clearly beneficial',
  },
};

export default function Federated() {
  const [data, setData] = useState<FLSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

  useEffect(() => {
    api.flSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load FL data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-56 mb-2" rounded="lg" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <MetricCardSkeleton key={i} />)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Skeleton className="h-64" rounded="xl" />
          <Skeleton className="h-64" rounded="xl" />
        </div>
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

  const clients = data.clients ?? []
  const convData = data.convergence.map(r => ({
    round: r.round,
    FedProx: r.fedprox_loss,
    FedAvg: r.fedavg_loss,
    ...(clients[0] ? { [clients[0].name]: r.client_0_loss } : {}),
    ...(clients[1] ? { [clients[1].name]: r.client_1_loss } : {}),
    ...(clients[2] ? { [clients[2].name]: r.client_2_loss } : {}),
    ...(clients[3] ? { [clients[3].name]: r.client_3_loss } : {}),
  }));

  const fairnessData = data.fairness.map(f => ({
    client: f.client.split(' + ')[0],
    withFL: Math.round(f.with_fl * 1000) / 1000,
    withoutFL: Math.round(f.without_fl * 1000) / 1000,
  }));

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Federated Learning"
          subtitle={`${data.strategy} + DP-SGD (ε=${data.privacy_epsilon}) — ${data.n_clients} sector-wise clients — ${data.n_rounds} rounds`}
          icon={<Users size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard title="FL Rounds" value={data.n_rounds} decimals={0} icon={<Activity size={18} />}
          badge={{ label: 'CONVERGED', variant: 'profit' }}
          onClick={() => setExpandedMetric(m => m === 'FL Rounds' ? null : 'FL Rounds')} active={expandedMetric === 'FL Rounds'} />
        <MetricCard title="Privacy ε" value={data.privacy_epsilon} decimals={1} icon={<Lock size={18} />}
          badge={getPrivacyBadge(data.privacy_epsilon)}
          onClick={() => setExpandedMetric(m => m === 'Privacy ε' ? null : 'Privacy ε')} active={expandedMetric === 'Privacy ε'} />
        <MetricCard title="Global Sharpe" value={data.global_sharpe} decimals={3} icon={<Shield size={18} />}
          badge={getSharpeBadge(data.global_sharpe)}
          onClick={() => setExpandedMetric(m => m === 'Global Sharpe' ? null : 'Global Sharpe')} active={expandedMetric === 'Global Sharpe'} />
        <MetricCard title="Clients" value={data.n_clients} decimals={0}
          badge={{ label: `${data.n_clients} SECTORS`, variant: 'neutral' }} />
      </motion.div>

      <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />

      {/* Client Info Cards */}
      <motion.div variants={staggerFast} initial="hidden" animate="visible"
        transition={{ delayChildren: 0.35 }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {data.clients.map((c, i) => (
          <motion.div key={c.client_id} variants={fadeSlideUp}
            whileHover={{ y: -3, boxShadow: '0 10px 28px rgba(0,0,0,0.08)' }}
            transition={{ type: 'spring', stiffness: 280, damping: 22 }}
            className="bg-white rounded-2xl border border-border p-5 relative overflow-hidden cursor-default"
            style={{ borderLeftColor: CLIENT_COLORS[i], borderLeftWidth: 3 }}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-transparent to-transparent pointer-events-none"
              style={{ background: `linear-gradient(135deg, ${CLIENT_COLORS[i]}08 0%, transparent 60%)` }} />
            <div className="flex items-center gap-2 mb-2 relative">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLIENT_COLORS[i] }} />
              <span className="text-sm font-semibold text-text">{c.name}</span>
            </div>
            <p className="text-xs text-text-muted mb-1 relative">{c.sectors.join(', ')}</p>
            <p className="text-2xl font-mono font-bold text-text relative">
              {c.n_stocks} <span className="text-sm font-medium text-text-secondary">stocks</span>
            </p>
          </motion.div>
        ))}
      </motion.div>

      {/* Convergence Curves */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Convergence — Loss vs FL Rounds (Real Sector Data)
        </h2>
        {convData.length === 0 ? (
          <div className="flex items-center justify-center h-48 text-sm text-text-muted">
            No convergence data — run federated training first
          </div>
        ) : null}
        <ResponsiveContainer width="100%" height={convData.length === 0 ? 0 : 350} minHeight={convData.length === 0 ? 0 : 1}>
          <LineChart data={convData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="round" tick={{ fontSize: 12, fill: '#9CA3AF' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="FedProx" stroke="#C15F3C" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="FedAvg" stroke="#6366F1" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
            {clients.map((c, i) => (
              <Line key={c.name} type="monotone" dataKey={c.name}
                name={`${c.name} (${c.n_stocks} stocks)`}
                stroke={CLIENT_COLORS[i]} strokeWidth={1.5} strokeOpacity={0.75}
                strokeDasharray="3 3" dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Fairness Comparison */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Client Fairness — With FL vs Without FL (Real Sharpe Ratios)
        </h2>
        <ResponsiveContainer width="100%" height={280} minHeight={1}>
          <BarChart data={fairnessData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="client" tick={{ fontSize: 12, fill: '#6B7280' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
              label={{ value: 'Sharpe Ratio', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#9CA3AF' } }} />
            <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="withFL" name="With FL" fill="#C15F3C" radius={[6, 6, 0, 0]} />
            <Bar dataKey="withoutFL" name="Without FL" fill="#D1D5DB" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <p className="text-center text-xs text-text-muted mt-6 mb-2">
        Real sector-split data — {clients.map(c => `${c.name}: ${c.n_stocks}`).join(' | ')} — Privacy: ε={data.privacy_epsilon}, δ={data.privacy_delta}
      </p>
    </div>
  );
}
