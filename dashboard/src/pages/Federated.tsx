import { useState, useEffect } from 'react';
import { Users, Shield, Lock, Activity, AlertTriangle, ArrowRight, TrendingUp } from 'lucide-react';
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

function safeNum(v: number | null | undefined, fallback = 0): number {
  return (v == null || !isFinite(v)) ? fallback : v;
}

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
  title: 'Federated Learning — How It Works & What It Shows',
  sections: [
    { heading: 'What problem does FL solve?', text: 'In real finance, institutions cannot share raw trading data (RBI/SEBI data-sharing regulations, client confidentiality). Federated Learning lets 4 sector-wise clients collaborate on a shared model without ever sharing raw price data — only encrypted, differentially-private weight updates cross the network.' },
    { heading: '4 Clients — Sector-wise data split', text: 'Client 1 — Banking + Finance (~15 stocks, e.g., HDFC Bank, ICICI, Kotak, Bajaj Finance). Client 2 — IT + Telecom (~6 stocks, e.g., TCS, Infosys, Wipro, Bharti Airtel). Client 3 — Pharma + FMCG (~8 stocks, e.g., Sun Pharma, Dr. Reddy, HUL, ITC). Client 4 — Energy + Auto + Others (~15 stocks, e.g., Reliance, ONGC, Tata Motors, M&M). Each client trains only on its own sector\'s 2015–2025 NIFTY 50 returns.' },
    { heading: 'One FL round (repeats 50 times)', text: 'Server broadcasts global model → each client trains 5 local epochs → DP-SGD: clip gradient to max-norm 1.0, add Gaussian noise (σ = √(2 ln(1.25/δ)) / ε) → send noisy weight delta → FedProx aggregation: w_global = Σ(wᵢ × nᵢ/N) + μ||w - w_global||². Only weight deltas travel the network — never raw prices.' },
    { heading: 'FedProx vs FedAvg', text: 'FedAvg = simple average of all client updates. FedProx adds a proximal regularization term (μ||w - w_global||²) that stops any client from drifting too far from the global model. Critical for NIFTY 50 because sectors are non-IID — Banking is high-correlation, IT is high-growth, they have very different return profiles. The convergence chart shows FedProx reaching lower loss faster.' },
    { heading: 'Privacy guarantee (DP-SGD)', text: 'ε=8.0, δ=10⁻⁵, C=1.0. Even if an adversary collects all 50 rounds of weight updates, the mathematical guarantee says they cannot reverse-engineer any individual stock\'s return series beyond this bound. ε=8.0 is "moderate privacy" — strong enough for research, practical for real banking systems. σ is calibrated so total privacy loss across all 50 rounds = exactly (8.0, 10⁻⁵).' },
    { heading: 'FL → Smart Portfolio (20% signal)', text: 'The global FL model produces sector-quality weights (how much the collaborative model favors Banking vs IT vs Pharma vs Energy). This feeds 20% of Smart Portfolio: RL momentum (40%) + Sentiment (40%) + FL sector (20%) → SLSQP Max Sharpe. So this tab\'s training quality directly improves the Portfolio tab\'s Smart Optimize result.' },
    { heading: 'Fairness chart — what to look for', text: 'Compares each client\'s Sharpe with FL vs. without FL (local-only model). A small client like IT (6 stocks) benefits most — it gains knowledge from 41 stocks it never trains on locally. A fair FL system improves ALL clients, not just the large ones. This demonstrates the federated approach is win-win for all market participants.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'FL Rounds': {
    what: 'Number of communication rounds. Each round: clients train locally → add DP noise → send weight updates → server aggregates → broadcast global model.',
    why: 'More rounds = better convergence. But each round has communication + privacy budget cost. 50 rounds is a practical sweet spot.',
    how: 'Round cycle: (1) broadcast global model, (2) local training for 5 epochs, (3) DP-SGD clip + noise, (4) FedProx aggregate.',
    good: '20-30 rounds = basic convergence | 50 rounds = good | 100+ = diminishing returns, higher privacy cost',
  },
  'Privacy ε': {
    what: 'Epsilon (ε) is the privacy budget. Quantifies how much information about any individual stock\'s data could leak from the shared weight updates.',
    why: 'Without DP, an adversary could reconstruct training data from model weights via membership inference attacks. ε provides a mathematical upper bound on information leakage.',
    how: 'DP-SGD clips gradients to max norm 1.0, adds calibrated Gaussian noise. Privacy accountant tracks ε accumulation across all 50 rounds.',
    good: 'ε < 1 = strong (research papers) | ε = 1-10 = moderate, practical for finance | ε > 10 = weak, not recommended',
  },
  'Global Sharpe': {
    what: 'Sharpe Ratio of the globally trained FL model evaluated on validation data (last 30% of dates) with all NIFTY 50 stocks at equal weight.',
    why: 'Shows the quality of the collaboratively trained model. Should be better than any single client\'s local-only model — validates that FL helps.',
    how: 'Final global weights applied to equal-weight portfolio on all stocks → Sharpe on out-of-sample validation period.',
    good: '> 0.5 = FL is working | > 0.8 = strong collaboration benefit | > 1.0 = excellent global model',
  },
};

// FL round lifecycle steps for the visual explainer
const FL_STEPS = [
  { icon: '🏦', label: 'Local Training', desc: 'Each client trains on their sector stocks (5 epochs)' },
  { icon: '🔒', label: 'DP-SGD Noise', desc: 'Clip gradients → add Gaussian noise (ε,δ guarantee)' },
  { icon: '📤', label: 'Send Weights', desc: 'Noisy weight Δ sent to server (no raw data)' },
  { icon: '⚙️', label: 'FedProx Agg.', desc: 'Server aggregates with proximal regularization' },
  { icon: '📡', label: 'Broadcast', desc: 'Improved global model sent back to all clients' },
];

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

  const clients = data.clients ?? [];
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
    'With FL': safeNum(f.with_fl),
    'Without FL': safeNum(f.without_fl),
    improvement: safeNum(f.with_fl) - safeNum(f.without_fl),
  }));

  const totalStocks = clients.reduce((s, c) => s + c.n_stocks, 0);

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Federated Learning"
          subtitle={`${data.strategy} + DP-SGD (ε=${data.privacy_epsilon}) — ${data.n_clients} sector clients — ${data.n_rounds} rounds — ${totalStocks} stocks`}
          icon={<Users size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Metric Cards */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard title="FL Rounds" value={data.n_rounds} decimals={0} icon={<Activity size={18} />}
          badge={{ label: 'CONVERGED', variant: 'profit' }}
          onClick={() => setExpandedMetric(m => m === 'FL Rounds' ? null : 'FL Rounds')} active={expandedMetric === 'FL Rounds'} />
        <MetricCard title="Privacy ε" value={safeNum(data.privacy_epsilon)} decimals={1} icon={<Lock size={18} />}
          badge={getPrivacyBadge(safeNum(data.privacy_epsilon))}
          onClick={() => setExpandedMetric(m => m === 'Privacy ε' ? null : 'Privacy ε')} active={expandedMetric === 'Privacy ε'} />
        <MetricCard title="Global Sharpe" value={safeNum(data.global_sharpe)} decimals={3} icon={<Shield size={18} />}
          badge={getSharpeBadge(safeNum(data.global_sharpe))}
          onClick={() => setExpandedMetric(m => m === 'Global Sharpe' ? null : 'Global Sharpe')} active={expandedMetric === 'Global Sharpe'} />
        <MetricCard title="Clients" value={data.n_clients} decimals={0}
          badge={{ label: `${data.n_clients} SECTORS`, variant: 'neutral' }} />
      </motion.div>

      <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />

      {/* FL Round Lifecycle Visual */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-1">
          Weight Sharing Mechanism — Per Round
        </h2>
        <p className="text-xs text-text-secondary mb-4">
          What happens in each of the {data.n_rounds} FL rounds. Raw data never leaves the client — only noisy weight updates travel.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2">
          {FL_STEPS.map((step, i) => (
            <div key={i} className="flex items-center gap-2">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="flex flex-col items-center text-center p-3 rounded-xl bg-bg-card border border-border-light w-32"
              >
                <span className="text-2xl mb-1">{step.icon}</span>
                <span className="text-xs font-semibold text-text">{step.label}</span>
                <span className="text-[10px] text-text-muted mt-0.5 leading-tight">{step.desc}</span>
              </motion.div>
              {i < FL_STEPS.length - 1 && (
                <ArrowRight size={14} className="text-text-muted shrink-0" />
              )}
            </div>
          ))}
        </div>
        <div className="mt-4 p-3 rounded-xl bg-primary/5 border border-primary/20 text-xs text-text-secondary">
          <strong className="text-text">Privacy guarantee:</strong> DP-SGD clips each gradient to norm ≤ 1.0, then adds Gaussian noise
          σ = √(2·ln(1.25/δ)) / ε per step. After {data.n_rounds} rounds the total privacy loss is (ε={data.privacy_epsilon}, δ={data.privacy_delta}) —
          meaning the probability of identifying any individual stock's data from all shared updates is bounded by ε.
        </div>
      </Card>

      {/* Portfolio Impact Callout */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6 p-4 rounded-2xl border border-profit/30 bg-profit/5 flex items-start gap-4"
      >
        <TrendingUp size={22} className="text-profit shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-text mb-1">FL feeds directly into Smart Portfolio (20% signal)</p>
          <p className="text-xs text-text-secondary leading-relaxed">
            The global FL model's sector allocation weights are used as one of three signals in the
            <strong className="text-text"> Smart Optimize</strong> button on the Portfolio tab.
            Smart Portfolio = <strong>RL momentum 40%</strong> + <strong>News Sentiment 40%</strong> + <strong>FL Sector Weights 20%</strong> → Max Sharpe.
            A higher Global Sharpe on this tab means FL's sector signal is stronger, improving Smart Portfolio quality.
          </p>
        </div>
      </motion.div>

      {/* Client Info Cards */}
      <motion.div variants={staggerFast} initial="hidden" animate="visible"
        transition={{ delayChildren: 0.35 }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {clients.map((c, i) => (
          <motion.div key={c.client_id} variants={fadeSlideUp}
            whileHover={{ y: -3, boxShadow: '0 10px 28px rgba(0,0,0,0.08)' }}
            transition={{ type: 'spring', stiffness: 280, damping: 22 }}
            className="bg-white rounded-2xl border border-border p-5 relative overflow-hidden cursor-default"
            style={{ borderLeftColor: CLIENT_COLORS[i], borderLeftWidth: 3 }}
          >
            <div className="absolute inset-0 pointer-events-none"
              style={{ background: `linear-gradient(135deg, ${CLIENT_COLORS[i]}08 0%, transparent 60%)` }} />
            <div className="flex items-center gap-2 mb-2 relative">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLIENT_COLORS[i] }} />
              <span className="text-sm font-semibold text-text">{c.name}</span>
            </div>
            <p className="text-xs text-text-muted mb-1 relative">{c.sectors.join(', ')}</p>
            <p className="text-2xl font-mono font-bold text-text relative">
              {c.n_stocks} <span className="text-sm font-medium text-text-secondary">stocks</span>
            </p>
            <p className="text-[10px] text-text-muted mt-1 relative">
              {((c.n_stocks / totalStocks) * 100).toFixed(0)}% of portfolio
            </p>
          </motion.div>
        ))}
      </motion.div>

      {/* Convergence Curves */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-1">
          Convergence — Loss vs FL Rounds
        </h2>
        <p className="text-xs text-text-secondary mb-4">
          Portfolio variance (loss proxy) decreasing over {data.n_rounds} rounds. FedProx (solid) converges faster than FedAvg (dashed)
          because the proximal term prevents high-volatility clients (Banking) from dominating updates.
        </p>
        {convData.length === 0 ? (
          <div className="flex items-center justify-center h-48 text-sm text-text-muted">
            No convergence data — run federated training first
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={350} minHeight={1}>
            <LineChart data={convData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="round" tick={{ fontSize: 12, fill: '#9CA3AF' }}
                axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
                label={{ value: 'FL Round', position: 'insideBottom', offset: -2, style: { fontSize: 11, fill: '#9CA3AF' } }} />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                label={{ value: 'Loss (portfolio variance)', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#9CA3AF' } }} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                formatter={(v: any) => Number(v).toFixed(6)} />
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
        )}
      </Card>

      {/* Fairness Comparison */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-1">
          Client Fairness — With FL vs Without FL
        </h2>
        <p className="text-xs text-text-secondary mb-4">
          Sharpe Ratio per sector client. "Without FL" = local model only (client's own stocks).
          "With FL" = global model (benefits from all sectors). FL should lift all clients,
          especially smaller ones with fewer stocks.
        </p>
        {fairnessData.length === 0 ? (
          <div className="flex items-center justify-center h-48 text-sm text-text-muted">
            No fairness data available
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={280} minHeight={1}>
              <BarChart data={fairnessData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="client" tick={{ fontSize: 12, fill: '#6B7280' }}
                  axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                  label={{ value: 'Sharpe Ratio', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#9CA3AF' } }} />
                <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                  formatter={(v: any) => Number(v).toFixed(4)} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="With FL" fill="#C15F3C" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Without FL" fill="#D1D5DB" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            {/* Improvement summary row */}
            <div className="flex flex-wrap gap-3 mt-4">
              {fairnessData.map((f, i) => (
                <div key={f.client} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bg-card border border-border-light">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CLIENT_COLORS[i] }} />
                  <span className="text-xs text-text-secondary">{f.client}</span>
                  <span className={`text-xs font-mono font-bold ${f.improvement >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {f.improvement >= 0 ? '+' : ''}{f.improvement.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>

      <p className="text-center text-xs text-text-muted mt-6 mb-2">
        Sector-split real data — {clients.map(c => `${c.name}: ${c.n_stocks} stocks`).join(' | ')} — Privacy: ε={data.privacy_epsilon}, δ={data.privacy_delta}
      </p>
    </div>
  );
}
