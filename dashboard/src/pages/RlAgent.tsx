import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain, Zap, Target, BarChart3, Loader2, AlertTriangle,
  TrendingUp, TrendingDown, Shield, PieChart as PieIcon, ArrowUpRight, ArrowDownRight,
} from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell,
  LineChart, Line, PieChart, Pie,
} from 'recharts';
import { api } from '../lib/api';
import type { RLSummaryResponse } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import Badge from '../components/ui/Badge';
import { staggerContainer, fadeSlideUp } from '../lib/animations';

const PAGE_INFO = {
  title: 'RL Agent Monitor — What Does This Page Show?',
  sections: [
    { heading: 'What is this page?', text: 'Monitors two Deep Reinforcement Learning agents (PPO and SAC) that learn to manage a stock portfolio by trial-and-error on real NIFTY 50 price data (2015-2021 train, 2022-2023 validation).' },
    { heading: 'PPO vs SAC', text: 'PPO (Proximal Policy Optimization) is stable, clipped updates prevent catastrophic forgetting. SAC (Soft Actor-Critic) uses entropy bonus for better exploration in continuous action spaces. Both are state-of-the-art.' },
    { heading: 'Training reward curve', text: 'Shows Sharpe Ratio per episode. An episode = 252 trading days (1 year). Upward trend = agent is learning better allocation. Compare PPO vs SAC convergence speed.' },
    { heading: 'Cumulative returns', text: 'Portfolio growth on validation data (2022-2023). Compares PPO, SAC, and equal-weight baseline. Shows real out-of-sample performance — data the agent never trained on.' },
    { heading: 'Sector allocation', text: 'How the agent distributes across sectors. Concentrated = high conviction. Diversified = risk management. Compare PPO vs SAC sector preferences.' },
    { heading: 'Constraints', text: 'Max 20% per stock (diversification), -5% stop loss, -15% circuit breaker, 0.1% transaction cost + 0.05% slippage. These are real Indian market constraints.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'Episodes': {
    what: 'Number of training episodes completed. Each episode = 252 trading days (1 year) of portfolio management on real returns.',
    why: 'More episodes = more experience with different market regimes (bull, bear, sideways). Agent learns to handle all conditions.',
    how: 'Train period 2015-2021 divided into 252-day windows. Agent trades through each, receives Sharpe as reward.',
    good: '> 50 = sufficient convergence | Episodes are limited by available training data.',
  },
  'Avg Reward': {
    what: 'Average Sharpe Ratio over last 10 training episodes. Higher = better risk-adjusted returns.',
    why: 'Reward = Sharpe Ratio. Directly measures the agent\'s objective: maximize return per unit of risk taken.',
    how: 'Mean of last 10 episode Sharpes. Uses 7% risk-free rate (Indian T-bill), 248 trading days/year.',
    good: '< 0.3 = still learning | 0.3-0.7 = decent | 0.7-1.2 = good | > 1.2 = excellent',
  },
  'Sharpe (Val)': {
    what: 'Sharpe Ratio on validation period (2022-2023). Out-of-sample performance the agent never trained on.',
    why: 'Training Sharpe can overfit. Validation Sharpe shows if the agent learned generalizable strategies.',
    how: 'Final agent weights applied to validation returns. Sharpe = (annualized return - 7%) / volatility.',
    good: '> 0.5 = promising | > 0.8 = strong | > 1.0 = excellent generalization',
  },
  'Max Drawdown': {
    what: 'Worst peak-to-trough decline during validation. The maximum pain an investor would experience.',
    why: 'Even profitable strategies can have scary drops. Max DD reveals the worst-case scenario.',
    how: 'Track cumulative portfolio value. Measure largest percentage drop from any peak to subsequent trough.',
    good: '> -5% = excellent risk management | -5% to -15% = acceptable | < -15% = triggers circuit breaker',
  },
};

const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Infra': '#14B8A6', 'Others': '#9CA3AF',
  'Unknown': '#9CA3AF',
};

export default function RlAgent() {
  const [agent, setAgent] = useState<'PPO' | 'SAC'>('PPO');
  const [data, setData] = useState<RLSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'rewards' | 'cumulative' | 'weights'>('rewards');

  useEffect(() => {
    api.rlSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load RL data');
        setLoading(false);
      });
  }, []);

  if (loading) return (
    <div className="flex flex-col items-center justify-center h-96 gap-4">
      <Loader2 size={32} className="animate-spin text-primary" />
      <p className="text-text-secondary text-sm">Loading RL agent data from real stock returns...</p>
    </div>
  );

  if (error || !data) return (
    <div className="flex flex-col items-center justify-center h-96 gap-4">
      <AlertTriangle size={32} className="text-loss" />
      <p className="text-loss text-sm font-medium">{error || 'No data available'}</p>
    </div>
  );

  const isPPO = agent === 'PPO';
  const rewardData = data.reward_curve.map(r => ({ episode: r.episode, ppo: r.ppo_reward, sac: r.sac_reward }));
  const weightData = data.weights.map(w => ({
    name: w.ticker, weight: isPPO ? w.ppo_weight : w.sac_weight, sector: w.sector,
  }));

  // Sector allocation pie data
  const sectorPieData = data.sector_allocation.map(s => ({
    name: s.sector,
    value: isPPO ? s.ppo_weight : s.sac_weight,
    color: SECTOR_COLORS[s.sector] || '#9CA3AF',
  })).filter(d => d.value > 1);

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="RL Agent Monitor"
          subtitle={`PPO + SAC on real NIFTY 50 — ${data.ppo_episodes} episodes — validation 2022-2023`}
          icon={<Brain size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Agent Toggle */}
      <div className="flex items-center gap-3 mb-6">
        {(['PPO', 'SAC'] as const).map(a => (
          <button key={a} onClick={() => setAgent(a)}
            className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
              agent === a
                ? 'bg-primary text-white shadow-md shadow-primary/20'
                : 'bg-bg-card text-text-secondary hover:bg-primary-subtle border border-border'
            }`}>
            {a}
          </button>
        ))}
        <span className="text-xs text-text-muted ml-2">
          {isPPO ? 'Stable, clipped updates' : 'Entropy-based exploration'}
        </span>
      </div>

      {/* Metric Cards */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
        <MetricCard title="Episodes" value={isPPO ? data.ppo_episodes : data.sac_episodes} decimals={0} icon={<Target size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Episodes' ? null : 'Episodes')} active={expandedMetric === 'Episodes'} />
        <MetricCard title="Avg Reward" value={isPPO ? data.ppo_avg_reward : data.sac_avg_reward} decimals={4} icon={<Zap size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Avg Reward' ? null : 'Avg Reward')} active={expandedMetric === 'Avg Reward'} />
        <MetricCard title="Sharpe (Val)" value={isPPO ? data.ppo_sharpe : data.sac_sharpe} decimals={4} icon={<BarChart3 size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Sharpe (Val)' ? null : 'Sharpe (Val)')} active={expandedMetric === 'Sharpe (Val)'} />
        <MetricCard title="Max Drawdown" value={(isPPO ? data.ppo_max_drawdown : data.sac_max_drawdown) * 100} decimals={2} suffix="%"
          onClick={() => setExpandedMetric(m => m === 'Max Drawdown' ? null : 'Max Drawdown')} active={expandedMetric === 'Max Drawdown'} />
      </motion.div>
      <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />

      {/* PPO vs SAC Head-to-Head Comparison */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">PPO vs SAC — Head-to-Head</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 font-medium text-text-secondary">Metric</th>
                <th className="text-right py-2 font-medium" style={{ color: '#C15F3C' }}>PPO</th>
                <th className="text-right py-2 font-medium" style={{ color: '#6366F1' }}>SAC</th>
                <th className="text-right py-2 font-medium text-text-secondary">Winner</th>
              </tr>
            </thead>
            <tbody>
              {[
                { metric: 'Sharpe Ratio', ppo: data.ppo_sharpe, sac: data.sac_sharpe, fmt: (v: number) => v.toFixed(4), higher: true },
                { metric: 'Sortino Ratio', ppo: data.ppo_sortino, sac: data.sac_sortino, fmt: (v: number) => v.toFixed(4), higher: true },
                { metric: 'Annual Return', ppo: data.ppo_annual_return, sac: data.sac_annual_return, fmt: (v: number) => `${v.toFixed(2)}%`, higher: true },
                { metric: 'Annual Volatility', ppo: data.ppo_annual_vol, sac: data.sac_annual_vol, fmt: (v: number) => `${v.toFixed(2)}%`, higher: false },
                { metric: 'Max Drawdown', ppo: data.ppo_max_drawdown * 100, sac: data.sac_max_drawdown * 100, fmt: (v: number) => `${v.toFixed(2)}%`, higher: false },
                { metric: 'Avg Reward (Last 10)', ppo: data.ppo_avg_reward, sac: data.sac_avg_reward, fmt: (v: number) => v.toFixed(4), higher: true },
              ].map(row => {
                const ppoWins = row.higher ? row.ppo >= row.sac : row.ppo <= row.sac;
                return (
                  <motion.tr key={row.metric} variants={fadeSlideUp}
                    className="border-b border-border-light hover:bg-bg-card transition-colors">
                    <td className="py-2.5 font-medium">{row.metric}</td>
                    <td className={`py-2.5 text-right font-mono ${ppoWins ? 'font-bold text-primary' : 'text-text-muted'}`}>
                      {row.fmt(row.ppo)}
                    </td>
                    <td className={`py-2.5 text-right font-mono ${!ppoWins ? 'font-bold text-accent-indigo' : 'text-text-muted'}`}>
                      {row.fmt(row.sac)}
                    </td>
                    <td className="py-2.5 text-right">
                      <Badge variant={ppoWins ? 'profit' : 'info'}>
                        {ppoWins ? 'PPO' : 'SAC'}
                      </Badge>
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Charts Section with Tab Switcher */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display font-bold text-lg text-secondary">Performance Charts</h2>
          <div className="flex bg-bg-card rounded-lg border border-border-light p-0.5">
            {([
              { key: 'rewards', label: 'Training Rewards' },
              { key: 'cumulative', label: 'Cumulative Returns' },
              { key: 'weights', label: 'Portfolio Weights' },
            ] as const).map(tab => (
              <button key={tab.key} onClick={() => setActiveTab(tab.key)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === tab.key
                    ? 'bg-primary text-white shadow-sm'
                    : 'text-text-secondary hover:text-text'
                }`}>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence mode="wait">
          {/* Training Rewards */}
          {activeTab === 'rewards' && (
            <motion.div key="rewards" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
              <p className="text-xs text-text-secondary mb-3">
                Sharpe Ratio per training episode. Upward trend = agent learning better risk-adjusted allocation from real 2015-2021 returns.
              </p>
              <ResponsiveContainer width="100%" height={340} minHeight={1}>
                <AreaChart data={rewardData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                  <defs>
                    <linearGradient id="grad-ppo" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#C15F3C" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#C15F3C" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="grad-sac" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366F1" stopOpacity={0.15} />
                      <stop offset="100%" stopColor="#6366F1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="episode" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
                    label={{ value: 'Episode', position: 'insideBottom', offset: -5, style: { fontSize: 11, fill: '#9CA3AF' } }} />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                    label={{ value: 'Sharpe', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9CA3AF' } }} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 13 }} />
                  <Area type="monotone" dataKey="ppo" name="PPO" stroke="#C15F3C" strokeWidth={2.5} fill="url(#grad-ppo)" dot={false} animationDuration={1500} />
                  <Area type="monotone" dataKey="sac" name="SAC" stroke="#6366F1" strokeWidth={2} fill="url(#grad-sac)" dot={false} animationDuration={1500} />
                </AreaChart>
              </ResponsiveContainer>
            </motion.div>
          )}

          {/* Cumulative Returns */}
          {activeTab === 'cumulative' && (
            <motion.div key="cumulative" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
              <p className="text-xs text-text-secondary mb-3">
                Portfolio growth on validation data (2022-2023). PPO vs SAC vs Equal-Weight baseline. All use real out-of-sample returns.
              </p>
              <ResponsiveContainer width="100%" height={340} minHeight={1}>
                <AreaChart data={data.cumulative_returns} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                  <defs>
                    <linearGradient id="grad-ppo-cum" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#C15F3C" stopOpacity={0.15} />
                      <stop offset="100%" stopColor="#C15F3C" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="grad-sac-cum" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366F1" stopOpacity={0.1} />
                      <stop offset="100%" stopColor="#6366F1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
                    label={{ value: 'Trading Days', position: 'insideBottom', offset: -5, style: { fontSize: 11, fill: '#9CA3AF' } }} />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                    tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                    formatter={(v: number) => `${v.toFixed(2)}%`} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 13 }} />
                  <Area type="monotone" dataKey="ppo" name="PPO" stroke="#C15F3C" strokeWidth={2.5} fill="url(#grad-ppo-cum)" dot={false} animationDuration={1200} />
                  <Area type="monotone" dataKey="sac" name="SAC" stroke="#6366F1" strokeWidth={2} fill="url(#grad-sac-cum)" dot={false} animationDuration={1200} />
                  <Line type="monotone" dataKey="equal_weight" name="Equal Weight" stroke="#9CA3AF" strokeWidth={1.5} strokeDasharray="5 5" dot={false} animationDuration={1200} />
                </AreaChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-6 mt-3">
                {[
                  { label: 'PPO', value: data.cumulative_returns[data.cumulative_returns.length - 1]?.ppo, color: '#C15F3C' },
                  { label: 'SAC', value: data.cumulative_returns[data.cumulative_returns.length - 1]?.sac, color: '#6366F1' },
                  { label: 'Equal Wt', value: data.cumulative_returns[data.cumulative_returns.length - 1]?.equal_weight, color: '#9CA3AF' },
                ].map(s => (
                  <div key={s.label} className="text-center">
                    <span className="text-xs text-text-muted">{s.label}</span>
                    <p className="font-mono font-bold text-lg" style={{ color: s.color }}>
                      {s.value > 0 ? '+' : ''}{s.value?.toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Portfolio Weights */}
          {activeTab === 'weights' && (
            <motion.div key="weights" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
              <p className="text-xs text-text-secondary mb-3">
                Top 15 stock allocations by {agent} agent. Bar color = sector. Max 20% per stock constraint enforced.
              </p>
              <ResponsiveContainer width="100%" height={340} minHeight={1}>
                <BarChart data={weightData} margin={{ top: 10, right: 10, bottom: 40, left: 10 }}>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#6B7280' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} angle={-35} textAnchor="end" />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                    formatter={(v: number) => `${v.toFixed(2)}%`} />
                  <Bar dataKey="weight" name="Weight %" radius={[6, 6, 0, 0]} animationDuration={800}>
                    {weightData.map((d, i) => (
                      <Cell key={i} fill={SECTOR_COLORS[d.sector] || '#9CA3AF'} opacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Sector Allocation Donut */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-1">
            Sector Allocation — {agent}
          </h2>
          <p className="text-xs text-text-secondary mb-3">
            How the agent distributes capital across sectors. Concentrated sectors = high conviction.
          </p>
          <ResponsiveContainer width="100%" height={260} minHeight={1}>
            <PieChart>
              <Pie data={sectorPieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                innerRadius={55} outerRadius={100} paddingAngle={2}
                animationBegin={200} animationDuration={1000}>
                {sectorPieData.map((d, i) => (
                  <Cell key={i} fill={d.color} stroke="#fff" strokeWidth={2} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #E5E7EB', fontSize: 12 }}
                formatter={(v: number) => `${v.toFixed(1)}%`} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap justify-center gap-2 mt-1">
            {sectorPieData.map(d => (
              <span key={d.name} className="flex items-center gap-1.5 text-[10px] text-text-secondary">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: d.color }} />
                {d.name} ({d.value.toFixed(1)}%)
              </span>
            ))}
          </div>
        </Card>

        {/* Stock Return Contributions */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-1">
            Return Contribution — PPO (Top 15)
          </h2>
          <p className="text-xs text-text-secondary mb-3">
            Which stocks drive portfolio returns. Contribution = weight × stock return.
          </p>
          <div className="space-y-1.5 max-h-[340px] overflow-y-auto">
            {data.stock_contributions.map((s, i) => (
              <motion.div key={s.ticker}
                initial={{ opacity: 0, x: -15 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.04 }}
                className="flex items-center gap-3 p-2.5 rounded-xl bg-bg-card hover:bg-primary-subtle/20 transition-colors"
              >
                <span className="w-6 text-center text-[10px] font-bold text-text-muted">{i + 1}</span>
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: SECTOR_COLORS[s.sector] || '#9CA3AF' }} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold text-sm">{s.ticker}</span>
                    <span className="text-[10px] text-text-muted">{s.sector}</span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] mt-0.5">
                    <span className="text-text-muted">Wt: {s.weight.toFixed(1)}%</span>
                    <span className={s.cumulative_return >= 0 ? 'text-profit' : 'text-loss'}>
                      Ret: {s.cumulative_return > 0 ? '+' : ''}{s.cumulative_return.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`inline-flex items-center gap-0.5 font-mono text-sm font-bold ${
                    s.return_contrib >= 0 ? 'text-profit' : 'text-loss'
                  }`}>
                    {s.return_contrib >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                    {s.return_contrib > 0 ? '+' : ''}{s.return_contrib.toFixed(2)}%
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </div>

      {/* Constraints Panel */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-4">Risk Constraints</h2>
        <p className="text-xs text-text-secondary mb-4">
          Real-world constraints applied during training. These enforce diversification, limit losses, and account for Indian market costs.
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          {[
            { label: 'Max Position', value: `${data.constraints.max_position * 100}%`, desc: 'No stock can exceed this weight', icon: <Shield size={16} />, color: '#C15F3C' },
            { label: 'Stop Loss', value: `${data.constraints.stop_loss * 100}%`, desc: 'Per-stock loss limit (force exit)', icon: <TrendingDown size={16} />, color: '#DC2626' },
            { label: 'Circuit Breaker', value: `${data.constraints.max_drawdown * 100}%`, desc: 'Portfolio-wide drawdown limit', icon: <AlertTriangle size={16} />, color: '#F59E0B' },
            { label: 'Transaction Cost', value: `${data.constraints.transaction_cost * 100}%`, desc: 'Brokerage + STT per trade', icon: <BarChart3 size={16} />, color: '#6366F1' },
            { label: 'Slippage', value: `${data.constraints.slippage * 100}%`, desc: 'Market impact cost per trade', icon: <Zap size={16} />, color: '#0D9488' },
          ].map((c, i) => (
            <motion.div key={c.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              className="p-4 rounded-xl bg-bg-card border border-border-light hover:border-primary/20 transition-colors"
            >
              <div className="flex items-center gap-2 mb-2">
                <span style={{ color: c.color }}>{c.icon}</span>
                <span className="text-xs font-medium text-text-secondary">{c.label}</span>
              </div>
              <p className="font-mono font-bold text-xl" style={{ color: c.color }}>{c.value}</p>
              <p className="text-[10px] text-text-muted mt-1">{c.desc}</p>
            </motion.div>
          ))}
        </div>
      </Card>

      <p className="text-center text-xs text-text-muted mt-6 mb-2">
        All data computed from real NIFTY 50 stock returns — Train: 2015-2021, Validation: 2022-2023
      </p>
    </div>
  );
}
