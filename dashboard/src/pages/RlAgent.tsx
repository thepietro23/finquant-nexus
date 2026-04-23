import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain, Zap, Target, BarChart3, AlertTriangle,
  TrendingDown, Shield, ArrowUpRight, ArrowDownRight,
} from 'lucide-react';
import { MetricCardSkeleton, Skeleton } from '../components/ui/Skeleton';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell,
  Line, PieChart, Pie,
} from 'recharts';
import { api, ALGO_PREFIX } from '../lib/api';
import type { RLSummaryResponse, AgentType } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import Badge from '../components/ui/Badge';
import { staggerContainer } from '../lib/animations';
import { toast } from '../lib/toast';
import type { MetricBadge } from '../components/ui/MetricCard';

function getRLBadge(metric: 'sharpe' | 'reward' | 'drawdown', value: number): MetricBadge {
  if (metric === 'sharpe') {
    if (value >= 1.0) return { label: 'EXCELLENT', variant: 'profit' };
    if (value >= 0.5) return { label: 'GOOD', variant: 'profit' };
    if (value >= 0)   return { label: 'AVERAGE', variant: 'warning' };
    return { label: 'POOR', variant: 'loss' };
  }
  if (metric === 'reward') {
    if (value >= 1.2) return { label: 'EXCELLENT', variant: 'profit' };
    if (value >= 0.7) return { label: 'GOOD', variant: 'profit' };
    if (value >= 0.3) return { label: 'LEARNING', variant: 'warning' };
    return { label: 'EARLY', variant: 'neutral' };
  }
  // drawdown
  const abs = Math.abs(value * 100);
  if (abs <= 5)  return { label: 'EXCELLENT', variant: 'profit' };
  if (abs <= 15) return { label: 'ACCEPTABLE', variant: 'warning' };
  return { label: 'HIGH RISK', variant: 'loss' };
}

const PAGE_INFO = {
  title: 'RL Agent Monitor — What Does This Page Show?',
  sections: [
    { heading: 'What is this page?', text: 'Compares 5 Deep RL algorithms (PPO, SAC, TD3, A2C, DDPG) + an Ensemble that learns to manage a NIFTY 50 portfolio by trial-and-error on real price data. Each algorithm uses a distinct investment strategy.' },
    { heading: '5 Algorithms', text: 'PPO: momentum blend with diversification. SAC: soft momentum, forced spread (max 8% per stock). TD3: short-term mean-reversion (bets against recent winners). A2C: inverse-volatility (more weight to stable stocks). DDPG: concentrated top-K momentum only.' },
    { heading: 'Ensemble strategy', text: 'Ensemble averages all 5 weight vectors, weighted by each algorithm\'s recent Sharpe performance. Algorithms that performed better recently get higher influence — reducing single-model bias.' },
    { heading: 'Training reward curve', text: 'Shows Sharpe Ratio per episode. An episode = 252 trading days (1 year) of real returns. Upward trend = agent learning better allocation. Each algorithm shows a different convergence pattern.' },
    { heading: 'Cumulative returns', text: 'Portfolio growth on out-of-sample validation data. Compares all 6 strategies plus equal-weight baseline. Shows real out-of-sample performance — data the agents never trained on.' },
    { heading: 'Constraints', text: 'Max 20% per stock (diversification), -5% stop loss, -15% circuit breaker, 0.1% transaction cost + 0.05% slippage. These are real Indian market constraints applied to all algorithms.' },
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


const ALGO_COLORS: Record<AgentType, string> = {
  PPO: '#C15F3C', SAC: '#6366F1', TD3: '#0D9488',
  A2C: '#F59E0B', DDPG: '#EC4899', Ensemble: '#16A34A',
}
const ALGO_DESC: Record<AgentType, string> = {
  PPO: 'Clipped policy gradient', SAC: 'Entropy-based exploration',
  TD3: 'Twin delayed actor-critic', A2C: 'Advantage actor-critic',
  DDPG: 'Deterministic policy gradient', Ensemble: 'Best-of-all average',
}

function getM(data: RLSummaryResponse, algo: AgentType, field: string): number {
  const key = `${ALGO_PREFIX[algo]}_${field}` as keyof RLSummaryResponse
  const v = data[key]
  return typeof v === 'number' ? v : 0
}

export default function RlAgent() {
  const [agent, setAgent] = useState<AgentType>('PPO');
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
    <div className="space-y-6">
      <Skeleton className="h-8 w-56 mb-2" rounded="lg" />
      <div className="flex gap-2">
        {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-9 w-24" rounded="xl" />)}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <MetricCardSkeleton key={i} />)}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Skeleton className="h-64" rounded="xl" />
        <Skeleton className="h-64" rounded="xl" />
      </div>
    </div>
  );

  if (error || !data) return (
    <div className="flex flex-col items-center justify-center h-96 gap-4">
      <AlertTriangle size={32} className="text-loss" />
      <p className="text-loss text-sm font-medium">{error || 'No data available'}</p>
    </div>
  );

  const ak = ALGO_PREFIX[agent]
  const rewardData = data.reward_curve.map(r => ({
    episode: r.episode,
    ppo: r.ppo_reward, sac: r.sac_reward, td3: r.td3_reward,
    a2c: r.a2c_reward, ddpg: r.ddpg_reward, ensemble: r.ensemble_reward,
  }));
  const weightKey = `${ak}_weight` as keyof typeof data.weights[0]
  const weightData = data.weights.slice(0, 15).map(w => {
    const v = w[weightKey];
    return { name: w.ticker, weight: typeof v === 'number' ? v : 0, sector: w.sector };
  });

  // Sector allocation pie data
  const sectorPieData = data.sector_allocation.map(s => {
    const sv = s[`${ak}_weight` as keyof typeof s];
    return { name: s.sector, value: typeof sv === 'number' ? sv : 0, color: SECTOR_COLORS[s.sector] || '#9CA3AF' };
  }).filter(d => d.value > 1);

  // All-algo comparison rows
  const allAlgos: AgentType[] = ['PPO', 'SAC', 'TD3', 'A2C', 'DDPG', 'Ensemble']

  // Per-column best values
  const colBest = {
    sharpe:        Math.max(...allAlgos.map(a => getM(data, a, 'sharpe'))),
    sortino:       Math.max(...allAlgos.map(a => getM(data, a, 'sortino'))),
    annual_return: Math.max(...allAlgos.map(a => getM(data, a, 'annual_return'))),
    annual_vol:    Math.min(...allAlgos.map(a => getM(data, a, 'annual_vol'))),
    max_drawdown:  Math.max(...allAlgos.map(a => getM(data, a, 'max_drawdown'))), // closest to 0 = highest
  }

  // Data-driven "Recommended" badge: composite score = Sharpe + Sortino*0.5 + annRet*0.02 - |maxDD|*2
  const recommendedAlgo = allAlgos.reduce<AgentType>((best, a) => {
    const score = (algo: AgentType) =>
      getM(data, algo, 'sharpe') +
      getM(data, algo, 'sortino') * 0.5 +
      getM(data, algo, 'annual_return') * 0.02 -
      Math.abs(getM(data, algo, 'max_drawdown')) * 2
    return score(a) > score(best) ? a : best
  }, allAlgos[0])


  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="RL Agent Monitor"
          subtitle={`5 algorithms + Ensemble on real NIFTY 50 — ${data.ppo_episodes} episodes — out-of-sample validation`}
          icon={<Brain size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* 6-Algorithm Selector */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        {(Object.keys(ALGO_PREFIX) as AgentType[]).map(a => (
          <motion.button
            key={a}
            onClick={() => { setAgent(a); toast.info(`Switched to ${a} — ${ALGO_DESC[a]}`); }}
            whileTap={{ scale: 0.94 }}
            whileHover={{ y: -1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 20 }}
            className={`relative px-4 py-2 rounded-xl text-sm font-medium border overflow-hidden transition-colors ${
              agent === a
                ? 'text-white border-transparent shadow-md'
                : 'bg-bg-card text-text-secondary hover:text-text border-border'
            }`}
            style={agent === a ? { backgroundColor: ALGO_COLORS[a], borderColor: ALGO_COLORS[a] } : {}}
          >
            {agent === a && (
              <motion.span
                layoutId="algo-active-bg"
                className="absolute inset-0 rounded-xl"
                style={{ backgroundColor: ALGO_COLORS[a] }}
                transition={{ type: 'spring', stiffness: 300, damping: 28 }}
              />
            )}
            <span className="relative z-10">{a === 'Ensemble' ? '★ Ensemble' : a}</span>
          </motion.button>
        ))}
        <motion.span
          key={agent}
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-xs text-text-muted ml-1 italic"
        >
          {ALGO_DESC[agent]}
        </motion.span>
      </div>

      {/* Metric Cards — selected algorithm */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
        <MetricCard title="Episodes" value={getM(data, agent, 'episodes')} decimals={0} icon={<Target size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Episodes' ? null : 'Episodes')} active={expandedMetric === 'Episodes'} />
        <MetricCard title="Avg Reward" value={getM(data, agent, 'avg_reward')} decimals={4} icon={<Zap size={18} />}
          badge={getRLBadge('reward', getM(data, agent, 'avg_reward'))}
          onClick={() => setExpandedMetric(m => m === 'Avg Reward' ? null : 'Avg Reward')} active={expandedMetric === 'Avg Reward'} />
        <MetricCard title="Sharpe (Val)" value={getM(data, agent, 'sharpe')} decimals={4} icon={<BarChart3 size={18} />}
          badge={getRLBadge('sharpe', getM(data, agent, 'sharpe'))}
          onClick={() => setExpandedMetric(m => m === 'Sharpe (Val)' ? null : 'Sharpe (Val)')} active={expandedMetric === 'Sharpe (Val)'} />
        <MetricCard title="Max Drawdown" value={getM(data, agent, 'max_drawdown') * 100} decimals={2} suffix="%"
          badge={getRLBadge('drawdown', getM(data, agent, 'max_drawdown'))}
          onClick={() => setExpandedMetric(m => m === 'Max Drawdown' ? null : 'Max Drawdown')} active={expandedMetric === 'Max Drawdown'} />
      </motion.div>
      <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />

      {/* 6-Algorithm Comparison Table */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">All Algorithms — Comparison</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 font-medium text-text-secondary">Algorithm</th>
                <th className="text-right py-2 font-medium text-text-secondary">Sharpe</th>
                <th className="text-right py-2 font-medium text-text-secondary">Sortino</th>
                <th className="text-right py-2 font-medium text-text-secondary">Ann. Return</th>
                <th className="text-right py-2 font-medium text-text-secondary">Volatility</th>
                <th className="text-right py-2 font-medium text-text-secondary">Max DD</th>
              </tr>
            </thead>
            <tbody>
              {allAlgos.map((a, rowIdx) => {
                const isRecommended = a === recommendedAlgo
                const sharpe = getM(data, a, 'sharpe')
                const sortino = getM(data, a, 'sortino')
                const annRet = getM(data, a, 'annual_return')
                const annVol = getM(data, a, 'annual_vol')
                const dd = getM(data, a, 'max_drawdown')
                const barPct = colBest.sharpe > 0 ? Math.max(0, (sharpe / colBest.sharpe)) * 100 : 0
                return (
                  <motion.tr
                    key={a}
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: rowIdx * 0.05, type: 'spring', stiffness: 180, damping: 20 }}
                    onClick={() => setAgent(a)}
                    className={`border-b border-border-light cursor-pointer transition-colors ${
                      agent === a
                        ? 'bg-primary-subtle border-l-[3px] border-l-primary'
                        : 'hover:bg-bg-card'
                    } ${isRecommended ? 'font-semibold' : ''}`}
                  >
                    <td className="py-2.5 pl-3 flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: ALGO_COLORS[a] }} />
                      {a}
                      {isRecommended && <Badge variant="profit">Recommended</Badge>}
                    </td>
                    {/* Sharpe with fill bar */}
                    <td className={`py-2.5 text-right font-mono relative pr-3 ${sharpe === colBest.sharpe ? 'text-profit font-bold' : ''}`}>
                      <div
                        className="absolute inset-y-1 right-0 rounded-l opacity-[0.12]"
                        style={{ width: `${barPct}%`, background: ALGO_COLORS[a] }}
                      />
                      <span className="relative z-10">{sharpe.toFixed(4)}</span>
                    </td>
                    <td className={`py-2.5 text-right font-mono ${sortino === colBest.sortino ? 'text-profit font-bold' : ''}`}>
                      {sortino.toFixed(4)}
                    </td>
                    <td className={`py-2.5 text-right font-mono ${annRet === colBest.annual_return ? 'text-profit font-bold' : annRet >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {annRet >= 0 ? '+' : ''}{annRet.toFixed(2)}%
                    </td>
                    <td className={`py-2.5 text-right font-mono ${annVol === colBest.annual_vol ? 'text-profit font-bold' : 'text-text-secondary'}`}>
                      {annVol.toFixed(2)}%
                    </td>
                    <td className={`py-2.5 text-right font-mono ${dd === colBest.max_drawdown ? 'text-profit font-bold' : 'text-loss'}`}>
                      {(dd * 100).toFixed(2)}%
                    </td>
                  </motion.tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Charts Section with Tab Switcher */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display font-bold text-lg text-secondary">Performance Charts</h2>
          <div className="flex bg-bg-card rounded-lg border border-border-light p-0.5 relative">
            {([
              { key: 'rewards', label: 'Training Rewards' },
              { key: 'cumulative', label: 'Cumulative Returns' },
              { key: 'weights', label: 'Portfolio Weights' },
            ] as const).map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className="relative px-3 py-1.5 rounded-md text-xs font-medium z-10 transition-colors"
                style={{ color: activeTab === tab.key ? '#fff' : undefined }}
              >
                {activeTab === tab.key && (
                  <motion.span
                    layoutId="chart-tab-pill"
                    className="absolute inset-0 bg-primary rounded-md shadow-sm"
                    transition={{ type: 'spring', stiffness: 340, damping: 28 }}
                  />
                )}
                <span className={`relative z-10 ${activeTab === tab.key ? 'text-white' : 'text-text-secondary hover:text-text'}`}>
                  {tab.label}
                </span>
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
                    {(Object.entries(ALGO_COLORS) as [AgentType, string][]).map(([a, c]) => (
                      <linearGradient key={a} id={`grad-rw-${a}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={c} stopOpacity={a === 'Ensemble' ? 0 : 0.12} />
                        <stop offset="100%" stopColor={c} stopOpacity={0} />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="episode" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="ppo" name="PPO" stroke="#C15F3C" strokeWidth={1.5} fill="url(#grad-rw-PPO)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="sac" name="SAC" stroke="#6366F1" strokeWidth={1.5} fill="url(#grad-rw-SAC)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="td3" name="TD3" stroke="#0D9488" strokeWidth={1.5} fill="url(#grad-rw-TD3)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="a2c" name="A2C" stroke="#F59E0B" strokeWidth={1.5} fill="url(#grad-rw-A2C)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="ddpg" name="DDPG" stroke="#EC4899" strokeWidth={1.5} fill="url(#grad-rw-DDPG)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="ensemble" name="Ensemble" stroke="#16A34A" strokeWidth={3} fill="none" dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </motion.div>
          )}

          {/* Cumulative Returns */}
          {activeTab === 'cumulative' && (
            <motion.div key="cumulative" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
              <p className="text-xs text-text-secondary mb-3">
                Portfolio growth on out-of-sample validation data. All 6 strategies vs equal-weight baseline — real returns the agents never trained on.
              </p>
              <ResponsiveContainer width="100%" height={340} minHeight={1}>
                <AreaChart data={data.cumulative_returns} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                  <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
                  <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                    formatter={(v) => `${Number(v).toFixed(2)}%`} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="ppo" name="PPO" stroke="#C15F3C" strokeWidth={1.5} fill="none" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="sac" name="SAC" stroke="#6366F1" strokeWidth={1.5} fill="none" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="td3" name="TD3" stroke="#0D9488" strokeWidth={1.5} fill="none" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="a2c" name="A2C" stroke="#F59E0B" strokeWidth={1.5} fill="none" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="ddpg" name="DDPG" stroke="#EC4899" strokeWidth={1.5} fill="none" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="ensemble" name="Ensemble" stroke="#16A34A" strokeWidth={3} fill="none" dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="equal_weight" name="Equal Weight" stroke="#9CA3AF" strokeWidth={1.5} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
              {(() => {
                const last = data.cumulative_returns[data.cumulative_returns.length - 1]
                const cols = [
                  { label: 'PPO',      value: last?.ppo,          color: '#C15F3C' },
                  { label: 'SAC',      value: last?.sac,          color: '#6366F1' },
                  { label: 'TD3',      value: last?.td3,          color: '#0D9488' },
                  { label: 'A2C',      value: last?.a2c,          color: '#F59E0B' },
                  { label: 'DDPG',     value: last?.ddpg,         color: '#EC4899' },
                  { label: '★ Ensemble', value: last?.ensemble,   color: '#16A34A' },
                  { label: 'Equal Wt', value: last?.equal_weight, color: '#9CA3AF' },
                ]
                return (
                  <div className="flex flex-wrap justify-center gap-4 mt-3">
                    {cols.map(s => (
                      <div key={s.label} className="text-center">
                        <span className="text-xs text-text-muted">{s.label}</span>
                        <p className="font-mono font-bold text-base" style={{ color: s.color }}>
                          {(s.value ?? 0) > 0 ? '+' : ''}{(s.value ?? 0).toFixed(1)}%
                        </p>
                      </div>
                    ))}
                  </div>
                )
              })()}
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
                    formatter={(v) => `${Number(v).toFixed(2)}%`} />
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
                formatter={(v) => `${Number(v).toFixed(1)}%`} />
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
            Return Contribution — {agent}
          </h2>
          <p className="text-xs text-text-secondary mb-3">
            Which stocks drive portfolio returns for {agent}. Contribution = weight × stock return.
          </p>
          {(() => {
            // Recompute contributions from selected algo's weights
            const agentContribs = data.weights.map(w => {
              const wPct = (w[`${ak}_weight` as keyof typeof w] as number) ?? 0
              const sc = data.stock_contributions.find(s => s.ticker === w.ticker)
              const cumRet = sc?.cumulative_return ?? 0
              return { ticker: w.ticker, sector: w.sector, weight: wPct, cumulative_return: cumRet, return_contrib: (wPct / 100) * cumRet }
            }).filter(s => s.weight > 0).sort((a, b) => b.return_contrib - a.return_contrib)
            const maxContrib = Math.max(...agentContribs.map(s => Math.abs(s.return_contrib)), 0.01);
            return (
              <div className="space-y-1.5 max-h-[340px] overflow-y-auto">
                {agentContribs.map((s, i) => {
                  const barPct = (Math.abs(s.return_contrib) / maxContrib) * 100;
                  return (
                    <motion.div key={s.ticker}
                      initial={{ opacity: 0, x: -15 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                      className="relative flex items-center gap-3 p-2.5 rounded-xl bg-bg-card hover:bg-primary-subtle/20 transition-colors overflow-hidden"
                    >
                      {/* Contribution fill bar */}
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${barPct}%` }}
                        transition={{ delay: i * 0.04 + 0.3, duration: 0.6, ease: 'easeOut' }}
                        className="absolute inset-y-0 left-0 opacity-[0.08] rounded-xl"
                        style={{ background: s.return_contrib >= 0 ? '#16A34A' : '#DC2626' }}
                      />
                      <span className="w-6 text-center text-[10px] font-bold text-text-muted relative z-10">{i + 1}</span>
                      <div className="w-3 h-3 rounded-full shrink-0 relative z-10" style={{ backgroundColor: SECTOR_COLORS[s.sector] || '#9CA3AF' }} />
                      <div className="flex-1 min-w-0 relative z-10">
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
                      <div className="text-right relative z-10">
                        <span className={`inline-flex items-center gap-0.5 font-mono text-sm font-bold ${
                          s.return_contrib >= 0 ? 'text-profit' : 'text-loss'
                        }`}>
                          {s.return_contrib >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                          {s.return_contrib > 0 ? '+' : ''}{s.return_contrib.toFixed(2)}%
                        </span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            );
          })()}
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
