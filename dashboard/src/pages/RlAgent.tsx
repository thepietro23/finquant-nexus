import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Target, BarChart3, Loader2, AlertTriangle } from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, BarChart, Bar, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { RLSummaryResponse } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import { staggerContainer } from '../lib/animations';

const PAGE_INFO = {
  title: 'RL Agent Monitor — What Does This Page Show?',
  sections: [
    { heading: 'What is this page?', text: 'Monitors two Deep Reinforcement Learning agents (PPO and SAC) that learn to manage a stock portfolio by trial-and-error on real NIFTY 50 price data.' },
    { heading: 'PPO vs SAC', text: 'PPO (Proximal Policy Optimization) is more stable during training. SAC (Soft Actor-Critic) uses entropy bonus for better exploration. Both are state-of-the-art RL algorithms for continuous action spaces.' },
    { heading: 'Training Reward Curve', text: 'Shows Sharpe Ratio achieved per episode. An episode = 252 trading days. Reward increases over time as agents learn better allocation strategies from real returns.' },
    { heading: 'Portfolio Weights', text: 'Shows the top 15 stocks by weight as decided by the selected agent. Higher weight = agent believes this stock contributes more to risk-adjusted returns.' },
    { heading: 'Constraints', text: 'Max 20% per stock (diversification), -5% stop loss per stock, -15% circuit breaker for total portfolio, 0.1% transaction cost + 0.05% slippage (real Indian market costs).' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'Episodes': {
    what: 'Number of training episodes completed. Each episode simulates 252 trading days (1 year) of portfolio management.',
    why: 'More episodes = more experience. The agent sees different market conditions and learns to handle bull markets, crashes, and sideways periods.',
    how: 'Train period (2015-2021) is divided into 252-day windows. Agent trades through each window and receives cumulative reward.',
    good: '> 50 episodes = sufficient for convergence | > 200 = well-trained | Our episodes are limited by available training data length.',
  },
  'Avg Reward': {
    what: 'Average Sharpe Ratio achieved over the last 10 episodes. Higher = agent is making better risk-adjusted returns.',
    why: 'Reward = Sharpe Ratio. This directly measures the agent\'s objective: maximize return per unit of risk.',
    how: 'Mean of Sharpe Ratios from the last 10 training episodes. Uses 7% risk-free rate, 248 trading days per year.',
    good: '< 0.3 = still learning | 0.3–0.7 = decent | 0.7–1.2 = good | > 1.2 = excellent',
  },
  'Sharpe (Val)': {
    what: 'Sharpe Ratio of the agent\'s portfolio on the validation period (2022-2023). This is out-of-sample performance — data the agent never trained on.',
    why: 'Training Sharpe can be misleading (overfitting). Validation Sharpe shows if the agent truly learned generalizable trading strategies.',
    how: 'Agent\'s final weights applied to validation period returns. Sharpe = (Return - 7%) / Volatility.',
    good: '> 0.5 = promising generalization | > 0.8 = strong | > 1.0 = excellent',
  },
  'Max Drawdown': {
    what: 'Worst peak-to-trough decline of the agent\'s portfolio during the validation period.',
    why: 'Even a profitable agent can have scary drops. Max Drawdown reveals the worst pain point an investor would experience.',
    how: 'Track cumulative portfolio value on validation data. Measure largest drop from any peak to subsequent trough.',
    good: '> -5% = excellent | -5% to -15% = acceptable | < -15% = triggers the circuit breaker constraint',
  },
};

export default function RlAgent() {
  const [agent, setAgent] = useState<'PPO' | 'SAC'>('PPO');
  const [data, setData] = useState<RLSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

  useEffect(() => {
    api.rlSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load RL data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <Loader2 size={32} className="animate-spin text-primary" />
        <p className="text-text-secondary text-sm">Loading RL agent data from real stock returns...</p>
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

  const isPPO = agent === 'PPO';
  const rewardData = data.reward_curve.map(r => ({
    episode: r.episode,
    ppo: r.ppo_reward,
    sac: r.sac_reward,
  }));

  const weightData = data.weights.map(w => ({
    name: w.ticker,
    weight: isPPO ? w.ppo_weight : w.sac_weight,
    sector: w.sector,
  }));

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="RL Agent Monitor"
          subtitle={`PPO + SAC trained on real NIFTY 50 returns — ${data.ppo_episodes} episodes — ${data.constraints.max_position * 100}% max position`}
          icon={<Brain size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Agent Toggle */}
      <div className="flex items-center gap-3 mb-6">
        {(['PPO', 'SAC'] as const).map(a => (
          <button key={a} onClick={() => setAgent(a)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              agent === a
                ? 'bg-primary text-white shadow-sm'
                : 'bg-bg-card text-text-secondary hover:bg-primary-subtle'
            }`}>
            {a}
          </button>
        ))}
      </div>

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

      {/* Training Reward Curve */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Training Progress — PPO vs SAC (Real Returns)
        </h2>
        <ResponsiveContainer width="100%" height={320} minHeight={1}>
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
            <XAxis dataKey="episode" tick={{ fontSize: 12, fill: '#9CA3AF' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{
              background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12,
              borderLeft: '3px solid #C15F3C', boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
            }} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 13 }} />
            <Area type="monotone" dataKey="ppo" name="PPO" stroke="#C15F3C"
              strokeWidth={2.5} fill="url(#grad-ppo)" dot={false} animationDuration={1500} />
            <Area type="monotone" dataKey="sac" name="SAC" stroke="#6366F1"
              strokeWidth={2} fill="url(#grad-sac)" dot={false} animationDuration={1500} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      {/* Portfolio Weights */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Portfolio Weights — {agent} Agent (Top 15 Stocks)
        </h2>
        <ResponsiveContainer width="100%" height={320} minHeight={1}>
          <BarChart data={weightData} margin={{ top: 10, right: 10, bottom: 40, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#6B7280' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} angle={-35} textAnchor="end" />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
              tickFormatter={(v: number) => `${v}%`} />
            <Tooltip contentStyle={{
              background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12,
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
            }} formatter={(v) => `${v}%`} />
            <Bar dataKey="weight" name="Weight %" radius={[6, 6, 0, 0]} animationDuration={800}>
              {weightData.map((_, i) => (
                <Cell key={i} fill={i % 2 === 0 ? '#C15F3C' : '#6366F1'} opacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex flex-wrap gap-4 mt-3 text-xs text-text-secondary">
          <span>Max position: {data.constraints.max_position * 100}%</span>
          <span>Stop loss: {data.constraints.stop_loss * 100}%</span>
          <span>Circuit breaker: {data.constraints.max_drawdown * 100}%</span>
          <span>Transaction cost: {data.constraints.transaction_cost * 100}%</span>
        </div>
      </Card>

      <p className="text-center text-xs text-text-muted mt-6 mb-2">
        Computed from real NIFTY 50 stock returns (2015-2021 train, 2022-2023 validation)
      </p>
    </div>
  );
}
