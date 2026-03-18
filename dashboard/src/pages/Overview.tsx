import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  LayoutDashboard, TrendingUp, TrendingDown, BarChart3, Activity,
  ArrowRight, ChevronUp, AlertTriangle, MessageSquare,
  Atom, ArrowUpDown, Loader2,
} from 'lucide-react';
import { staggerContainer } from '../lib/animations';
import { api } from '../lib/api';
import type { PortfolioSummaryResponse, PortfolioHolding } from '../lib/api';
import MetricCard from '../components/ui/MetricCard';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import PerformanceChart from '../components/charts/PerformanceChart';
import SectorDonut from '../components/charts/SectorDonut';
import PageInfoPanel from '../components/ui/PageInfoPanel';

const PAGE_INFO = {
  title: 'Portfolio Overview — What Does This Page Show?',
  sections: [
    { heading: 'What is this page?', text: 'The main dashboard showing your NIFTY 50 portfolio performance at a glance. All data is computed from real stock prices (Yahoo Finance, 2015-2025). Starting capital: ₹1 Crore.' },
    { heading: 'Metric cards', text: 'Portfolio Value (current worth), Sharpe Ratio (risk-adjusted return), Max Drawdown (worst loss), Annual Return (yearly %). Click any card for detailed explanation.' },
    { heading: 'Performance chart', text: 'Compares your portfolio cumulative return (orange area) against the NIFTY 50 index (dashed line) over time. If portfolio is above NIFTY = outperformance.' },
    { heading: 'Sector allocation', text: 'Donut chart showing how portfolio capital is distributed across 11 sectors. Equal-weight means sector weight depends on how many stocks that sector has.' },
    { heading: 'Top holdings', text: 'Top 10 stocks by cumulative return. Sortable by any column. Green = profit, Red = loss. Click "View all" to see all 44 stocks.' },
  ],
};

// ── Metric detail definitions ──
const METRIC_DETAILS: Record<string, {
  description: string;
  benchmark: string;
  interpretation: (data: PortfolioSummaryResponse) => string;
  link: string;
  linkLabel: string;
}> = {
  'Portfolio Value': {
    description: 'Total current value of the optimized NIFTY 50 portfolio, starting from ₹1 Crore capital with equal-weight allocation across all available stocks.',
    benchmark: 'Starting capital: ₹1 Cr | NIFTY 50 index used as benchmark',
    interpretation: (d) => {
      const ret = (d.annualized_return * 100).toFixed(1);
      return `The portfolio has generated ${ret}% annualized return over ${d.n_days} trading days (${d.date_start} to ${d.date_end}), computed from real NIFTY 50 stock prices.`;
    },
    link: '/portfolio',
    linkLabel: 'View detailed portfolio breakdown',
  },
  'Sharpe Ratio': {
    description: 'Risk-adjusted return metric. Measures excess return per unit of risk (volatility). Higher is better. Uses 7% risk-free rate (Indian govt bonds).',
    benchmark: 'Industry: >1.0 good, >1.5 excellent | Computed using 248 trading days/year',
    interpretation: (d) => {
      const sr = d.sharpe_ratio.toFixed(2);
      return `A Sharpe of ${sr} means for every 1% of risk taken, the portfolio earns ${sr}% excess return over the 7% risk-free rate. Based on real daily returns.`;
    },
    link: '/portfolio',
    linkLabel: 'View all risk metrics',
  },
  'Max Drawdown': {
    description: 'Worst peak-to-trough decline during the evaluation period. Measures the maximum loss an investor would have experienced.',
    benchmark: 'NIFTY 50 typical: -30% to -40% in crashes',
    interpretation: (d) => {
      const md = (d.max_drawdown * 100).toFixed(1);
      return `The maximum drawdown was ${md}% over the ${d.n_days}-day period. This measures worst-case loss from peak portfolio value.`;
    },
    link: '/stress',
    linkLabel: 'Run stress test scenarios',
  },
  'Annual Return': {
    description: 'Annualized percentage return of the portfolio, computed from real daily stock returns over the evaluation period.',
    benchmark: 'NIFTY 50 avg: 12-14% | FD rate: 7%',
    interpretation: (d) => {
      const ar = (d.annualized_return * 100).toFixed(1);
      const vol = (d.annualized_volatility * 100).toFixed(1);
      return `Annualized return of ${ar}% with ${vol}% volatility. Computed from ${d.n_stocks} real NIFTY 50 stocks using equal-weight allocation.`;
    },
    link: '/rl',
    linkLabel: 'View RL agent training details',
  },
};

// ── Sort types ──
type SortKey = 'ticker' | 'sector' | 'weight' | 'cumulative_return';
type SortDir = 'asc' | 'desc';

export default function Overview() {
  const navigate = useNavigate();
  const [data, setData] = useState<PortfolioSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('cumulative_return');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  useEffect(() => {
    api.portfolioSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load portfolio data — is the backend running?');
        setLoading(false);
      });
  }, []);

  // Sector data for donut chart
  const sectorData = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.sector_weights).map(([name, value]) => ({
      name, value: Math.round(value),
    }));
  }, [data]);

  // Top 10 holdings sorted
  const topHoldings = useMemo(() => {
    if (!data) return [];
    const sorted = [...data.holdings].sort((a, b) => {
      const valA = a[sortKey];
      const valB = b[sortKey];
      if (typeof valA === 'string') return sortDir === 'asc' ? valA.localeCompare(valB as string) : (valB as string).localeCompare(valA);
      return sortDir === 'asc' ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
    });
    return sorted.slice(0, 10);
  }, [data, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  }

  function toggleMetric(title: string) {
    setExpandedMetric(prev => prev === title ? null : title);
  }

  const SortIcon = ({ col }: { col: SortKey }) => (
    <ArrowUpDown size={12} className={`inline ml-1 ${sortKey === col ? 'text-primary' : 'text-text-muted'}`} />
  );

  // Loading state
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <Loader2 size={32} className="animate-spin text-primary" />
        <p className="text-text-secondary text-sm">Loading real portfolio data...</p>
      </div>
    );
  }

  // Error state
  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <AlertTriangle size={32} className="text-loss" />
        <p className="text-loss text-sm font-medium">{error || 'No data available'}</p>
        <p className="text-text-muted text-xs">Make sure the backend is running: uvicorn src.api.main:app --port 8000</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Portfolio Overview"
          subtitle={`${data.n_stocks} NIFTY 50 stocks — ${data.date_start} to ${data.date_end} (${data.n_days} trading days)`}
          icon={<LayoutDashboard size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* ── Metric Cards Row ── */}
      <motion.div
        variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-2"
      >
        <MetricCard
          title="Portfolio Value" prefix="₹" value={data.portfolio_value} decimals={0}
          suffix="" change={data.annualized_return}
          icon={<TrendingUp size={18} />}
          onClick={() => toggleMetric('Portfolio Value')}
          active={expandedMetric === 'Portfolio Value'}
        />
        <MetricCard
          title="Sharpe Ratio" value={data.sharpe_ratio} decimals={2}
          icon={<BarChart3 size={18} />}
          onClick={() => toggleMetric('Sharpe Ratio')}
          active={expandedMetric === 'Sharpe Ratio'}
        />
        <MetricCard
          title="Max Drawdown" value={data.max_drawdown * 100} decimals={1} suffix="%"
          icon={<TrendingDown size={18} />}
          onClick={() => toggleMetric('Max Drawdown')}
          active={expandedMetric === 'Max Drawdown'}
        />
        <MetricCard
          title="Annual Return" value={data.annualized_return * 100} decimals={1} suffix="%"
          icon={<Activity size={18} />}
          onClick={() => toggleMetric('Annual Return')}
          active={expandedMetric === 'Annual Return'}
        />
      </motion.div>

      {/* ── Expanded Metric Detail Panel ── */}
      <AnimatePresence>
        {expandedMetric && METRIC_DETAILS[expandedMetric] && (
          <motion.div
            key={expandedMetric}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 25 }}
            className="overflow-hidden mb-4"
          >
            <div className="bg-primary-subtle/50 border border-primary-light rounded-2xl p-5 mt-2">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <h3 className="font-display font-bold text-base text-secondary mb-2">
                    {expandedMetric}
                  </h3>
                  <p className="text-sm text-text-secondary mb-3">
                    {METRIC_DETAILS[expandedMetric].description}
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
                    <div className="bg-white/70 rounded-xl px-4 py-3">
                      <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">Benchmark</p>
                      <p className="text-sm text-text font-mono">{METRIC_DETAILS[expandedMetric].benchmark}</p>
                    </div>
                    <div className="bg-white/70 rounded-xl px-4 py-3">
                      <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">Interpretation</p>
                      <p className="text-sm text-text-secondary">{METRIC_DETAILS[expandedMetric].interpretation(data)}</p>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setExpandedMetric(null)}
                  className="shrink-0 p-1.5 rounded-lg hover:bg-white/50 transition-colors text-text-muted"
                >
                  <ChevronUp size={18} />
                </button>
              </div>
              <button
                onClick={() => navigate(METRIC_DETAILS[expandedMetric].link)}
                className="flex items-center gap-2 text-sm font-medium text-primary hover:text-primary-hover transition-colors"
              >
                {METRIC_DETAILS[expandedMetric].linkLabel}
                <ArrowRight size={14} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Quick Actions ── */}
      <div className="flex flex-wrap gap-2 mb-6">
        {[
          { label: 'Analyze Sentiment', icon: MessageSquare, path: '/sentiment' },
          { label: 'Stress Test', icon: AlertTriangle, path: '/stress' },
          { label: 'Run QAOA', icon: Atom, path: '/quantum' },
          { label: 'View Graph', icon: Activity, path: '/graph' },
        ].map(action => (
          <button key={action.label}
            onClick={() => navigate(action.path)}
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-xl border border-border
              text-text-secondary hover:border-primary hover:text-primary hover:bg-primary-subtle transition-all"
          >
            <action.icon size={14} />
            {action.label}
          </button>
        ))}
      </div>

      {/* ── Performance Chart (real data) ── */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display font-bold text-lg text-secondary">Performance vs NIFTY 50</h2>
          <span className="text-xs text-text-muted font-mono">{data.date_start} → {data.date_end}</span>
        </div>
        <PerformanceChart data={data.performance} />
      </Card>

      {/* ── Two columns: Sector + Holdings ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display font-bold text-lg text-secondary">Sector Allocation</h2>
            <button onClick={() => navigate('/portfolio')}
              className="text-xs text-primary font-medium flex items-center gap-1 hover:text-primary-hover transition-colors">
              Details <ArrowRight size={12} />
            </button>
          </div>
          {sectorData.length > 0 ? (
            <SectorDonut data={sectorData} />
          ) : (
            <div className="h-60 flex items-center justify-center text-text-muted">No sector data</div>
          )}
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display font-bold text-lg text-secondary">Top Holdings</h2>
            <button onClick={() => navigate('/portfolio')}
              className="text-xs text-primary font-medium flex items-center gap-1 hover:text-primary-hover transition-colors">
              View all {data.n_stocks} <ArrowRight size={12} />
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 font-medium text-text-secondary cursor-pointer select-none"
                    onClick={() => toggleSort('ticker')}>Stock <SortIcon col="ticker" /></th>
                  <th className="text-left py-2 font-medium text-text-secondary cursor-pointer select-none"
                    onClick={() => toggleSort('sector')}>Sector <SortIcon col="sector" /></th>
                  <th className="text-right py-2 font-medium text-text-secondary cursor-pointer select-none"
                    onClick={() => toggleSort('weight')}>Weight <SortIcon col="weight" /></th>
                  <th className="text-right py-2 font-medium text-text-secondary cursor-pointer select-none"
                    onClick={() => toggleSort('cumulative_return')}>Return <SortIcon col="cumulative_return" /></th>
                </tr>
              </thead>
              <tbody>
                {topHoldings.map((h: PortfolioHolding) => (
                  <tr key={h.ticker} className="border-b border-border-light hover:bg-bg-card transition-colors">
                    <td className="py-2.5 font-mono font-medium text-text">{h.ticker.replace('.NS', '')}</td>
                    <td className="py-2.5 text-text-secondary">{h.sector}</td>
                    <td className="py-2.5 text-right font-mono">{h.weight.toFixed(1)}%</td>
                    <td className={`py-2.5 text-right font-mono font-medium ${
                      h.cumulative_return >= 0 ? 'text-profit' : 'text-loss'
                    }`}>
                      {h.cumulative_return > 0 ? '+' : ''}{h.cumulative_return.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* ── Data Source Note ── */}
      <p className="text-center text-xs text-text-muted mt-6 mb-2">
        Real data from {data.n_stocks} NIFTY 50 stocks — Click metric cards for details — Click table headers to sort
      </p>
    </div>
  );
}
