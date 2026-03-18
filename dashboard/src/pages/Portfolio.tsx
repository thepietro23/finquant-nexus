import { useEffect, useState } from 'react';
import { PieChart, Loader2, AlertTriangle, TrendingUp, TrendingDown, X } from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, AreaChart, Area,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../lib/api';
import type { PortfolioSummaryResponse, StockDetailResponse } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import MetricCard from '../components/ui/MetricCard';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import type { MetricDetail } from '../components/ui/MetricInfoPanel';
import { staggerContainer } from '../lib/animations';

// ── Metric info definitions ──
function getMetricDetails(data: PortfolioSummaryResponse): Record<string, MetricDetail & { interpret: string }> {
  return {
    'Sharpe Ratio': {
      what: 'Sharpe Ratio measures how much excess return you earn for each unit of risk (volatility). It is the gold standard of risk-adjusted performance metrics.',
      why: 'A 20% return sounds great — but if the volatility is 50%, you are riding a rollercoaster. Sharpe tells you the real truth: return per unit of risk.',
      how: 'Formula: (Portfolio Return − Risk-Free Rate) / Volatility. India risk-free rate = 7% (govt bonds). Annualized using 248 trading days.',
      good: '< 0 = losing money | 0–0.5 = average | 0.5–1.0 = good | 1.0–1.5 = very good | > 1.5 = exceptional (hedge fund level)',
      interpret: `Your portfolio Sharpe is ${data.sharpe_ratio.toFixed(4)}. This means for every 1% of risk, you earn ${data.sharpe_ratio.toFixed(2)}% excess return above the 7% risk-free rate. Computed from ${data.n_stocks} real NIFTY 50 stocks with equal-weight allocation.`,
    },
    'Sortino Ratio': {
      what: 'Sortino Ratio is like Sharpe but smarter — it only considers downside risk (losses). Upside volatility (gains) is not penalized because gains are desirable.',
      why: 'Sharpe penalizes both up and down movements equally. But going up is good! Sortino only penalizes downside moves, giving a more accurate picture of risk-adjusted returns.',
      how: 'Formula: (Portfolio Return − Risk-Free Rate) / Downside Deviation. Downside Deviation = standard deviation of only negative returns.',
      good: 'Sortino is always ≥ Sharpe (smaller denominator). > 1.0 = good | > 2.0 = excellent | > 3.0 = exceptional',
      interpret: `Sortino is ${data.sortino_ratio.toFixed(4)} vs Sharpe ${data.sharpe_ratio.toFixed(4)}. The gap means your upside volatility exceeds your downside — the portfolio gains more than it loses on average.`,
    },
    'Annual Return': {
      what: 'The annualized percentage return of the portfolio over the evaluation period. This is the most intuitive metric — "how much did I earn?"',
      why: 'Compare directly with alternatives: Fixed Deposit (7%), NIFTY 50 index (12-14%), or a good mutual fund (15-18%).',
      how: 'Take the mean daily return, then annualize: mean × √248 (geometric: (1 + mean_daily)^248 − 1). 248 = trading days per year in India.',
      good: '> 7% = beats FD | > 12% = competitive with NIFTY | > 18% = excellent | > 25% = exceptional',
      interpret: (() => {
        const ar = (data.annualized_return * 100).toFixed(2);
        const vs_fd = (data.annualized_return * 100 - 7).toFixed(1);
        return `Portfolio delivered ${ar}% annualized return — that is ${vs_fd}% above the FD rate (7%). Computed from ${data.n_days} real trading days.`;
      })(),
    },
    'Volatility': {
      what: 'Annualized volatility measures how much the portfolio value fluctuates day-to-day. Higher volatility = more uncertainty = more risk.',
      why: 'High returns with high volatility means wild swings. Low volatility with decent returns is the sweet spot for consistent wealth building.',
      how: 'Standard deviation of daily returns × √248. Converts daily volatility to annual scale.',
      good: '< 10% = low risk | 10–15% = moderate | 15–25% = high | > 25% = very high risk',
      interpret: (() => {
        const vol = (data.annualized_volatility * 100).toFixed(2);
        const daily = (data.annualized_volatility / Math.sqrt(248) * 100).toFixed(2);
        return `Portfolio volatility is ${vol}% annually (~${daily}% daily). This is in the moderate range — typical for a diversified NIFTY 50 equal-weight portfolio.`;
      })(),
    },
    'Max Drawdown': {
      what: 'Maximum Drawdown is the worst peak-to-trough decline during the evaluation period. It answers: "What is the maximum loss I would have experienced?"',
      why: 'Every investor fears crashes. Max Drawdown tells you the worst-case scenario actually observed in your portfolio — not hypothetical, but real.',
      how: 'Track cumulative portfolio value daily. Record the running peak. Drawdown = (current − peak) / peak. Max Drawdown = the largest such drop.',
      good: '> −5% = excellent | −5% to −10% = good | −10% to −20% = moderate | < −20% = concerning (NIFTY crashes: −30% to −40%)',
      interpret: (() => {
        const md = (data.max_drawdown * 100).toFixed(2);
        return `Max drawdown was ${md}%. The portfolio dropped at most ${Math.abs(data.max_drawdown * 100).toFixed(1)}% from its peak. For reference, NIFTY 50 dropped 30-40% in 2008 and COVID crashes — this portfolio was much better controlled.`;
      })(),
    },
  };
}

const PAGE_INFO = {
  title: 'Portfolio Analysis — What Does This Page Show?',
  sections: [
    {
      heading: 'What is this page?',
      text: 'A complete breakdown of your NIFTY 50 portfolio with 44 real stocks. Data sourced from Yahoo Finance (2015-2025). Equal-weight allocation means each stock gets the same investment (₹1 Crore ÷ 44 ≈ ₹2.27 Lakh per stock).',
    },
    {
      heading: 'What are the 5 metric cards?',
      text: 'Sharpe Ratio (risk-adjusted return), Sortino Ratio (downside-only risk), Annual Return (yearly %), Volatility (fluctuation risk), Max Drawdown (worst loss). Click any card for a detailed explanation.',
    },
    {
      heading: 'What is the Sector Weights chart?',
      text: 'Shows how portfolio capital is distributed across 11 NIFTY 50 sectors. In equal-weight, sectors with more stocks have higher weight. E.g., Energy+Auto with 20 stocks ≈ 45% weight.',
    },
    {
      heading: 'What is the Holdings table?',
      text: 'All 44 stocks listed with ticker, sector, portfolio weight (%), and cumulative return. Green = profit, Red = loss. Sorted by return descending (best performers on top).',
    },
    {
      heading: 'Where does the data come from?',
      text: 'Backend (FastAPI) reads real CSV files and computes all metrics on the fly. No hardcoded or random values. Source: data/all_close_prices.csv from Yahoo Finance.',
    },
  ],
};

export default function Portfolio() {
  const [data, setData] = useState<PortfolioSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [selectedStock, setSelectedStock] = useState<StockDetailResponse | null>(null);
  const [stockLoading, setStockLoading] = useState(false);

  function handleStockClick(ticker: string) {
    if (selectedStock?.ticker === ticker.replace('.NS', '')) {
      setSelectedStock(null);
      return;
    }
    setStockLoading(true);
    api.stockDetail(ticker.replace('.NS', ''))
      .then(d => { setSelectedStock(d); setStockLoading(false); })
      .catch(() => setStockLoading(false));
  }

  useEffect(() => {
    api.portfolioSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load portfolio data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-4">
        <Loader2 size={32} className="animate-spin text-primary" />
        <p className="text-text-secondary text-sm">Loading portfolio data...</p>
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

  const metricDetails = getMetricDetails(data);

  const allocData = Object.entries(data.sector_weights)
    .sort((a, b) => b[1] - a[1])
    .map(([sector, weight]) => ({ sector, weight: Math.round(weight) }));

  const COLORS = ['#C15F3C', '#6366F1', '#0D9488', '#F59E0B', '#EC4899', '#8B5CF6', '#10B981', '#F97316', '#06B6D4', '#EF4444', '#84CC16'];

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Portfolio Analysis"
          subtitle={`${data.n_stocks} stocks — ${data.date_start} to ${data.date_end} (${data.n_days} trading days) — Real NIFTY 50 data`}
          icon={<PieChart size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* ── Clickable Metric Cards ── */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-2">
        <MetricCard title="Sharpe Ratio" value={data.sharpe_ratio} decimals={4}
          onClick={() => setExpandedMetric(m => m === 'Sharpe Ratio' ? null : 'Sharpe Ratio')}
          active={expandedMetric === 'Sharpe Ratio'} />
        <MetricCard title="Sortino Ratio" value={data.sortino_ratio} decimals={4}
          onClick={() => setExpandedMetric(m => m === 'Sortino Ratio' ? null : 'Sortino Ratio')}
          active={expandedMetric === 'Sortino Ratio'} />
        <MetricCard title="Annual Return" value={data.annualized_return * 100} decimals={2} suffix="%"
          onClick={() => setExpandedMetric(m => m === 'Annual Return' ? null : 'Annual Return')}
          active={expandedMetric === 'Annual Return'} />
        <MetricCard title="Volatility" value={data.annualized_volatility * 100} decimals={2} suffix="%"
          onClick={() => setExpandedMetric(m => m === 'Volatility' ? null : 'Volatility')}
          active={expandedMetric === 'Volatility'} />
        <MetricCard title="Max Drawdown" value={data.max_drawdown * 100} decimals={2} suffix="%"
          onClick={() => setExpandedMetric(m => m === 'Max Drawdown' ? null : 'Max Drawdown')}
          active={expandedMetric === 'Max Drawdown'} />
      </motion.div>

      {/* ── Expanded Metric Info ── */}
      <MetricInfoPanel
        expandedMetric={expandedMetric}
        onClose={() => setExpandedMetric(null)}
        details={metricDetails}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6 mt-4">
        {/* Sector Allocation */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">Sector Weights</h2>
          <ResponsiveContainer width="100%" height={300} minHeight={1}>
            <BarChart data={allocData} layout="vertical" margin={{ top: 5, right: 10, bottom: 5, left: 80 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12, fill: '#9CA3AF' }}
                axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
                tickFormatter={(v: number) => `${v}%`} />
              <YAxis type="category" dataKey="sector" tick={{ fontSize: 11, fill: '#6B7280' }}
                axisLine={false} tickLine={false} width={75} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                formatter={(v) => `${v}%`} />
              <Bar dataKey="weight" radius={[0, 6, 6, 0]} animationDuration={800}>
                {allocData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Stock List */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">
            All Holdings ({data.n_stocks})
            <span className="text-xs font-normal text-text-muted ml-2">Click any stock for details</span>
          </h2>
          <div className="max-h-[300px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-border">
                  <th className="text-left py-2 font-medium text-text-secondary">#</th>
                  <th className="text-left py-2 font-medium text-text-secondary">Ticker</th>
                  <th className="text-left py-2 font-medium text-text-secondary">Sector</th>
                  <th className="text-right py-2 font-medium text-text-secondary">Weight</th>
                  <th className="text-right py-2 font-medium text-text-secondary">Return</th>
                </tr>
              </thead>
              <tbody>
                {data.holdings.map((h, i) => (
                  <tr key={h.ticker}
                    onClick={() => handleStockClick(h.ticker)}
                    className={`border-b border-border-light cursor-pointer transition-all ${
                      selectedStock?.ticker === h.ticker.replace('.NS', '')
                        ? 'bg-primary-subtle border-l-[3px] border-l-primary'
                        : 'hover:bg-bg-card hover:shadow-[0_2px_8px_rgba(193,95,60,0.06)]'
                    }`}>
                    <td className="py-2 text-text-muted font-mono">{i + 1}</td>
                    <td className="py-2 font-mono font-medium text-text">{h.ticker.replace('.NS', '')}</td>
                    <td className="py-2 text-text-secondary">{h.sector}</td>
                    <td className="py-2 text-right font-mono">{h.weight.toFixed(1)}%</td>
                    <td className={`py-2 text-right font-mono font-medium ${
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

      {/* ── Stock Detail Panel ── */}
      <AnimatePresence>
        {stockLoading && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex items-center gap-3 p-4 mb-6"
          >
            <Loader2 size={18} className="animate-spin text-primary" />
            <span className="text-sm text-text-secondary">Loading stock data...</span>
          </motion.div>
        )}

        {selectedStock && !stockLoading && (
          <motion.div
            key={selectedStock.ticker}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 25 }}
            className="overflow-hidden mb-6"
          >
            <div className="bg-white border border-primary-light rounded-2xl p-6 shadow-[0_10px_30px_rgba(193,95,60,0.08)]">
              {/* Header */}
              <div className="flex items-start justify-between mb-5">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="font-display font-bold text-xl text-secondary">{selectedStock.ticker}</h3>
                    <span className="px-2.5 py-0.5 rounded-lg text-xs font-medium bg-primary-subtle text-primary">{selectedStock.sector}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-3xl font-bold font-mono text-text">₹{selectedStock.current_price.toLocaleString('en-IN')}</span>
                    <span className={`flex items-center gap-1 text-sm font-medium px-2 py-1 rounded-lg ${
                      selectedStock.daily_change >= 0 ? 'bg-profit-light text-profit' : 'bg-loss-light text-loss'
                    }`}>
                      {selectedStock.daily_change >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                      {selectedStock.daily_change >= 0 ? '+' : ''}₹{selectedStock.daily_change.toLocaleString('en-IN')} ({selectedStock.daily_change_pct}%)
                    </span>
                  </div>
                  <p className="text-xs text-text-muted mt-1">Previous close: ₹{selectedStock.prev_close.toLocaleString('en-IN')}</p>
                </div>
                <button onClick={() => setSelectedStock(null)}
                  className="p-2 rounded-xl hover:bg-bg-card transition-colors text-text-muted">
                  <X size={18} />
                </button>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 mb-5">
                {[
                  { label: '52W High', value: `₹${selectedStock.high_52w.toLocaleString('en-IN')}`, color: 'text-profit' },
                  { label: '52W Low', value: `₹${selectedStock.low_52w.toLocaleString('en-IN')}`, color: 'text-loss' },
                  { label: '1Y Return', value: `${selectedStock.cumulative_return_1y > 0 ? '+' : ''}${selectedStock.cumulative_return_1y}%`, color: selectedStock.cumulative_return_1y >= 0 ? 'text-profit' : 'text-loss' },
                  { label: 'Volatility', value: `${selectedStock.annualized_volatility}%`, color: 'text-text' },
                  { label: 'Sharpe Ratio', value: selectedStock.sharpe_ratio.toFixed(4), color: 'text-text' },
                  { label: 'Max Drawdown', value: `${selectedStock.max_drawdown}%`, color: 'text-loss' },
                ].map(m => (
                  <div key={m.label} className="bg-bg-card rounded-xl px-3 py-2.5">
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-0.5">{m.label}</p>
                    <p className={`font-mono font-bold text-sm ${m.color}`}>{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Price Chart (60 days) */}
              <div>
                <p className="text-xs font-medium text-text-secondary mb-2">Price — Last 60 Days</p>
                <ResponsiveContainer width="100%" height={160} minHeight={1}>
                  <AreaChart data={selectedStock.price_history} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
                    <defs>
                      <linearGradient id="stockGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={selectedStock.cumulative_return_1y >= 0 ? '#16A34A' : '#DC2626'} stopOpacity={0.2} />
                        <stop offset="100%" stopColor={selectedStock.cumulative_return_1y >= 0 ? '#16A34A' : '#DC2626'} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                      domain={['auto', 'auto']} tickFormatter={(v: number) => `₹${(v / 1000).toFixed(1)}k`} />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12 }}
                      formatter={(v: number) => [`₹${v.toLocaleString('en-IN')}`, 'Price']} />
                    <Area type="monotone" dataKey="price"
                      stroke={selectedStock.cumulative_return_1y >= 0 ? '#16A34A' : '#DC2626'}
                      strokeWidth={2} fill="url(#stockGrad)" dot={false} animationDuration={800} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* 52W Range Bar */}
              <div className="mt-4">
                <p className="text-xs font-medium text-text-secondary mb-1.5">52-Week Range</p>
                <div className="relative h-2 bg-border-light rounded-full">
                  <div className="absolute h-2 bg-primary/30 rounded-full"
                    style={{
                      left: '0%',
                      width: `${((selectedStock.current_price - selectedStock.low_52w) / (selectedStock.high_52w - selectedStock.low_52w)) * 100}%`,
                    }} />
                  <div className="absolute w-3 h-3 bg-primary rounded-full -top-0.5 shadow-sm"
                    style={{
                      left: `${((selectedStock.current_price - selectedStock.low_52w) / (selectedStock.high_52w - selectedStock.low_52w)) * 100}%`,
                      transform: 'translateX(-50%)',
                    }} />
                </div>
                <div className="flex justify-between text-[10px] text-text-muted mt-1">
                  <span>₹{selectedStock.low_52w.toLocaleString('en-IN')}</span>
                  <span>₹{selectedStock.high_52w.toLocaleString('en-IN')}</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
