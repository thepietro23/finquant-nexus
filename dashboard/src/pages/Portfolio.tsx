import { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import { PieChart, AlertTriangle, TrendingUp, TrendingDown, X, Calculator, IndianRupee, ChevronDown, ChevronUp, RefreshCw, Database, Activity, Zap, Clock } from 'lucide-react';
import { MetricCardSkeleton, TableRowSkeleton, Skeleton } from '../components/ui/Skeleton';
import { toast } from '../lib/toast';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, AreaChart, Area,
  LineChart, Line, Legend, ReferenceLine, LabelList,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../lib/api';
import type {
  PortfolioSummaryResponse, StockDetailResponse, GrowthResponse,
  DataRefreshResponse, LivePortfolioResponse, OptimizedPortfolioResponse,
  SmartPortfolioResponse, OptimizedStock,
} from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import MetricCard from '../components/ui/MetricCard';
import type { MetricBadge } from '../components/ui/MetricCard';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import type { MetricDetail } from '../components/ui/MetricInfoPanel';
import { staggerContainer } from '../lib/animations';

// ── Semantic badge thresholds ──
function getMetricBadge(metric: string, value: number): MetricBadge {
  if (!isFinite(value)) return { label: 'N/A', variant: 'neutral' };
  switch (metric) {
    case 'sharpe': {
      if (value >= 1.5) return { label: 'EXCEPTIONAL', variant: 'profit' };
      if (value >= 1.0) return { label: 'EXCELLENT', variant: 'profit' };
      if (value >= 0.5) return { label: 'GOOD', variant: 'profit' };
      if (value >= 0)   return { label: 'AVERAGE', variant: 'warning' };
      return { label: 'POOR', variant: 'loss' };
    }
    case 'sortino': {
      if (value >= 3.0) return { label: 'EXCEPTIONAL', variant: 'profit' };
      if (value >= 2.0) return { label: 'EXCELLENT', variant: 'profit' };
      if (value >= 1.0) return { label: 'GOOD', variant: 'profit' };
      if (value >= 0)   return { label: 'AVERAGE', variant: 'warning' };
      return { label: 'POOR', variant: 'loss' };
    }
    case 'return': {
      if (value >= 25) return { label: 'EXCEPTIONAL', variant: 'profit' };
      if (value >= 18) return { label: 'EXCELLENT', variant: 'profit' };
      if (value >= 12) return { label: 'BEATS NIFTY', variant: 'profit' };
      if (value >= 7)  return { label: 'BEATS FD', variant: 'warning' };
      return { label: 'BELOW FD', variant: 'loss' };
    }
    case 'volatility': {
      if (value <= 10) return { label: 'LOW RISK', variant: 'profit' };
      if (value <= 15) return { label: 'MODERATE', variant: 'warning' };
      if (value <= 25) return { label: 'HIGH', variant: 'warning' };
      return { label: 'VERY HIGH', variant: 'loss' };
    }
    case 'drawdown': {
      const abs = Math.abs(value);
      if (abs <= 5)  return { label: 'EXCELLENT', variant: 'profit' };
      if (abs <= 10) return { label: 'CONTROLLED', variant: 'profit' };
      if (abs <= 20) return { label: 'MODERATE', variant: 'warning' };
      return { label: 'HIGH RISK', variant: 'loss' };
    }
    default: return { label: '', variant: 'neutral' };
  }
}

// ── Metric info definitions ──
function getMetricDetails(data: PortfolioSummaryResponse): Record<string, MetricDetail & { interpret: string }> {
  const s  = (v: number | null | undefined, d = 4) => safeNum(v).toFixed(d);
  return {
    'Sharpe Ratio': {
      what: 'Sharpe Ratio measures how much excess return you earn for each unit of risk (volatility). It is the gold standard of risk-adjusted performance metrics.',
      why: 'A 20% return sounds great — but if the volatility is 50%, you are riding a rollercoaster. Sharpe tells you the real truth: return per unit of risk.',
      how: 'Formula: (Portfolio Return − Risk-Free Rate) / Volatility. India risk-free rate = 7% (govt bonds). Annualized using 248 trading days.',
      good: '< 0 = losing money | 0–0.5 = average | 0.5–1.0 = good | 1.0–1.5 = very good | > 1.5 = exceptional (hedge fund level)',
      interpret: `Your portfolio Sharpe is ${s(data.sharpe_ratio)}. This means for every 1% of risk, you earn ${s(data.sharpe_ratio, 2)}% excess return above the risk-free rate. Computed from ${data.n_stocks} real NIFTY 50 stocks with equal-weight allocation over ${data.n_days} trading days.`,
    },
    'Sortino Ratio': {
      what: 'Sortino Ratio is like Sharpe but smarter — it only considers downside risk (losses). Upside volatility (gains) is not penalized because gains are desirable.',
      why: 'Sharpe penalizes both up and down movements equally. But going up is good! Sortino only penalizes downside moves, giving a more accurate picture of risk-adjusted returns.',
      how: 'Formula: (Portfolio Return − Risk-Free Rate) / Downside Deviation. Downside Deviation = standard deviation of only negative returns.',
      good: 'Sortino is always ≥ Sharpe (smaller denominator). > 1.0 = good | > 2.0 = excellent | > 3.0 = exceptional',
      interpret: `Sortino is ${s(data.sortino_ratio)} vs Sharpe ${s(data.sharpe_ratio)}. The gap means your upside volatility exceeds your downside — the portfolio gains more than it loses on average.`,
    },
    'Annual Return': {
      what: 'The annualized percentage return of the portfolio over the evaluation period. This is the most intuitive metric — "how much did I earn?"',
      why: 'Compare directly with alternatives: Fixed Deposit (7%), NIFTY 50 index (12-14%), or a good mutual fund (15-18%).',
      how: 'Take the mean daily return, then annualize: mean × √248 (geometric: (1 + mean_daily)^248 − 1). 248 = trading days per year in India.',
      good: '> 7% = beats FD | > 12% = competitive with NIFTY | > 18% = excellent | > 25% = exceptional',
      interpret: (() => {
        const ar = s(safeNum(data.annualized_return) * 100, 2);
        const vs_fd = (safeNum(data.annualized_return) * 100 - 7).toFixed(1);
        return `Portfolio delivered ${ar}% annualized return — that is ${vs_fd}% above the FD rate (7%). Computed from ${data.n_days} real trading days.`;
      })(),
    },
    'Volatility': {
      what: 'Annualized volatility measures how much the portfolio value fluctuates day-to-day. Higher volatility = more uncertainty = more risk.',
      why: 'High returns with high volatility means wild swings. Low volatility with decent returns is the sweet spot for consistent wealth building.',
      how: 'Standard deviation of daily returns × √248. Converts daily volatility to annual scale.',
      good: '< 10% = low risk | 10–15% = moderate | 15–25% = high | > 25% = very high risk',
      interpret: (() => {
        const vol = s(safeNum(data.annualized_volatility) * 100, 2);
        const daily = (safeNum(data.annualized_volatility) / Math.sqrt(248) * 100).toFixed(2);
        return `Portfolio volatility is ${vol}% annually (~${daily}% daily). This is in the moderate range — typical for a diversified NIFTY 50 equal-weight portfolio.`;
      })(),
    },
    'Max Drawdown': {
      what: 'Maximum Drawdown is the worst peak-to-trough decline during the evaluation period. It answers: "What is the maximum loss I would have experienced?"',
      why: 'Every investor fears crashes. Max Drawdown tells you the worst-case scenario actually observed in your portfolio — not hypothetical, but real.',
      how: 'Track cumulative portfolio value daily. Record the running peak. Drawdown = (current − peak) / peak. Max Drawdown = the largest such drop.',
      good: '> −5% = excellent | −5% to −10% = good | −10% to −20% = moderate | < −20% = concerning (NIFTY crashes: −30% to −40%)',
      interpret: (() => {
        const md = s(safeNum(data.max_drawdown) * 100, 2);
        return `Max drawdown was ${md}%. The portfolio dropped at most ${Math.abs(safeNum(data.max_drawdown) * 100).toFixed(1)}% from its peak. For reference, NIFTY 50 dropped 30-40% in 2008 and COVID crashes — this portfolio was much better controlled.`;
      })(),
    },
  };
}

const PAGE_INFO = {
  title: 'Portfolio Analysis — What Does This Page Show?',
  sections: [
    {
      heading: 'What is this page?',
      text: 'Core output of FINQUANT-NEXUS v4. Shows a 47-stock NIFTY 50 portfolio built on real Yahoo Finance data (2015–2025, stored in CSV). Three portfolio modes are available: Equal-Weight (baseline, every stock gets 1/47), Max Sharpe (SLSQP-optimized weights), and Smart Portfolio (AI-multi-signal blend).',
    },
    {
      heading: 'What are the 5 metric cards?',
      text: 'Sharpe Ratio (return ÷ risk — higher is better, >1.0 is excellent), Sortino Ratio (same but only downside risk counts), Annualized Return (yearly average %), Volatility (how much returns fluctuate), Max Drawdown (worst peak-to-trough loss %). Click any card for a full explanation with good/bad benchmarks.',
    },
    {
      heading: 'Smart Optimize — the main research output',
      text: 'Smart Portfolio blends three AI signals: RL momentum weights (40% from RL Agent tab) + FinBERT news sentiment weights (40% from Sentiment tab) + Federated Learning sector allocation (20% from Federated tab) → then runs SLSQP Max Sharpe optimization. This multi-signal approach is the dissertation\'s core contribution.',
    },
    {
      heading: 'Investment Simulator & Portfolio Growth',
      text: 'Enter any ₹ amount → see exact rupee profit per stock at equal-weight. Portfolio Growth tab shows how ₹1 invested on any historical date grows under: (1) our portfolio, (2) NIFTY 50 index, (3) Fixed Deposit at 7% p.a. — a like-for-like comparison proving AI beats passive investing.',
    },
    {
      heading: 'Holdings table & Sector Weights',
      text: 'All stocks with ticker, sector, equal weight (%), and cumulative return since CSV start date. Sectors with more NIFTY 50 stocks naturally get higher equal-weight (Energy+Auto ~15 stocks ≈ 32%). Holdings sorted by cumulative return descending — best performers at top.',
    },
    {
      heading: 'Where does the data come from?',
      text: 'FastAPI backend reads data/all_close_prices.csv (2015–2025, 47 tickers, downloaded via Yahoo Finance + yfinance library). All metrics computed live — no hardcoded values. Risk-free rate = 5% (India historical average for 2015–2021 backtest period).',
    },
  ],
};

// ── Investment Simulator types ──
interface SimStock {
  ticker: string; sector: string; weight: number;
  invested: number; currentValue: number; profit: number; returnPct: number;
}
interface SimResult {
  totalInvested: number; totalValue: number; totalProfit: number;
  totalReturnPct: number; perStock: SimStock[];
  bySector: Record<string, { invested: number; value: number; profit: number }>;
}

function safeNum(v: number | null | undefined, fallback = 0): number {
  return (v == null || !isFinite(v)) ? fallback : v;
}

function runSimulation(amount: number, holdings: PortfolioSummaryResponse['holdings']): SimResult {
  const perStock: SimStock[] = holdings.map(h => {
    const ret = safeNum(h.cumulative_return);
    const wt  = safeNum(h.weight);
    const invested = amount * (wt / 100);
    const currentValue = invested * (1 + ret / 100);
    const profit = currentValue - invested;
    return { ticker: h.ticker.replace('.NS', ''), sector: h.sector, weight: wt, invested, currentValue, profit, returnPct: ret };
  });
  const totalInvested = amount;
  const totalValue = perStock.reduce((s, x) => s + x.currentValue, 0);
  const totalProfit = totalValue - totalInvested;
  const totalReturnPct = (totalProfit / totalInvested) * 100;
  const bySector: SimResult['bySector'] = {};
  for (const s of perStock) {
    if (!bySector[s.sector]) bySector[s.sector] = { invested: 0, value: 0, profit: 0 };
    bySector[s.sector].invested += s.invested;
    bySector[s.sector].value += s.currentValue;
    bySector[s.sector].profit += s.profit;
  }
  return { totalInvested, totalValue, totalProfit, totalReturnPct, perStock, bySector };
}

const fmt = (n: number) => isFinite(n) ? '₹' + Math.round(n).toLocaleString('en-IN') : '₹0';

const COLORS = ['#C15F3C', '#6366F1', '#0D9488', '#F59E0B', '#EC4899', '#8B5CF6', '#10B981', '#F97316', '#06B6D4', '#EF4444', '#84CC16'];

export default function Portfolio() {
  const [data, setData] = useState<PortfolioSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [selectedStock, setSelectedStock] = useState<StockDetailResponse | null>(null);
  const [stockLoading, setStockLoading] = useState(false);

  const [refreshing, setRefreshing] = useState(false);
  const [refreshStatus, setRefreshStatus] = useState<DataRefreshResponse | null>(null);

  // Unified controls — one set of inputs drives everything
  const [rangeStart, setRangeStart] = useState('');
  const [rangeEnd,   setRangeEnd]   = useState('');
  const [startingCapital, setStartingCapital] = useState(1_000_000);
  const [capInput, setCapInput] = useState('10,00,000');
  const [rangeLoading, setRangeLoading] = useState(false);

  // Growth chart + breakdown (auto-populated on Apply)
  const [growthResult, setGrowthResult] = useState<GrowthResponse | null>(null);
  const [growthLoading, setGrowthLoading] = useState(false);
  const [showBreakdown, setShowBreakdown] = useState(false);
  const [simStockSort, setSimStockSort] = useState<'profit' | 'loss' | 'weight'>('profit');
  const growthTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Live portfolio — real-time intraday polling
  const [liveData, setLiveData]       = useState<LivePortfolioResponse | null>(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [countdown, setCountdown]     = useState(0);
  const liveIntervalRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const LIVE_POLL_SECS = 300; // 5 min — matches server-side cache TTL to avoid rate limits

  // Portfolio optimization — Max Sharpe via SLSQP
  const [optimized, setOptimized]           = useState<OptimizedPortfolioResponse | null>(null);
  const [optimizedLoading, setOptimizedLoading] = useState(false);
  const [showOptimized, setShowOptimized]   = useState(false);

  // Smart portfolio — RL + Sentiment + FL blended optimization
  const [smart, setSmart]           = useState<SmartPortfolioResponse | null>(null);
  const [smartLoading, setSmartLoading] = useState(false);
  const [showSmart, setShowSmart]   = useState(false);

  // Active portfolio mode — drives metric cards, holdings table, simulation
  const [activeMode, setActiveMode] = useState<'equal' | 'optimized' | 'smart'>('equal');

  // All useMemo hooks must be above early returns — hooks must run same order every render

  // Build a holdings array with optimized weights but original per-stock returns
  const displayHoldings = useMemo(() => {
    const makeHoldings = (weights: OptimizedStock[]) => {
      if (!data) return [];
      const returnMap: Record<string, { daily: number; cum: number }> = {};
      for (const h of data.holdings)
        returnMap[h.ticker.replace('.NS', '')] = { daily: h.daily_return, cum: h.cumulative_return };
      return weights.map(w => ({
        ticker: w.ticker,
        sector: w.sector,
        weight: w.optimized_weight,
        daily_return: returnMap[w.ticker]?.daily ?? 0,
        cumulative_return: returnMap[w.ticker]?.cum ?? 0,
      }));
    };
    if (activeMode === 'optimized' && optimized) return makeHoldings(optimized.weights);
    if (activeMode === 'smart'     && smart)     return makeHoldings(smart.weights);
    return data?.holdings ?? [];
  }, [activeMode, optimized, smart, data]);

  // Metrics for the 5 summary cards
  const displayMetrics = useMemo(() => {
    if (activeMode === 'optimized' && optimized) return {
      sharpe_ratio:          safeNum(optimized.optimized_sharpe),
      sortino_ratio:         safeNum(optimized.optimized_sortino),
      annualized_return:     safeNum(optimized.optimized_return)     / 100,
      annualized_volatility: safeNum(optimized.optimized_volatility) / 100,
      max_drawdown:          safeNum(optimized.optimized_drawdown)   / 100,
    };
    if (activeMode === 'smart' && smart) return {
      sharpe_ratio:          safeNum(smart.smart_sharpe),
      sortino_ratio:         safeNum(smart.smart_sortino),
      annualized_return:     safeNum(smart.smart_return)     / 100,
      annualized_volatility: safeNum(smart.smart_volatility) / 100,
      max_drawdown:          safeNum(smart.smart_drawdown)   / 100,
    };
    return data ? {
      sharpe_ratio:          safeNum(data.sharpe_ratio),
      sortino_ratio:         safeNum(data.sortino_ratio),
      annualized_return:     safeNum(data.annualized_return),
      annualized_volatility: safeNum(data.annualized_volatility),
      max_drawdown:          safeNum(data.max_drawdown),
    } : null;
  }, [activeMode, optimized, smart, data]);

  // Sector weights for the bar chart
  const displaySectorWeights = useMemo(() => {
    const activeWeights = activeMode === 'optimized' ? optimized?.weights
      : activeMode === 'smart' ? smart?.weights : null;
    if (activeWeights) {
      const map: Record<string, number> = {};
      for (const w of activeWeights) map[w.sector] = (map[w.sector] ?? 0) + w.optimized_weight;
      return map;
    }
    return data?.sector_weights ?? {};
  }, [activeMode, optimized, smart, data]);

  const simResult: SimResult | null = useMemo(
    () => (data ? runSimulation(startingCapital, displayHoldings) : null),
    [data, startingCapital, displayHoldings],
  );

  const metricDetails = useMemo(() => (data ? getMetricDetails(data) : {}), [data]);

  const allocData = useMemo(
    () => Object.entries(displaySectorWeights)
        .sort((a, b) => b[1] - a[1])
        .map(([sector, weight]) => ({ sector, weight: Math.round(weight) })),
    [displaySectorWeights],
  );

  const sectorColorMap = useMemo(
    () => Object.fromEntries(allocData.map((d, i) => [d.sector, COLORS[i % COLORS.length]])),
    [allocData],
  );

  const maxAbsReturn = useMemo(
    () => Math.max(...displayHoldings.map(h => Math.abs(safeNum(h.cumulative_return))), 1),
    [displayHoldings],
  );

  function applyCapital(raw: string) {
    const num = parseInt(raw.replace(/[^0-9]/g, ''), 10);
    if (!isNaN(num) && num >= 1000) {
      setStartingCapital(num);
      setCapInput(num.toLocaleString('en-IN'));
    }
  }

  const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
  function fetchRange(s: string, e: string) {
    if (!s || !e || !ISO_DATE.test(s) || !ISO_DATE.test(e) || s >= e) {
      toast.warning('Select a valid date range (From < To)');
      return;
    }
    setRangeLoading(true);
    setGrowthResult(null);
    setRangeStart(s);
    setRangeEnd(e);
    // Clear stale optimization results — they were computed for a different date range
    setActiveMode('equal');
    setOptimized(null);
    setSmart(null);
    setShowOptimized(false);
    setShowSmart(false);

    if (growthTimeoutRef.current) clearTimeout(growthTimeoutRef.current);
    setGrowthLoading(true);
    growthTimeoutRef.current = setTimeout(() => {
      setGrowthLoading(false);
      toast.error('Growth chart timed out — backend may be slow');
    }, 40_000);

    Promise.all([
      api.portfolioSummary({ start_date: s, end_date: e }),
      api.portfolioGrowth(startingCapital, s, e),
    ])
      .then(([summary, growth]) => {
        setData(summary);
        setGrowthResult(growth);
        clearTimeout(growthTimeoutRef.current!);
      })
      .catch(err => toast.error(err instanceof Error ? err.message : 'Failed to load data'))
      .finally(() => { setRangeLoading(false); setGrowthLoading(false); });
  }

  function handleRefresh() {
    setRefreshing(true);
    api.refreshData()
      .then(r => {
        setRefreshStatus(r);
        if (r.status === 'updated') {
          const params = rangeStart && rangeEnd ? { start_date: rangeStart, end_date: rangeEnd } : undefined;
          return api.portfolioSummary(params).then(d => setData(d));
        }
      })
      .catch(e => toast.error(e instanceof Error ? e.message : 'Refresh failed'))
      .finally(() => setRefreshing(false));
  }

  function handleStockClick(ticker: string) {
    if (selectedStock?.ticker === ticker.replace('.NS', '')) {
      setSelectedStock(null);
      return;
    }
    setStockLoading(true);
    api.stockDetail(ticker.replace('.NS', ''))
      .then(d => { setSelectedStock(d); setStockLoading(false); })
      .catch(e => {
        setStockLoading(false);
        toast.error(e instanceof Error ? e.message : 'Failed to load stock details');
      });
  }

  useEffect(() => {
    api.portfolioSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to load portfolio data');
        setLoading(false);
      });
  }, []);

  // Live portfolio polling — fetch immediately, then every LIVE_POLL_SECS seconds
  const fetchLive = useCallback(() => {
    setLiveLoading(true);
    api.livePortfolio()
      .then(d => { setLiveData(d); setCountdown(LIVE_POLL_SECS); })
      .catch(() => { /* silent fail — live is best-effort */ })
      .finally(() => setLiveLoading(false));
  }, []);

  useEffect(() => {
    fetchLive();
    liveIntervalRef.current = setInterval(fetchLive, LIVE_POLL_SECS * 1000);
    countdownIntervalRef.current = setInterval(
      () => setCountdown(c => (c > 0 ? c - 1 : 0)),
      1000,
    );
    return () => {
      if (liveIntervalRef.current)      clearInterval(liveIntervalRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    };
  }, [fetchLive]);

  function handleOptimize() {
    setOptimizedLoading(true);
    setShowOptimized(true);
    const params = rangeStart && rangeEnd ? { start_date: rangeStart, end_date: rangeEnd } : undefined;
    api.portfolioOptimized(params)
      .then(d => { setOptimized(d); setActiveMode('optimized'); })
      .catch(e => {
        toast.error(e instanceof Error ? e.message : 'Optimization failed');
        setShowOptimized(false);
      })
      .finally(() => setOptimizedLoading(false));
  }

  function handleSmartOptimize() {
    setSmartLoading(true);
    setShowSmart(true);
    const params = rangeStart && rangeEnd ? { start_date: rangeStart, end_date: rangeEnd } : undefined;
    api.portfolioSmart(params)
      .then(d => { setSmart(d); setActiveMode('smart'); })
      .catch(e => {
        toast.error(e instanceof Error ? e.message : 'Smart optimization failed');
        setShowSmart(false);
      })
      .finally(() => setSmartLoading(false));
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64 mb-2" rounded="lg" />
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => <MetricCardSkeleton key={i} />)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Skeleton className="h-64" rounded="xl" />
          <Skeleton className="h-64" rounded="xl" />
        </div>
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => <TableRowSkeleton key={i} cols={4} />)}
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

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Portfolio Analysis"
          subtitle="NIFTY 50 portfolio — AI-optimized allocation with risk analysis"
          icon={<PieChart size={24} />}
        />
        <div className="flex items-center gap-3 shrink-0">
          {/* Live intraday badge */}
          {liveData && (
            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium ${
              liveData.is_market_open
                ? 'bg-profit/8 border-profit/30 text-profit'
                : 'bg-surface border-border text-text-muted'
            }`}>
              <Activity size={12} className={liveData.is_market_open ? 'animate-pulse' : ''} />
              <span className={liveData.portfolio_change_pct >= 0 ? 'text-profit' : 'text-loss'}>
                {liveData.portfolio_change_pct >= 0 ? '+' : ''}{liveData.portfolio_change_pct.toFixed(2)}% today
              </span>
              <span className="text-text-muted">·</span>
              <Clock size={10} className="text-text-muted" />
              <span className="text-text-muted tabular-nums">{countdown}s</span>
            </div>
          )}
          {liveLoading && !liveData && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface border border-border text-xs text-text-muted">
              <Activity size={12} className="animate-pulse" />
              <span>Fetching live…</span>
            </div>
          )}
          {/* Data freshness badge */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface border border-border text-xs text-text-muted">
            <Database size={12} />
            <span>
              Today: <span className="text-text font-medium">
                {new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
              </span>
              <span className="mx-1.5 opacity-40">·</span>
              Last close: <span className="text-text font-medium">{data.data_as_of || data.date_end}</span>
            </span>
          </div>
          {/* Refresh button */}
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            title={refreshing ? 'Downloading latest prices…' : 'Fetch latest prices from Yahoo Finance'}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface border border-border text-xs text-text-muted hover:text-secondary hover:border-secondary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw size={12} className={refreshing ? 'animate-spin' : ''} />
            <span>{refreshing ? 'Refreshing…' : refreshStatus?.status === 'updated' ? `+${refreshStatus.added_rows} rows` : 'Refresh Data'}</span>
          </button>
          <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
        </div>
      </div>

      {/* ── Control Bar: Date Range + Starting Capital ── */}
      <div className="flex flex-wrap items-end gap-3 my-4 p-4 rounded-xl bg-surface border border-border">
        {/* Date Range */}
        <div className="flex items-end gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-text-muted font-medium">From</label>
            <input
              type="date"
              value={rangeStart || data.date_start}
              min={data.csv_date_start || '2018-01-01'}
              max={rangeEnd || data.date_end}
              onChange={e => setRangeStart(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg border border-border bg-background text-text focus:outline-none focus:border-secondary"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-text-muted font-medium">To</label>
            <input
              type="date"
              value={rangeEnd || data.date_end}
              min={rangeStart || data.csv_date_start || '2018-01-01'}
              max={data.date_end}
              onChange={e => setRangeEnd(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg border border-border bg-background text-text focus:outline-none focus:border-secondary"
            />
          </div>
          <button
            onClick={() => fetchRange(rangeStart || data.date_start, rangeEnd || data.date_end)}
            disabled={rangeLoading}
            className="px-3 py-1.5 text-xs rounded-lg bg-secondary text-white font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {rangeLoading ? 'Loading…' : 'Apply'}
          </button>
          <button
            onClick={() => {
              setRangeStart('');
              setRangeEnd('');
              setGrowthResult(null);
              setActiveMode('equal');
              setOptimized(null);
              setSmart(null);
              setShowOptimized(false);
              setShowSmart(false);
              api.portfolioSummary().then(d => setData(d));
            }}
            className="px-3 py-1.5 text-xs rounded-lg border border-border text-text-muted hover:text-text transition-colors"
          >
            Reset
          </button>
        </div>

        <div className="w-px h-8 bg-border hidden lg:block" />

        {/* Starting Capital */}
        <div className="flex items-end gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-text-muted font-medium">Starting Capital</label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs text-text-muted">₹</span>
              <input
                type="text"
                value={capInput}
                onChange={e => setCapInput(e.target.value)}
                onBlur={() => applyCapital(capInput)}
                onKeyDown={e => e.key === 'Enter' && applyCapital(capInput)}
                className="pl-6 pr-3 py-1.5 w-36 text-xs rounded-lg border border-border bg-background text-text focus:outline-none focus:border-secondary font-mono"
              />
            </div>
          </div>
          {/* Preset buttons */}
          <div className="flex gap-1">
            {[['1L', 100_000], ['10L', 1_000_000], ['50L', 5_000_000], ['1Cr', 10_000_000], ['5Cr', 50_000_000]].map(([label, val]) => (
              <button
                key={label}
                onClick={() => { setStartingCapital(val as number); setCapInput((val as number).toLocaleString('en-IN')); }}
                className={`px-2 py-1.5 text-xs rounded-lg border transition-colors ${startingCapital === val ? 'bg-secondary text-white border-secondary' : 'border-border text-text-muted hover:text-text'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="w-px h-8 bg-border hidden lg:block" />

        {/* Live portfolio value — always derived from simResult so it matches the simulation cards */}
        {(() => {
          const ret  = safeNum(simResult ? simResult.totalReturnPct : (data.total_return_pct ?? 0));
          const val  = safeNum(simResult ? simResult.totalValue     : startingCapital * (1 + ret / 100));
          return (
            <div className="flex flex-col gap-0.5 ml-auto">
              <span className="text-xs text-text-muted">Portfolio Value</span>
              <span className="text-lg font-bold font-mono text-secondary">
                {fmt(val)}
              </span>
              <span className={`text-xs font-mono ${ret >= 0 ? 'text-profit' : 'text-loss'}`}>
                {ret >= 0 ? '+' : ''}{ret.toFixed(2)}% over period
              </span>
            </div>
          );
        })()}
      </div>

      {/* ── Active Mode Banner ── */}
      {activeMode !== 'equal' && displayMetrics && (
        <div className={`flex items-center justify-between px-4 py-2.5 rounded-xl border mb-3 text-sm font-medium ${
          activeMode === 'optimized'
            ? 'bg-profit/6 border-profit/30 text-profit'
            : 'bg-violet-50 border-violet-200 text-violet-700'
        }`}>
          <div className="flex items-center gap-2">
            <Zap size={14} />
            <span>
              {activeMode === 'optimized'
                ? 'Viewing: Max Sharpe Optimized Portfolio — metrics, simulation & holdings updated'
                : 'Viewing: Smart Portfolio (RL + Sentiment + FL) — metrics, simulation & holdings updated'}
            </span>
          </div>
          <button
            onClick={() => setActiveMode('equal')}
            className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg border border-current/30 hover:bg-white/50 transition-colors"
          >
            <X size={11} /> Reset to Equal Weight
          </button>
        </div>
      )}

      {/* ── Clickable Metric Cards ── */}
      {displayMetrics && (
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-2">
        <MetricCard title="Sharpe Ratio" value={displayMetrics.sharpe_ratio} decimals={4}
          badge={getMetricBadge('sharpe', displayMetrics.sharpe_ratio)}
          onClick={() => setExpandedMetric(m => m === 'Sharpe Ratio' ? null : 'Sharpe Ratio')}
          active={expandedMetric === 'Sharpe Ratio'} />
        <MetricCard title="Sortino Ratio" value={displayMetrics.sortino_ratio} decimals={4}
          badge={getMetricBadge('sortino', displayMetrics.sortino_ratio)}
          onClick={() => setExpandedMetric(m => m === 'Sortino Ratio' ? null : 'Sortino Ratio')}
          active={expandedMetric === 'Sortino Ratio'} />
        <MetricCard title="Annual Return" value={displayMetrics.annualized_return * 100} decimals={2} suffix="%"
          badge={getMetricBadge('return', displayMetrics.annualized_return * 100)}
          onClick={() => setExpandedMetric(m => m === 'Annual Return' ? null : 'Annual Return')}
          active={expandedMetric === 'Annual Return'} />
        <MetricCard title="Volatility" value={displayMetrics.annualized_volatility * 100} decimals={2} suffix="%"
          badge={getMetricBadge('volatility', displayMetrics.annualized_volatility * 100)}
          onClick={() => setExpandedMetric(m => m === 'Volatility' ? null : 'Volatility')}
          active={expandedMetric === 'Volatility'} />
        <MetricCard title="Max Drawdown" value={displayMetrics.max_drawdown * 100} decimals={2} suffix="%"
          badge={getMetricBadge('drawdown', displayMetrics.max_drawdown * 100)}
          onClick={() => setExpandedMetric(m => m === 'Max Drawdown' ? null : 'Max Drawdown')}
          active={expandedMetric === 'Max Drawdown'} />
      </motion.div>
      )}

      {/* ── Expanded Metric Info ── */}
      <MetricInfoPanel
        expandedMetric={expandedMetric}
        onClose={() => setExpandedMetric(null)}
        details={metricDetails}
      />

      {/* ── Portfolio Optimization buttons ── */}
      <div className="mt-4 mb-2 flex flex-wrap gap-2">
        <button
          onClick={handleOptimize}
          disabled={optimizedLoading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-primary to-secondary text-white text-sm font-semibold shadow hover:opacity-90 disabled:opacity-60 transition-opacity"
        >
          <Zap size={15} />
          {optimizedLoading ? 'Optimizing…' : 'Optimize Portfolio (Max Sharpe)'}
        </button>
        <button
          onClick={handleSmartOptimize}
          disabled={smartLoading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-500 text-white text-sm font-semibold shadow hover:opacity-90 disabled:opacity-60 transition-opacity"
        >
          <Zap size={15} />
          {smartLoading ? 'Running signals…' : 'Smart Optimize (RL + Sentiment + FL)'}
        </button>
      </div>

      <AnimatePresence>
        {showOptimized && (
          <motion.div
            initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ type: 'spring', stiffness: 200, damping: 26 }}
            className="overflow-hidden mb-6"
          >
            {optimizedLoading && (
              <div className="flex items-center gap-2 py-6 text-sm text-text-muted justify-center">
                <svg className="animate-spin h-4 w-4 text-primary" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                Running SLSQP optimizer on {data.n_stocks} stocks…
              </div>
            )}

            {optimized && !optimizedLoading && (
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-display font-bold text-lg text-secondary flex items-center gap-2">
                    <Zap size={18} className="text-primary" /> Optimization Results
                    <span className="text-xs font-normal text-text-muted ml-1">({optimized.method})</span>
                  </h2>
                  <button onClick={() => setShowOptimized(false)} className="p-1 rounded-lg hover:bg-bg-card text-text-muted">
                    <X size={16} />
                  </button>
                </div>

                {/* Side-by-side metric comparison */}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
                  {[
                    { label: 'Sharpe Ratio',   eq: optimized.equal_sharpe,      opt: optimized.optimized_sharpe,     hi: true,  fmt: (v: number) => safeNum(v).toFixed(4) },
                    { label: 'Sortino Ratio',  eq: optimized.equal_sortino,     opt: optimized.optimized_sortino,    hi: true,  fmt: (v: number) => safeNum(v).toFixed(4) },
                    { label: 'Annual Return',  eq: optimized.equal_return,      opt: optimized.optimized_return,     hi: true,  fmt: (v: number) => `${safeNum(v).toFixed(2)}%` },
                    { label: 'Volatility',     eq: optimized.equal_volatility,  opt: optimized.optimized_volatility, hi: false, fmt: (v: number) => `${safeNum(v).toFixed(2)}%` },
                    { label: 'Max Drawdown',   eq: optimized.equal_drawdown,    opt: optimized.optimized_drawdown,   hi: false, fmt: (v: number) => `${safeNum(v).toFixed(2)}%` },
                    { label: 'Sharpe Gain',    eq: null,                        opt: optimized.sharpe_improvement,   hi: true,  fmt: (v: number) => `+${safeNum(v).toFixed(4)}` },
                  ].map(m => {
                    const better = m.hi
                      ? (m.opt > (m.eq ?? 0))
                      : (m.opt < (m.eq ?? 0));
                    return (
                      <div key={m.label} className={`rounded-xl px-3 py-2.5 border ${better ? 'bg-profit/5 border-profit/25' : 'bg-bg-card border-border-light'}`}>
                        <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">{m.label}</p>
                        {m.eq !== null && (
                          <p className="text-[10px] text-text-muted font-mono line-through mb-0.5">
                            {m.fmt(m.eq)} <span className="no-underline text-[9px]">equal</span>
                          </p>
                        )}
                        <p className={`font-mono font-bold text-sm ${better ? 'text-profit' : 'text-loss'}`}>
                          {m.fmt(m.opt)}
                        </p>
                      </div>
                    );
                  })}
                </div>

                {/* Top-weight stocks table */}
                <p className="text-xs font-medium text-text-secondary mb-2">
                  Top stocks by optimized weight (max 15%, min 0.5%)
                </p>
                <div className="max-h-64 overflow-y-auto rounded-xl border border-border-light">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-white border-b border-border">
                      <tr>
                        <th className="text-left py-2 px-3 font-medium text-text-secondary text-xs">#</th>
                        <th className="text-left py-2 px-3 font-medium text-text-secondary text-xs">Ticker</th>
                        <th className="text-left py-2 px-3 font-medium text-text-secondary text-xs">Sector</th>
                        <th className="text-right py-2 px-3 font-medium text-text-secondary text-xs">Equal Wt</th>
                        <th className="text-right py-2 px-3 font-medium text-text-secondary text-xs">Opt Wt</th>
                        <th className="text-right py-2 px-3 font-medium text-text-secondary text-xs">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {optimized.weights.slice(0, 20).map((w, i) => {
                        const diff = w.optimized_weight - w.equal_weight;
                        return (
                          <tr key={w.ticker} className="border-b border-border-light hover:bg-bg-card transition-colors">
                            <td className="py-1.5 px-3 text-text-muted font-mono text-xs">{i + 1}</td>
                            <td className="py-1.5 px-3 font-mono font-semibold text-text text-xs">{w.ticker}</td>
                            <td className="py-1.5 px-3 text-text-secondary text-xs">{w.sector}</td>
                            <td className="py-1.5 px-3 text-right font-mono text-xs text-text-muted">{safeNum(w.equal_weight).toFixed(2)}%</td>
                            <td className="py-1.5 px-3 text-right font-mono font-semibold text-xs text-secondary">{safeNum(w.optimized_weight).toFixed(2)}%</td>
                            <td className={`py-1.5 px-3 text-right font-mono text-xs ${diff >= 0 ? 'text-profit' : 'text-loss'}`}>
                              {diff >= 0 ? '+' : ''}{safeNum(diff).toFixed(2)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-text-muted mt-2">
                  These weights maximise the Sharpe ratio on historical data ({data.n_days} trading days).
                  Past optimization does not guarantee future outperformance.
                </p>
              </Card>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Smart Portfolio results panel ── */}
      <AnimatePresence>
        {showSmart && (
          <motion.div
            initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ type: 'spring', stiffness: 200, damping: 26 }}
            className="overflow-hidden mb-6"
          >
            {smartLoading && (
              <div className="flex items-center gap-2 py-6 text-sm text-text-muted justify-center">
                <svg className="animate-spin h-4 w-4 text-violet-500" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                Blending RL momentum + Sentiment + FL sector signals…
              </div>
            )}

            {smart && !smartLoading && (
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-display font-bold text-lg flex items-center gap-2" style={{ color: '#7c3aed' }}>
                    <Zap size={18} style={{ color: '#a855f7' }} /> Smart Portfolio
                    <span className="text-xs font-normal text-text-muted ml-1">({smart.method})</span>
                  </h2>
                  <button onClick={() => setShowSmart(false)} className="p-1 rounded-lg hover:bg-bg-card text-text-muted">
                    <X size={16} />
                  </button>
                </div>

                {/* Signal breakdown row */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mb-5 p-3 rounded-xl bg-violet-50 border border-violet-100">
                  {[
                    { label: 'RL Signal',        value: smart.signals.rl_sharpe,        color: '#6366f1' },
                    { label: 'Sentiment Signal',  value: smart.signals.sentiment_sharpe,  color: '#f59e0b' },
                    { label: 'FL Signal',         value: smart.signals.fl_sharpe,         color: '#0d9488' },
                    { label: 'Blended Prior',     value: smart.signals.blended_sharpe,    color: '#8b5cf6' },
                    { label: 'Final Sharpe',      value: smart.signals.final_sharpe,      color: '#7c3aed' },
                  ].map(s => (
                    <div key={s.label} className="flex flex-col items-center gap-0.5">
                      <span className="text-[10px] text-text-muted font-medium text-center">{s.label}</span>
                      <span className="font-mono font-bold text-sm" style={{ color: s.color }}>
                        {s.value.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Side-by-side metric comparison */}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
                  {[
                    { label: 'Sharpe Ratio',  eq: smart.equal_sharpe,     opt: smart.smart_sharpe,     hi: true,  fmt: (v: number) => safeNum(v).toFixed(4) },
                    { label: 'Sortino Ratio', eq: smart.equal_sortino,    opt: smart.smart_sortino,    hi: true,  fmt: (v: number) => safeNum(v).toFixed(4) },
                    { label: 'Annual Return', eq: smart.equal_return,     opt: smart.smart_return,     hi: true,  fmt: (v: number) => `${safeNum(v).toFixed(2)}%` },
                    { label: 'Volatility',    eq: smart.equal_volatility, opt: smart.smart_volatility, hi: false, fmt: (v: number) => `${safeNum(v).toFixed(2)}%` },
                    { label: 'Max Drawdown',  eq: smart.equal_drawdown,   opt: smart.smart_drawdown,   hi: false, fmt: (v: number) => `${safeNum(v).toFixed(2)}%` },
                    { label: 'Sharpe Gain',   eq: null,                   opt: smart.sharpe_improvement, hi: true, fmt: (v: number) => `+${safeNum(v).toFixed(4)}` },
                  ].map(m => {
                    const better = m.hi ? (m.opt > (m.eq ?? 0)) : (m.opt < (m.eq ?? 0));
                    return (
                      <div key={m.label} className={`rounded-xl px-3 py-2.5 border ${better ? 'bg-profit/5 border-profit/25' : 'bg-bg-card border-border-light'}`}>
                        <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">{m.label}</p>
                        {m.eq !== null && (
                          <p className="text-[10px] text-text-muted font-mono line-through mb-0.5">
                            {m.fmt(m.eq)} <span className="no-underline text-[9px]">equal</span>
                          </p>
                        )}
                        <p className={`font-mono font-bold text-sm ${better ? 'text-profit' : 'text-loss'}`}>
                          {m.fmt(m.opt)}
                        </p>
                      </div>
                    );
                  })}
                </div>

                {/* Weights table */}
                <p className="text-xs font-medium text-text-secondary mb-2">
                  Top stocks by smart weight (RL momentum 40% · Sentiment 40% · FL sector 20% → SLSQP)
                </p>
                <div className="max-h-64 overflow-y-auto rounded-xl border border-border-light">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-white border-b border-border">
                      <tr>
                        <th className="text-left py-2 px-3 font-medium text-text-secondary text-xs">#</th>
                        <th className="text-left py-2 px-3 font-medium text-text-secondary text-xs">Ticker</th>
                        <th className="text-left py-2 px-3 font-medium text-text-secondary text-xs">Sector</th>
                        <th className="text-right py-2 px-3 font-medium text-text-secondary text-xs">Equal Wt</th>
                        <th className="text-right py-2 px-3 font-medium text-text-secondary text-xs">Smart Wt</th>
                        <th className="text-right py-2 px-3 font-medium text-text-secondary text-xs">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {smart.weights.slice(0, 20).map((w, i) => {
                        const diff = w.optimized_weight - w.equal_weight;
                        return (
                          <tr key={w.ticker} className="border-b border-border-light hover:bg-bg-card transition-colors">
                            <td className="py-1.5 px-3 text-text-muted font-mono text-xs">{i + 1}</td>
                            <td className="py-1.5 px-3 font-mono font-semibold text-text text-xs">{w.ticker}</td>
                            <td className="py-1.5 px-3 text-text-secondary text-xs">{w.sector}</td>
                            <td className="py-1.5 px-3 text-right font-mono text-xs text-text-muted">{safeNum(w.equal_weight).toFixed(2)}%</td>
                            <td className="py-1.5 px-3 text-right font-mono font-semibold text-xs" style={{ color: '#7c3aed' }}>{safeNum(w.optimized_weight).toFixed(2)}%</td>
                            <td className={`py-1.5 px-3 text-right font-mono text-xs ${diff >= 0 ? 'text-profit' : 'text-loss'}`}>
                              {diff >= 0 ? '+' : ''}{safeNum(diff).toFixed(2)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-text-muted mt-2">
                  Signal weights: RL momentum uses rolling 30-day Sharpe softmax across all agents. Sentiment uses FinBERT-adjusted holdings from the Sentiment tab. FL uses per-sector Sharpe as client quality proxy. All three are blended then SLSQP-optimised.
                </p>
              </Card>
            )}
          </motion.div>
        )}
      </AnimatePresence>

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
                <LabelList
                  dataKey="weight"
                  position="right"
                  formatter={(v: unknown) => `${v}%`}
                  style={{ fontSize: 11, fill: '#6B7280', fontFamily: 'JetBrains Mono, monospace' }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Stock List */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-4">
            All Holdings ({displayHoldings.length})
            <span className="text-xs font-normal text-text-muted ml-2">
              {activeMode === 'equal' ? 'Click any stock for details' : `${activeMode === 'optimized' ? 'Max Sharpe' : 'Smart'} weights applied`}
            </span>
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
                {displayHoldings.map((h, i) => (
                  <tr key={h.ticker}
                    onClick={() => handleStockClick(h.ticker)}
                    className={`border-b border-border-light cursor-pointer transition-all ${
                      selectedStock?.ticker === h.ticker.replace('.NS', '')
                        ? 'bg-primary-subtle border-l-[3px] border-l-primary'
                        : 'hover:bg-bg-card hover:shadow-[0_2px_8px_rgba(193,95,60,0.06)]'
                    }`}>
                    <td className="py-2 text-text-muted font-mono text-xs">{i + 1}</td>
                    <td className="py-2 font-mono font-semibold text-text">{h.ticker.replace('.NS', '')}</td>
                    <td className="py-2">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ background: sectorColorMap[h.sector] ?? '#9CA3AF' }}
                        />
                        <span className="text-text-secondary text-xs truncate max-w-[90px]">{h.sector}</span>
                      </div>
                    </td>
                    <td className="py-2 text-right font-mono text-xs">{safeNum(h.weight).toFixed(1)}%</td>
                    <td className="py-2 text-right relative">
                      {/* Return bar fill behind text */}
                      <div
                        className="absolute inset-y-0.5 right-0 rounded-l opacity-[0.13]"
                        style={{
                          width: `${(Math.abs(safeNum(h.cumulative_return)) / maxAbsReturn) * 100}%`,
                          background: safeNum(h.cumulative_return) >= 0 ? '#16A34A' : '#DC2626',
                        }}
                      />
                      <span className={`relative z-10 font-mono font-semibold text-xs ${
                        safeNum(h.cumulative_return) >= 0 ? 'text-profit' : 'text-loss'
                      }`}>
                        {safeNum(h.cumulative_return) > 0 ? '+' : ''}{safeNum(h.cumulative_return).toFixed(1)}%
                      </span>
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
            <Skeleton className="h-4 w-4 rounded-full" />
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
            <div className="bg-white/90 backdrop-blur-md border border-primary/20 rounded-2xl p-6 shadow-[0_16px_40px_rgba(193,95,60,0.12),0_4px_12px_rgba(0,0,0,0.06)]">
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
                      formatter={(v) => [`₹${Number(v).toLocaleString('en-IN')}`, 'Price']} />
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

      {/* ── Portfolio Results — auto-populated when Apply is clicked ── */}
      {simResult && (
        <div className="space-y-6 mb-6">

          {/* Summary cards */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-display font-bold text-lg text-secondary flex items-center gap-2">
                <Calculator size={18} className="text-primary" /> Portfolio Simulation
              </h2>
              <span className="text-xs text-text-muted">
                {data.date_start} → {data.date_end} · {data.n_days} trading days
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Invested', val: fmt(simResult.totalInvested), color: 'text-text', bg: 'bg-bg-card border-border-light' },
                { label: 'Current Value', val: fmt(simResult.totalValue), color: simResult.totalProfit >= 0 ? 'text-profit' : 'text-loss', bg: simResult.totalProfit >= 0 ? 'bg-profit/5 border-profit/20' : 'bg-loss/5 border-loss/20' },
                { label: 'Total P&L', val: `${simResult.totalProfit >= 0 ? '+' : ''}${fmt(simResult.totalProfit)}`, color: simResult.totalProfit >= 0 ? 'text-profit' : 'text-loss', bg: simResult.totalProfit >= 0 ? 'bg-profit/5 border-profit/20' : 'bg-loss/5 border-loss/20' },
                { label: 'Return %', val: `${simResult.totalReturnPct >= 0 ? '+' : ''}${simResult.totalReturnPct.toFixed(2)}%`, color: simResult.totalReturnPct >= 0 ? 'text-profit' : 'text-loss', bg: simResult.totalReturnPct >= 0 ? 'bg-profit/5 border-profit/20' : 'bg-loss/5 border-loss/20' },
              ].map(c => (
                <div key={c.label} className={`rounded-xl px-4 py-3 border ${c.bg}`}>
                  <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">{c.label}</p>
                  <p className={`font-mono font-bold text-base ${c.color}`}>{c.val}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* Growth Chart */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-display font-bold text-lg text-secondary flex items-center gap-2">
                <TrendingUp size={18} className="text-primary" /> Growth Chart
              </h2>
              {growthResult && <span className="text-xs text-text-muted">{growthResult.n_days} trading days</span>}
            </div>

            {growthLoading && (
              <div className="flex items-center gap-2 py-8 text-sm text-text-muted justify-center">
                <svg className="animate-spin h-4 w-4 text-primary" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                Loading growth data…
              </div>
            )}

            {!growthResult && !growthLoading && (
              <p className="text-sm text-text-muted text-center py-8">
                Select a date range and click <strong>Apply</strong> to see growth chart (Portfolio vs NIFTY vs FD)
              </p>
            )}

            {growthResult && !growthLoading && (
              <>
                <div className="grid grid-cols-3 gap-3 mb-4">
                  {[
                    { label: 'Our Portfolio', final: growthResult.final_portfolio, ret: growthResult.portfolio_return_pct, profit: growthResult.portfolio_profit, color: '#C15F3C' },
                    { label: 'NIFTY 50 Index', final: growthResult.final_nifty,    ret: growthResult.nifty_return_pct,    profit: growthResult.nifty_profit,    color: '#6366F1' },
                    { label: 'Fixed Deposit (7%)', final: growthResult.final_fd,   ret: growthResult.fd_return_pct,       profit: growthResult.fd_profit,       color: '#10B981' },
                  ].map(c => (
                    <div key={c.label} className="bg-bg-card rounded-xl px-3 py-3 border border-border-light">
                      <div className="flex items-center gap-1.5 mb-1">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: c.color }} />
                        <p className="text-[10px] font-medium text-text-muted">{c.label}</p>
                      </div>
                      <p className="font-mono font-bold text-sm text-text">{fmt(c.final)}</p>
                      <p className={`font-mono text-xs mt-0.5 ${c.profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {c.profit >= 0 ? '+' : ''}{fmt(c.profit)} ({c.ret >= 0 ? '+' : ''}{c.ret}%)
                      </p>
                    </div>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={240} minHeight={1}>
                  <LineChart data={growthResult.series} margin={{ top: 5, right: 5, bottom: 0, left: 10 }}>
                    <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                      tickFormatter={d => d.slice(0, 7)} interval={Math.max(1, Math.floor(growthResult.series.length / 6))} />
                    <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                      tickFormatter={v => `₹${(v / 1000).toFixed(0)}K`} />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12, fontSize: 12 }}
                      formatter={(v, name) => [`₹${Math.round(Number(v)).toLocaleString('en-IN')}`, name === 'portfolio_value' ? 'Our Portfolio' : name === 'nifty_value' ? 'NIFTY 50' : 'FD (7%)']}
                      labelFormatter={d => `Date: ${d}`} />
                    <ReferenceLine y={growthResult.amount} stroke="#9CA3AF" strokeDasharray="4 4"
                      label={{ value: 'Invested', position: 'left', fontSize: 10, fill: '#9CA3AF' }} />
                    <Legend formatter={v => v === 'portfolio_value' ? 'Our Portfolio' : v === 'nifty_value' ? 'NIFTY 50' : 'FD (7%)'} />
                    <Line type="monotone" dataKey="portfolio_value" stroke="#C15F3C" strokeWidth={2} dot={false} animationDuration={600} />
                    <Line type="monotone" dataKey="nifty_value"     stroke="#6366F1" strokeWidth={2} dot={false} animationDuration={600} />
                    <Line type="monotone" dataKey="fd_value"        stroke="#10B981" strokeWidth={1.5} strokeDasharray="4 4" dot={false} animationDuration={600} />
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}
          </Card>

          {/* Per-stock breakdown — collapsible */}
          <Card>
            <button onClick={() => setShowBreakdown(o => !o)} className="w-full flex items-center justify-between">
              <h2 className="font-display font-bold text-lg text-secondary flex items-center gap-2">
                <IndianRupee size={18} className="text-primary" /> Per-Stock Breakdown
              </h2>
              <div className="flex items-center gap-3">
                <span className="text-xs text-text-muted">{simResult.perStock.length} stocks</span>
                {showBreakdown ? <ChevronUp size={16} className="text-text-muted" /> : <ChevronDown size={16} className="text-text-muted" />}
              </div>
            </button>

            <AnimatePresence>
              {showBreakdown && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }} transition={{ type: 'spring', stiffness: 220, damping: 26 }}
                  className="overflow-hidden">
                  <div className="mt-4">
                    {/* Sector summary row */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 mb-4">
                      {Object.entries(simResult.bySector).sort((a, b) => b[1].invested - a[1].invested).map(([sector, s]) => (
                        <div key={sector} className="bg-bg-card rounded-xl px-3 py-2.5 border border-border-light">
                          <p className="text-[10px] font-medium text-text-muted truncate mb-1">{sector}</p>
                          <p className="font-mono text-xs font-bold text-text">{fmt(s.invested)}</p>
                          <p className={`font-mono text-xs mt-0.5 ${s.profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                            {s.profit >= 0 ? '+' : ''}{fmt(s.profit)}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* Sort controls */}
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-text-muted">All {simResult.perStock.length} stocks</span>
                      <div className="flex gap-1.5">
                        {(['profit', 'loss', 'weight'] as const).map(s => (
                          <button key={s} onClick={() => setSimStockSort(s)}
                            className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
                              simStockSort === s ? 'bg-primary text-white border-primary' : 'bg-bg-card border-border text-text-muted hover:border-primary'
                            }`}>
                            {s === 'profit' ? 'Top Gainers' : s === 'loss' ? 'Top Losers' : 'By Weight'}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="max-h-72 overflow-y-auto rounded-xl border border-border-light">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-white border-b border-border">
                          <tr>
                            <th className="text-left py-2 px-3 font-medium text-text-secondary">#</th>
                            <th className="text-left py-2 px-3 font-medium text-text-secondary">Stock</th>
                            <th className="text-left py-2 px-3 font-medium text-text-secondary hidden sm:table-cell">Sector</th>
                            <th className="text-right py-2 px-3 font-medium text-text-secondary">Invested</th>
                            <th className="text-right py-2 px-3 font-medium text-text-secondary">Value</th>
                            <th className="text-right py-2 px-3 font-medium text-text-secondary">P&amp;L</th>
                            <th className="text-right py-2 px-3 font-medium text-text-secondary">Ret%</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...simResult.perStock]
                            .sort((a, b) => simStockSort === 'profit' ? safeNum(b.profit) - safeNum(a.profit) : simStockSort === 'loss' ? safeNum(a.profit) - safeNum(b.profit) : safeNum(b.weight) - safeNum(a.weight))
                            .map((s, i) => (
                              <tr key={s.ticker} className="border-b border-border-light hover:bg-bg-card transition-colors">
                                <td className="py-2 px-3 text-text-muted font-mono text-xs">{i + 1}</td>
                                <td className="py-2 px-3 font-mono font-semibold text-text text-xs">{s.ticker}</td>
                                <td className="py-2 px-3 text-text-secondary text-xs hidden sm:table-cell">{s.sector}</td>
                                <td className="py-2 px-3 text-right font-mono text-xs text-text">{fmt(s.invested)}</td>
                                <td className="py-2 px-3 text-right font-mono text-xs text-text">{fmt(s.currentValue)}</td>
                                <td className={`py-2 px-3 text-right font-mono font-semibold text-xs ${s.profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                                  {s.profit >= 0 ? '+' : ''}{fmt(s.profit)}
                                </td>
                                <td className={`py-2 px-3 text-right font-mono text-xs ${s.returnPct >= 0 ? 'text-profit' : 'text-loss'}`}>
                                  {s.returnPct >= 0 ? '+' : ''}{s.returnPct.toFixed(1)}%
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>

        </div>
      )}
    </div>
  );
}
