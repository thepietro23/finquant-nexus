import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare, Send, TrendingUp, TrendingDown, Minus,
  Newspaper, ArrowUpRight, ArrowDownRight, RefreshCw,
  BarChart3, Briefcase, ChevronDown, ChevronUp, Zap, Activity,
} from 'lucide-react';
import { api } from '../lib/api';
import { toast } from '../lib/toast';
import type {
  SentimentResponse, NewsSentimentResponse, NewsItem,
} from '../lib/api';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, PieChart as RPieChart, Pie,
  LineChart, Line, ReferenceLine,
} from 'recharts';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import type { MetricBadge } from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import Badge from '../components/ui/Badge';
import { staggerContainer } from '../lib/animations';

function safeNum(v: unknown, fallback = 0): number {
  const n = Number(v);
  return isFinite(n) ? n : fallback;
}

function getMoodBadge(mood: string): MetricBadge {
  if (mood === 'Bullish')  return { label: 'BULLISH',  variant: 'profit' };
  if (mood === 'Bearish')  return { label: 'BEARISH',  variant: 'loss' };
  return { label: 'NEUTRAL', variant: 'neutral' };
}

function getAvgScoreBadge(score: number): MetricBadge {
  if (score > 0.15)  return { label: 'STRONG +VE', variant: 'profit' };
  if (score > 0.05)  return { label: 'POSITIVE',   variant: 'profit' };
  if (score > -0.05) return { label: 'NEUTRAL',    variant: 'neutral' };
  if (score > -0.15) return { label: 'NEGATIVE',   variant: 'loss' };
  return { label: 'STRONG -VE', variant: 'loss' };
}

const PAGE_INFO = {
  title: 'Sentiment Monitor — How to Use This Tab',
  sections: [
    { heading: '1. What is this tab?', text: 'Live financial news sentiment powered by FinBERT — a BERT model fine-tuned on 10,000+ financial texts (earnings calls, analyst reports, market news). Fetches headlines from Indian RSS feeds (Economic Times, Business Standard, LiveMint) → yFinance news → Google News RSS as fallback. Scores headlines in real-time: −1.0 (very negative) to +1.0 (very positive).' },
    { heading: '2. Auto-refresh & caching', text: 'Page auto-refreshes every 3 minutes (FinBERT inference takes ~15-20s per batch, so shorter intervals are wasteful). TTL cache = 15 minutes server-side — clicking "Force Refresh" bypasses cache and re-fetches all feeds. Session trend chart stores up to 48 data points in localStorage (~2.4 hours of mood history across reloads).' },
    { heading: '3. How FinBERT scores work', text: 'Input: a raw headline string. Output: [P(positive), P(negative), P(neutral)] — three probabilities summing to 100%. Score = P(positive) − P(negative). Range: −1.0 to +1.0. Why FinBERT and not general NLP? "Markets turn bearish" = Negative in finance, Neutral in general text. "Profit booking seen" = Neutral in finance, Positive elsewhere. Domain matters.' },
    { heading: '4. Portfolio Impact tab — weight formula', text: 'adjusted_weight = base_weight × (1 + score × 2.0), then all weights normalized to sum 100%. Sensitivity = 2.0. Example: HDFC Bank base = 2.13%, score = +0.30 → adjusted = 2.13 × 1.60 = 3.41% (+1.28% overweight). Negative score → underweight. Stocks with no specific news inherit their sector\'s average score.' },
    { heading: '5. Where does it fit in Smart Portfolio?', text: 'Sentiment contributes 40% of the Smart Portfolio signal: RL momentum weights (40%) + Sentiment-adjusted weights (40%) + Federated Learning sector weights (20%) → SLSQP Max Sharpe optimization. A "Bullish" market (avg score >0.08) → positive-news stocks overweight. "Bearish" → negative-news stocks trimmed automatically.' },
    { heading: '6. Sectors tab & Market Mood', text: 'Market Mood = "Bullish" (avg >0.08), "Bearish" (avg <−0.08), "Neutral" (between). Sector tab aggregates all headlines per sector — identifies which sectors have news momentum right now. Score buckets: Very Negative (<−0.30), Negative, Neutral, Positive, Very Positive (>+0.30). High-Impact Alerts = any score beyond ±0.3.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'Headlines Analyzed': {
    what: 'Total number of real financial news headlines fetched and analyzed by FinBERT.',
    why: 'More headlines = more robust sentiment signal. A single headline can be misleading; aggregating many gives a clearer picture.',
    how: 'Indian RSS feeds (ET, BusinessStandard, LiveMint) as primary source, yFinance and Google News as fallbacks. Each headline passed through FinBERT for sentiment scoring.',
    good: '50-100 headlines is ideal for NIFTY 50 coverage. <20 = too few for reliable sector-level signals.',
  },
  'Market Mood': {
    what: 'Overall market sentiment derived from averaging all headline scores. Bullish (>0.08), Bearish (<-0.08), or Neutral.',
    why: 'A quick pulse check on the market. When most news is positive, market tends to be in risk-on mode; negative = risk-off.',
    how: 'Simple average of all FinBERT scores. Threshold: >0.08 = Bullish, <-0.08 = Bearish, else Neutral.',
    good: 'Score between -0.1 and +0.1 = typical market. Beyond ±0.2 = strong sentiment (unusual, often around earnings season or macro events).',
  },
  'Avg Score': {
    what: 'Mean sentiment score across all analyzed headlines. Ranges from -1.0 (extremely negative) to +1.0 (extremely positive).',
    why: 'Quantifies overall market sentiment into a single number. Used as an input feature for the RL agent.',
    how: 'Average of FinBERT scores: score = positive_prob - negative_prob. Each headline contributes equally.',
    good: 'Most days score between -0.05 and +0.15 (slight positive bias in financial news). Extreme values (>0.3 or <-0.3) are rare.',
  },
  'Top Mover': {
    what: 'The stock with the largest sentiment-driven weight change from equal-weight baseline.',
    why: 'Identifies which stock sentiment would most affect portfolio allocation. High positive = overweight, high negative = underweight.',
    how: 'Weight adjustment = base_weight × (1 + sentiment × 2.0), then normalized. Biggest absolute change = top mover.',
    good: 'Weight changes of ±0.5% are normal. >±1% = strong sentiment signal for that stock.',
  },
};

const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Infra': '#14B8A6', 'Others': '#9CA3AF',
  'Market': '#374151', 'Unknown': '#9CA3AF',
};

function ScoreBar({ score }: { score: number }) {
  const pct = ((score + 1) / 2) * 100;
  return (
    <div className="relative h-2 bg-gray-100 rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        className="h-full rounded-full"
        style={{
          background: score >= 0.1
            ? 'linear-gradient(90deg, #F59E0B, #16A34A)'
            : score <= -0.1
            ? 'linear-gradient(90deg, #DC2626, #F59E0B)'
            : 'linear-gradient(90deg, #F59E0B 40%, #EAB308 60%)',
        }}
      />
    </div>
  );
}

// Color helpers
const LABEL_COLOR = { positive: '#16A34A', negative: '#DC2626', neutral: '#9CA3AF' }
const LABEL_BG = { positive: '#DCFCE7', negative: '#FEE2E2', neutral: '#F3F4F6' }

function SentimentBar({ positive, negative, neutral }: { positive: number; negative: number; neutral: number }) {
  return (
    <div className="w-full">
      <div className="flex rounded-full overflow-hidden h-2.5">
        <div style={{ width: `${positive * 100}%`, background: '#16A34A' }} />
        <div style={{ width: `${neutral * 100}%`, background: '#D1D5DB' }} />
        <div style={{ width: `${negative * 100}%`, background: '#DC2626' }} />
      </div>
      <div className="flex justify-between mt-1.5 text-[10px] font-medium">
        <span className="text-[#16A34A]">▲ Positive {(positive * 100).toFixed(0)}%</span>
        <span className="text-gray-400">Neutral {(neutral * 100).toFixed(0)}%</span>
        <span className="text-[#DC2626]">▼ Negative {(negative * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

function NewsCard({ item, index, isNew }: { item: NewsItem; index: number; isNew?: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const label = item.label as 'positive' | 'negative' | 'neutral'
  const scoreAbs = Math.abs(item.score)
  const confidence = scoreAbs > 0.5 ? 'High' : scoreAbs > 0.25 ? 'Medium' : 'Low'
  const confColor = scoreAbs > 0.5 ? '#16A34A' : scoreAbs > 0.25 ? '#F59E0B' : '#9CA3AF'

  return (
    <motion.div
      initial={{ opacity: 0, y: isNew ? -12 : 0, scale: isNew ? 0.98 : 1 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: isNew ? 0 : index * 0.03, duration: 0.28 }}
      className={`rounded-xl border transition-all duration-200 overflow-hidden ${
        expanded ? 'border-primary/40 shadow-md shadow-primary/8' : 'border-border-light hover:border-border'
      } ${isNew ? 'ring-1 ring-blue-300' : ''}`}
      style={{ background: expanded ? (LABEL_BG[label] + '55') : '#fff' }}
    >
      {/* Main row — always visible */}
      <div
        className="flex items-start gap-3 p-3 cursor-pointer"
        onClick={() => setExpanded(e => !e)}
      >
        {/* Score badge */}
        <div
          className="shrink-0 w-10 h-10 rounded-xl flex flex-col items-center justify-center text-white text-[10px] font-bold gap-0"
          style={{ background: LABEL_COLOR[label] }}
        >
          <span className="text-base font-black leading-none">
            {item.score > 0.05 ? '▲' : item.score < -0.05 ? '▼' : '—'}
          </span>
          <span className="text-[9px] leading-none opacity-90">
            {item.score > 0 ? '+' : ''}{(item.score * 100).toFixed(0)}
          </span>
        </div>

        {/* Headline + meta */}
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text leading-snug font-medium" style={{ display: '-webkit-box', WebkitLineClamp: expanded ? undefined : 2, WebkitBoxOrient: 'vertical', overflow: expanded ? 'visible' : 'hidden' }}>
            {item.headline}
          </p>
          <div className="flex flex-wrap items-center gap-2 mt-1.5">
            {item.ticker && item.ticker !== 'MARKET' && (
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border"
                style={{ color: SECTOR_COLORS[item.sector] || '#6B7280', borderColor: SECTOR_COLORS[item.sector] + '50' || '#E5E7EB', background: SECTOR_COLORS[item.sector] + '12' || '#F9FAFB' }}>
                {item.ticker}
              </span>
            )}
            {item.sector && (
              <span className="text-[10px] text-text-muted">{item.sector}</span>
            )}
            {item.published && (
              <span className="text-[10px] text-text-muted">{item.published}</span>
            )}
            {isNew && (
              <span className="text-[9px] font-bold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded-full border border-blue-200">NEW</span>
            )}
          </div>
        </div>

        {/* Right: confidence + expand */}
        <div className="shrink-0 flex flex-col items-end gap-1.5">
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full border"
            style={{ color: confColor, borderColor: confColor + '40', background: confColor + '12' }}>
            {confidence} conf.
          </span>
          <button className="text-text-muted hover:text-primary transition-colors">
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Expanded: Full FinBERT breakdown — inline, no redirect */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 space-y-3 border-t border-border-light">
              {/* FinBERT label */}
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: LABEL_COLOR[label] }}>
                  FinBERT verdict: {label}
                </span>
                <Zap size={12} style={{ color: LABEL_COLOR[label] }} />
                <span className="text-xs text-text-muted ml-1">score = P(positive) − P(negative)</span>
              </div>

              {/* Stacked probability bar */}
              <SentimentBar positive={item.positive} negative={item.negative} neutral={item.neutral} />

              {/* Three numbers */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: 'Positive', val: item.positive, color: '#16A34A' },
                  { label: 'Neutral',  val: item.neutral,  color: '#9CA3AF' },
                  { label: 'Negative', val: item.negative, color: '#DC2626' },
                ].map(b => (
                  <div key={b.label} className="rounded-lg p-2.5 text-center" style={{ background: b.color + '10', border: `1px solid ${b.color}30` }}>
                    <p className="text-[10px] font-medium mb-0.5" style={{ color: b.color }}>{b.label}</p>
                    <p className="font-mono font-bold text-sm" style={{ color: b.color }}>{(b.val * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>

              {/* Net score */}
              <div className="flex items-center justify-between bg-bg-card rounded-lg px-3 py-2">
                <span className="text-xs text-text-muted">Net Sentiment Score</span>
                <span className="font-mono font-bold text-sm" style={{ color: LABEL_COLOR[label] }}>
                  {item.score > 0 ? '+' : ''}{safeNum(item.score).toFixed(4)}
                </span>
              </div>

              {/* Source */}
              {item.source && (
                <p className="text-[10px] text-text-muted italic">Source: {item.source}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

const REFRESH_INTERVAL = 180_000 // 3 minutes
const HISTORY_KEY = 'fqn_sentiment_history'
const MAX_HISTORY = 48

interface TrendPoint { time: string; score: number; mood: string }

function useTimeAgo(date: Date | null): string {
  const [label, setLabel] = useState('—')
  useEffect(() => {
    if (!date) return
    const update = () => {
      const secs = Math.floor((Date.now() - date.getTime()) / 1000)
      const next = secs < 60 ? `${secs}s ago` : `${Math.floor(secs / 60)}m ago`
      setLabel(prev => prev === next ? prev : next)
    }
    update()
    // 10s is fine — display only changes at minute boundaries
    const id = setInterval(update, 10_000)
    return () => clearInterval(id)
  }, [date])
  return label
}

export default function Sentiment() {
  // Manual analysis state
  const [text, setText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SentimentResponse | null>(null);

  // Live news state
  const [newsData, setNewsData] = useState<NewsSentimentResponse | null>(null);
  const [newsLoading, setNewsLoading] = useState(true);
  const [newsError, setNewsError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'news' | 'portfolio' | 'sectors'>('news');

  // Real-time state
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [newCount, setNewCount] = useState(0);
  const [newHeadlines, setNewHeadlines] = useState<Set<string>>(new Set());
  const prevHeadlinesRef = useRef<Set<string>>(new Set());
  const loadingRef = useRef(false);
  const [history, setHistory] = useState<TrendPoint[]>(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') }
    catch { return [] }
  });

  const timeAgo = useTimeAgo(lastUpdated);

  function loadNewsSentiment(force = false) {
    if (loadingRef.current) return;  // prevent overlapping calls
    loadingRef.current = true;
    setNewsLoading(true);
    setNewsError(null);
    api.newsSentiment(force)
      .then(d => {
        // Track new headlines
        const incoming = new Set(d.news.map(n => n.headline))
        const brandNew = [...incoming].filter(h => !prevHeadlinesRef.current.has(h))
        if (prevHeadlinesRef.current.size > 0) {
          setNewCount(brandNew.length)
          setNewHeadlines(new Set(brandNew))
          // Clear new badges after 8s
          setTimeout(() => setNewHeadlines(new Set()), 8000)
        }
        prevHeadlinesRef.current = incoming

        // Update trend history in localStorage
        const point: TrendPoint = {
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          score: d.avg_score,
          mood: d.market_mood,
        }
        setHistory(prev => {
          const next = [...prev.slice(-(MAX_HISTORY - 1)), point]
          try { localStorage.setItem(HISTORY_KEY, JSON.stringify(next)) } catch { /* quota exceeded */ }
          return next
        })

        setNewsData(d);
        setLastUpdated(new Date());
      })
      .catch(e => {
        setNewsError(e instanceof Error ? e.message : 'Failed to fetch news sentiment');
      })
      .finally(() => {
        setNewsLoading(false);
        loadingRef.current = false;
      });
  }

  // Auto-refresh every 3 minutes — pauses when the tab is hidden to avoid wasted API calls
  useEffect(() => {
    loadNewsSentiment();
    const id = setInterval(() => {
      if (!document.hidden) loadNewsSentiment();
    }, REFRESH_INTERVAL);
    return () => clearInterval(id);
  }, []);

  async function analyze() {
    if (!text.trim()) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await api.sentiment(text);
      setResult(res);
      const label = res.label ?? 'neutral';
      const icon = label === 'positive' ? '📈' : label === 'negative' ? '📉' : '➖';
      toast.info(`${icon} FinBERT: ${label} (score ${safeNum(res.score).toFixed(3)})`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Sentiment analysis failed — is the backend running?';
      setError(msg);
      toast.error(msg);
    } finally {
      setAnalyzing(false);
    }
  }

  // High-impact articles: abs(score) > 0.3, sorted strongest first, capped at 8
  const highImpactNews = newsData
    ? [...newsData.news]
        .filter(n => Math.abs(n.score) > 0.3)
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
        .slice(0, 8)
    : [];

  // Derived data — guard against empty array (not just undefined)
  const topMover = newsData?.portfolio_impact?.length ? newsData.portfolio_impact[0] : undefined;
  const moodValue = newsData ? (newsData.market_mood === 'Bullish' ? 1 : newsData.market_mood === 'Bearish' ? -1 : 0) : 0;

  // Score distribution for pie chart
  const distData = newsData ? Object.entries(newsData.score_distribution)
    .map(([key, value]) => ({
      name: key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value,
      color: key === 'very_positive' ? '#16A34A' : key === 'positive' ? '#86EFAC'
        : key === 'neutral' ? '#F59E0B' : key === 'negative' ? '#FCA5A5' : '#DC2626',
    }))
    .filter(d => d.value > 0) : [];

  // Sector bar chart data
  const sectorChartData = newsData?.sector_sentiment.map(s => ({
    sector: s.sector.length > 10 ? s.sector.slice(0, 10) : s.sector,
    score: s.avg_score,
    headlines: s.n_headlines,
  })) || [];

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <PageHeader
            title="Sentiment Monitor"
            subtitle={`FinBERT NLP — Indian RSS feeds + Google News · contributes 40% to Smart Portfolio`}
            icon={<MessageSquare size={24} />}
          />
          {/* LIVE indicator */}
          <div className="flex items-center gap-2 ml-2">
            {newsLoading ? (
              <span className="flex items-center gap-1.5 text-xs text-text-muted">
                <RefreshCw size={12} className="animate-spin" /> Refreshing...
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs font-medium text-[#16A34A]">
                <span className="w-2 h-2 rounded-full bg-[#16A34A] animate-pulse" />
                LIVE · {timeAgo}
              </span>
            )}
            {newCount > 0 && (
              <span className="text-xs bg-blue-50 text-blue-600 font-medium px-2 py-0.5 rounded-full">
                +{newCount} new
              </span>
            )}
          </div>
        </div>
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Sentiment Trend Chart (localStorage persistent) */}
      {history.length > 1 && (
        <div className="mb-4 bg-white border border-border rounded-xl px-4 pt-3 pb-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-text-secondary">Sentiment Trend (Session History)</span>
            <span className="text-xs text-text-muted">{history.length} data points · auto-saves</span>
          </div>
          <ResponsiveContainer width="100%" height={70}>
            <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
              <ReferenceLine y={0} stroke="#D1D5DB" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="score" stroke="#16A34A" strokeWidth={2} dot={false} isAnimationActive={false} />
              <XAxis dataKey="time" hide />
              <YAxis domain={[-1, 1]} hide />
              <Tooltip
                contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 8, fontSize: 11 }}
                formatter={(v) => [Number(v).toFixed(3), 'Avg Score']}
                labelFormatter={(l) => `Time: ${l}`}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metric Cards */}
      {newsData && (
        <>
          <motion.div variants={staggerContainer} initial="hidden" animate="visible"
            className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
            <MetricCard title="Headlines Analyzed" value={newsData.n_headlines} decimals={0}
              icon={<Newspaper size={18} />}
              badge={newsData.n_headlines >= 50 ? { label: 'RICH SIGNAL', variant: 'profit' } : newsData.n_headlines >= 20 ? { label: 'ADEQUATE', variant: 'warning' } : { label: 'LOW SIGNAL', variant: 'loss' }}
              onClick={() => setExpandedMetric(m => m === 'Headlines Analyzed' ? null : 'Headlines Analyzed')}
              active={expandedMetric === 'Headlines Analyzed'} />
            <MetricCard title="Market Mood" value={moodValue} decimals={0}
              prefix={newsData.market_mood === 'Bullish' ? '↑ ' : newsData.market_mood === 'Bearish' ? '↓ ' : '→ '}
              suffix={` ${newsData.market_mood}`}
              icon={newsData.market_mood === 'Bullish' ? <TrendingUp size={18} /> : newsData.market_mood === 'Bearish' ? <TrendingDown size={18} /> : <Minus size={18} />}
              badge={getMoodBadge(newsData.market_mood)}
              onClick={() => setExpandedMetric(m => m === 'Market Mood' ? null : 'Market Mood')}
              active={expandedMetric === 'Market Mood'} />
            <MetricCard title="Avg Score" value={newsData.avg_score} decimals={4}
              icon={<BarChart3 size={18} />}
              badge={getAvgScoreBadge(newsData.avg_score)}
              onClick={() => setExpandedMetric(m => m === 'Avg Score' ? null : 'Avg Score')}
              active={expandedMetric === 'Avg Score'} />
            <MetricCard title="Top Mover"
              value={topMover ? topMover.weight_change : 0} decimals={2} suffix="%"
              prefix={topMover ? `${topMover.ticker} ` : ''}
              icon={<Briefcase size={18} />}
              badge={topMover ? (topMover.weight_change > 0 ? { label: 'OVERWEIGHT', variant: 'profit' } : { label: 'UNDERWEIGHT', variant: 'loss' }) : undefined}
              onClick={() => setExpandedMetric(m => m === 'Top Mover' ? null : 'Top Mover')}
              active={expandedMetric === 'Top Mover'} />
          </motion.div>
          <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />
        </>
      )}

      {/* Smart Portfolio Integration Callout */}
      <div className="mb-6 rounded-xl border border-primary/20 bg-primary/4 px-4 py-3">
        <div className="flex items-start gap-3">
          <Briefcase size={16} className="text-primary shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-xs font-bold text-primary uppercase tracking-wide mb-1">Smart Portfolio Integration — Sentiment = 40%</p>
            <div className="grid grid-cols-3 gap-2 text-[11px] mb-2">
              <div className="rounded-lg border border-border-light bg-white px-2.5 py-1.5 text-center">
                <p className="text-text-muted">RL Agents</p>
                <p className="font-bold text-text">× 40%</p>
              </div>
              <div className="rounded-lg border-2 border-primary/30 bg-primary/6 px-2.5 py-1.5 text-center">
                <p className="text-primary font-semibold">Sentiment</p>
                <p className="font-bold text-primary">× 40% ← you are here</p>
              </div>
              <div className="rounded-lg border border-border-light bg-white px-2.5 py-1.5 text-center">
                <p className="text-text-muted">Federated</p>
                <p className="font-bold text-text">× 20%</p>
              </div>
            </div>
            <p className="text-[11px] text-text-secondary">
              Combined weights → SLSQP Max Sharpe optimization → <span className="font-semibold text-text">Final Smart Portfolio</span>.
              Positive market mood boosts high-sentiment stocks; bearish mood trims them. Formula: <span className="font-mono text-[10px]">adj_w = base × (1 + score × 2.0)</span>, normalized.
            </p>
          </div>
        </div>
      </div>

      {/* Manual Analysis */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-3">Analyze Custom Text</h2>
        <div className="flex gap-3">
          <input
            type="text" value={text} onChange={e => setText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && analyze()}
            placeholder="Enter any financial headline..."
            maxLength={500}
            className="flex-1 px-4 py-2.5 border border-border rounded-xl text-sm focus:outline-none focus:border-primary transition-colors"
          />
          <motion.button onClick={analyze} disabled={analyzing || !text.trim()}
            whileHover={!analyzing ? { scale: 1.03, y: -1 } : {}}
            whileTap={!analyzing ? { scale: 0.96 } : {}}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
              hover:bg-primary-hover transition-colors disabled:opacity-50 shadow-sm">
            <motion.span animate={analyzing ? { rotate: 360 } : { rotate: 0 }}
              transition={analyzing ? { repeat: Infinity, duration: 0.8, ease: 'linear' } : {}}>
              <Send size={16} />
            </motion.span>
            {analyzing ? 'Analyzing…' : 'Analyze'}
          </motion.button>
        </div>

        {error && <p className="mt-3 text-sm text-loss">{error}</p>}

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-bg-card rounded-xl overflow-hidden"
            >
              <div className="flex items-center gap-3 mb-3">
                <Badge variant={result.label === 'positive' ? 'profit' : result.label === 'negative' ? 'loss' : 'neutral'}>
                  {result.label === 'positive' ? <TrendingUp size={12} /> : result.label === 'negative' ? <TrendingDown size={12} /> : <Minus size={12} />}
                  <span className="ml-1">{result.label.toUpperCase()}</span>
                </Badge>
                <span className="font-mono text-xl font-bold text-text">{safeNum(result.score).toFixed(4)}</span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm mb-3">
                <div className="text-center p-2 rounded-lg bg-white">
                  <p className="text-text-muted text-xs mb-1">Positive</p>
                  <p className="font-mono font-semibold text-profit">{(safeNum(result.positive) * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center p-2 rounded-lg bg-white">
                  <p className="text-text-muted text-xs mb-1">Negative</p>
                  <p className="font-mono font-semibold text-loss">{(safeNum(result.negative) * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center p-2 rounded-lg bg-white">
                  <p className="text-text-muted text-xs mb-1">Neutral</p>
                  <p className="font-mono font-semibold text-text-secondary">{(safeNum(result.neutral) * 100).toFixed(1)}%</p>
                </div>
              </div>
              <ScoreBar score={result.score} />
              <div className="flex justify-between text-[10px] text-text-muted mt-1">
                <span>-1.0 (Very Negative)</span><span>0 (Neutral)</span><span>+1.0 (Very Positive)</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Live News Section */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="font-display font-bold text-lg text-secondary">Live News Feed</h2>
            {newsData && (
              <span className="text-xs text-text-muted bg-bg-card px-2 py-0.5 rounded-md border border-border-light">
                {newsData.n_headlines} headlines
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Tab switcher with layoutId sliding pill */}
            <div className="flex bg-bg-card rounded-lg border border-border-light p-0.5 relative">
              {(['news', 'portfolio', 'sectors'] as const).map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)}
                  className={`relative px-3 py-1.5 rounded-md text-xs font-medium transition-colors z-10 ${
                    activeTab === tab ? 'text-white' : 'text-text-secondary hover:text-text'
                  }`}>
                  {activeTab === tab && (
                    <motion.span layoutId="sentiment-tab-pill"
                      className="absolute inset-0 bg-primary rounded-md shadow-sm"
                      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-1">
                    {tab === 'news' ? 'News' : tab === 'portfolio' ? 'Portfolio Impact' : 'Sectors'}
                    {tab === 'news' && activeTab !== 'news' && newCount > 0 && (
                      <span className="bg-blue-500 text-white text-[9px] font-bold px-1.5 rounded-full">+{newCount}</span>
                    )}
                  </span>
                </button>
              ))}
            </div>
            <button onClick={() => loadNewsSentiment(true)} disabled={newsLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-primary bg-primary-subtle rounded-lg hover:bg-primary-light transition-colors disabled:opacity-50">
              <RefreshCw size={12} className={newsLoading ? 'animate-spin' : ''} />
              {newsLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>

        {newsError && <p className="text-sm text-loss mb-3">{newsError}</p>}

        {newsLoading && !newsData && (
          <div className="space-y-3 py-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex gap-3 p-3 rounded-xl border border-border-light">
                <div className="skeleton-shimmer rounded-full w-8 h-8 shrink-0" />
                <div className="flex-1 space-y-2">
                  <div className="skeleton-shimmer rounded h-4 w-3/4" />
                  <div className="skeleton-shimmer rounded h-3 w-1/2" />
                </div>
                <div className="skeleton-shimmer rounded-full w-16 h-6 shrink-0" />
              </div>
            ))}
            <p className="text-xs text-text-muted text-center pt-2">Running FinBERT on live news... (15–30s)</p>
          </div>
        )}

        {newsData && (
          <AnimatePresence mode="wait">
            {/* NEWS TAB */}
            {activeTab === 'news' && (
              <motion.div key="news" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>

                {/* Live ticker strip */}
                {newsData.news.length > 0 && (
                  <div className="mb-3 overflow-hidden rounded-lg bg-bg-card border border-border-light py-2 px-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Activity size={11} className="text-primary shrink-0" />
                      <span className="text-[10px] font-bold text-primary uppercase tracking-wider">Live Feed</span>
                      <span className="text-[10px] text-text-muted">— Click any card to see full FinBERT analysis</span>
                    </div>
                    <div className="flex gap-3 overflow-x-auto pb-1 scrollbar-hide">
                      {newsData.news.slice(0, 8).map((n, i) => (
                        <div key={i} className="shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border"
                          style={{
                            background: n.label === 'positive' ? '#DCFCE7' : n.label === 'negative' ? '#FEE2E2' : '#F3F4F6',
                            borderColor: n.label === 'positive' ? '#16A34A40' : n.label === 'negative' ? '#DC262640' : '#D1D5DB',
                            color: n.label === 'positive' ? '#16A34A' : n.label === 'negative' ? '#DC2626' : '#6B7280',
                          }}>
                          {n.ticker && n.ticker !== 'MARKET' && <span className="font-mono font-bold">{n.ticker}</span>}
                          <span>{n.score > 0 ? '+' : ''}{(n.score * 100).toFixed(0)}</span>
                          {n.label === 'positive' ? <TrendingUp size={10} /> : n.label === 'negative' ? <TrendingDown size={10} /> : <Minus size={10} />}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="space-y-2 max-h-[540px] overflow-y-auto pr-1">
                  {newsData.news.map((item, i) => (
                    <NewsCard
                      key={item.headline}
                      item={item}
                      index={i}
                      isNew={newHeadlines.has(item.headline)}
                    />
                  ))}
                </div>
              </motion.div>
            )}

            {/* PORTFOLIO IMPACT TAB */}
            {activeTab === 'portfolio' && (
              <motion.div key="portfolio" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <p className="text-xs text-text-secondary mb-3">
                  How sentiment would adjust equal-weight portfolio. Positive sentiment → overweight, negative → underweight.
                </p>
                <div className="overflow-auto max-h-[480px]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-white z-10">
                      <tr className="border-b border-border">
                        <th className="text-left py-2 font-medium text-text-secondary">Stock</th>
                        <th className="text-left py-2 font-medium text-text-secondary">Sector</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Sentiment</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Base Wt</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Adj Wt</th>
                        <th className="text-right py-2 font-medium text-text-secondary">Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {newsData.portfolio_impact.slice(0, 25).map((h, i) => (
                        <motion.tr key={h.ticker}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.03 }}
                          className="border-b border-border-light hover:bg-bg-card transition-colors">
                          <td className="py-2 font-mono font-medium">{h.ticker}</td>
                          <td className="py-2">
                            <span className="text-xs px-2 py-0.5 rounded-full bg-bg-card"
                              style={{ color: SECTOR_COLORS[h.sector] || '#9CA3AF' }}>
                              {h.sector}
                            </span>
                          </td>
                          <td className="py-2 text-right">
                            <span className={`font-mono text-xs font-semibold ${
                              h.sentiment_score > 0.05 ? 'text-profit' : h.sentiment_score < -0.05 ? 'text-loss' : 'text-text-muted'
                            }`}>
                              {h.sentiment_score > 0 ? '+' : ''}{safeNum(h.sentiment_score).toFixed(3)}
                            </span>
                          </td>
                          <td className="py-2 text-right font-mono text-text-muted">{safeNum(h.base_weight).toFixed(1)}%</td>
                          <td className="py-2 text-right font-mono font-semibold">{safeNum(h.adjusted_weight).toFixed(2)}%</td>
                          <td className="py-2 text-right">
                            <span className={`inline-flex items-center gap-0.5 font-mono text-xs font-semibold ${
                              h.weight_change > 0 ? 'text-profit' : h.weight_change < 0 ? 'text-loss' : 'text-text-muted'
                            }`}>
                              {h.weight_change > 0 ? <ArrowUpRight size={10} /> : h.weight_change < 0 ? <ArrowDownRight size={10} /> : null}
                              {h.weight_change > 0 ? '+' : ''}{safeNum(h.weight_change).toFixed(2)}%
                            </span>
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {/* SECTORS TAB */}
            {activeTab === 'sectors' && (
              <motion.div key="sectors" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Sector bar chart */}
                  <div>
                    <h3 className="text-sm font-medium text-text-secondary mb-3">Sector Sentiment Scores</h3>
                    <ResponsiveContainer width="100%" height={300} minHeight={1}>
                      <BarChart data={sectorChartData} layout="vertical" margin={{ top: 5, right: 10, bottom: 5, left: 70 }}>
                        <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 11, fill: '#9CA3AF' }}
                          axisLine={{ stroke: '#E5E7EB' }} tickLine={false} domain={[-0.5, 0.5]} />
                        <YAxis type="category" dataKey="sector" tick={{ fontSize: 11, fill: '#6B7280' }}
                          axisLine={false} tickLine={false} width={65} />
                        <Tooltip
                          contentStyle={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 12, fontSize: 12 }}
                          formatter={(v) => [Number(v).toFixed(4), 'Score']}
                        />
                        <Bar dataKey="score" name="Sentiment" radius={[0, 6, 6, 0]} animationDuration={1000}>
                          {sectorChartData.map((s, i) => (
                            <Cell key={i} fill={s.score >= 0.05 ? '#16A34A' : s.score <= -0.05 ? '#DC2626' : '#F59E0B'} opacity={0.8} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Sector detail cards */}
                  <div className="space-y-2 max-h-[340px] overflow-y-auto">
                    {newsData.sector_sentiment.map((s, i) => (
                      <motion.div key={s.sector}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.06 }}
                        className="p-3 rounded-xl bg-bg-card border border-border-light"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: SECTOR_COLORS[s.sector] || '#9CA3AF' }} />
                            <span className="font-medium text-sm">{s.sector}</span>
                            <span className="text-[10px] text-text-muted">({s.n_headlines} headlines)</span>
                          </div>
                          <span className={`font-mono text-sm font-bold ${
                            s.avg_score > 0.05 ? 'text-profit' : s.avg_score < -0.05 ? 'text-loss' : 'text-text-muted'
                          }`}>
                            {s.avg_score > 0 ? '+' : ''}{safeNum(s.avg_score).toFixed(4)}
                          </span>
                        </div>
                        <div className="flex gap-2">
                          <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden flex">
                            <div className="h-full bg-profit rounded-l-full" style={{ width: `${s.positive_pct}%` }} />
                            <div className="h-full bg-amber-300" style={{ width: `${100 - s.positive_pct - s.negative_pct}%` }} />
                            <div className="h-full bg-loss rounded-r-full" style={{ width: `${s.negative_pct}%` }} />
                          </div>
                        </div>
                        <div className="flex justify-between text-[10px] text-text-muted mt-1">
                          <span className="text-profit">{s.positive_pct}% positive</span>
                          <span className="text-loss">{s.negative_pct}% negative</span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </Card>

      {/* ── High Impact Alerts ── */}
      {highImpactNews.length > 0 && (
        <Card className="mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex items-center gap-2">
              <Zap size={18} className="text-amber-500" />
              <h2 className="font-display font-bold text-lg text-secondary">High Impact Alerts</h2>
            </div>
            <span className="text-xs bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded-full font-medium">
              {highImpactNews.length} strong signals
            </span>
            <span className="text-xs text-text-muted">abs(score) &gt; 0.30 — headlines with strongest FinBERT conviction</span>
          </div>
          <div className="space-y-2">
            {highImpactNews.map((item, i) => {
              const label = item.label as 'positive' | 'negative' | 'neutral';
              const borderColor = label === 'positive' ? '#16A34A' : label === 'negative' ? '#DC2626' : '#F59E0B';
              return (
                <motion.div
                  key={item.headline}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="flex items-center gap-3 p-3 rounded-xl border-l-4"
                  style={{ borderLeftColor: borderColor, background: borderColor + '08', border: `1px solid ${borderColor}25`, borderLeft: `4px solid ${borderColor}` }}
                >
                  {/* Score badge */}
                  <div className="shrink-0 w-12 h-12 rounded-xl flex flex-col items-center justify-center text-white text-[10px] font-bold"
                    style={{ background: borderColor }}>
                    <span className="text-base font-black leading-none">
                      {item.score > 0.05 ? '▲' : item.score < -0.05 ? '▼' : '—'}
                    </span>
                    <span className="text-[9px] leading-none opacity-90">
                      {item.score > 0 ? '+' : ''}{(item.score * 100).toFixed(0)}
                    </span>
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-text leading-snug line-clamp-2">{item.headline}</p>
                    <div className="flex flex-wrap items-center gap-2 mt-1">
                      {item.ticker && item.ticker !== 'MARKET' && (
                        <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border"
                          style={{ color: SECTOR_COLORS[item.sector] || '#6B7280', borderColor: (SECTOR_COLORS[item.sector] || '#6B7280') + '50', background: (SECTOR_COLORS[item.sector] || '#6B7280') + '12' }}>
                          {item.ticker}
                        </span>
                      )}
                      <span className="text-[10px] text-text-muted">{item.sector}</span>
                      {item.published && <span className="text-[10px] text-text-muted">{item.published}</span>}
                      {item.source && <span className="text-[10px] text-text-muted italic">{item.source}</span>}
                    </div>
                  </div>

                  {/* Abs score pill */}
                  <div className="shrink-0 text-right">
                    <span className="text-xs font-bold px-2 py-1 rounded-lg"
                      style={{ color: borderColor, background: borderColor + '15' }}>
                      {(Math.abs(item.score) * 100).toFixed(0)}% conf
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Score Distribution */}
      {newsData && distData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <h2 className="font-display font-bold text-lg text-secondary mb-3">Score Distribution</h2>
            <p className="text-xs text-text-secondary mb-3">
              How headlines are distributed across sentiment buckets. A healthy market has mostly neutral headlines.
            </p>
            <ResponsiveContainer width="100%" height={240} minHeight={1}>
              <RPieChart>
                <Pie data={distData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={55} outerRadius={95} paddingAngle={3}
                  animationBegin={200} animationDuration={800}>
                  {distData.map((d, i) => (
                    <Cell key={i} fill={d.color} stroke="#fff" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #E5E7EB', fontSize: 12 }}
                  formatter={(v) => [`${v} headlines`, 'Count']} />
              </RPieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap justify-center gap-3 mt-2">
              {distData.map(d => (
                <span key={d.name} className="flex items-center gap-1.5 text-[10px] text-text-secondary">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: d.color }} />
                  {d.name} ({d.value})
                </span>
              ))}
            </div>
          </Card>

          {/* How Sentiment Affects Portfolio */}
          <Card>
            <h2 className="font-display font-bold text-lg text-secondary mb-1">How Sentiment Feeds Smart Portfolio</h2>
            <p className="text-xs text-text-muted mb-3">These sentiment-adjusted weights are blended as 40% of the Smart Portfolio optimization.</p>
            <div className="space-y-3 text-sm">
              <div className="p-3 rounded-xl bg-profit-light/50 border border-profit/20">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp size={14} className="text-profit" />
                  <span className="font-semibold text-profit">Positive Sentiment → Overweight</span>
                </div>
                <p className="text-xs text-text-secondary">
                  <span className="font-mono">adj_w = base × (1 + score × 2.0)</span>. A stock with +0.3 sentiment gets ~60% more weight than baseline.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-loss-light/50 border border-loss/20">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingDown size={14} className="text-loss" />
                  <span className="font-semibold text-loss">Negative Sentiment → Underweight</span>
                </div>
                <p className="text-xs text-text-secondary">
                  A stock with -0.3 sentiment gets ~40% less weight. Acts as news-driven risk management.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-amber-50 border border-amber-200/50">
                <div className="flex items-center gap-2 mb-1">
                  <Minus size={14} className="text-amber-600" />
                  <span className="font-semibold text-amber-700">No News → Inherits Sector Average</span>
                </div>
                <p className="text-xs text-text-secondary">
                  Stocks without specific news inherit their sector's average sentiment. If Banking sector is +0.15, unmentioned banking stocks get that as their score.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-bg-card border border-border-light font-mono text-[10px] leading-relaxed">
                <p className="text-text-muted font-sans text-xs font-semibold mb-1.5">Smart Portfolio formula (Portfolio tab)</p>
                <p><span className="text-text-secondary">smart_weights =</span></p>
                <p className="pl-3"><span className="text-primary">RL weights</span> × 0.40</p>
                <p className="pl-3">+ <span className="text-[#16A34A] font-bold">Sentiment weights ← this tab</span> × 0.40</p>
                <p className="pl-3">+ <span className="text-amber-600">FL sector weights</span> × 0.20</p>
                <p className="pl-0 mt-1 text-text-muted">→ SLSQP Max Sharpe → Final weights</p>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
