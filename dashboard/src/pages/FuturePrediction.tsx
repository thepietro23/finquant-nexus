import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Play, AlertCircle } from 'lucide-react';
import { toast } from '../lib/toast';
import {
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, BarChart, Bar, ReferenceLine, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { FuturePredictionResponse, AlgoFutureStat } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import { staggerContainer, fadeSlideUp } from '../lib/animations';

// ── helpers ──────────────────────────────────────────────────────────────────

function safeNum(v: unknown, fallback = 0): number {
  const n = Number(v);
  return isFinite(n) ? n : fallback;
}

function pctFmt(v: number, d = 1) {
  const n = safeNum(v);
  return `${n >= 0 ? '+' : ''}${n.toFixed(d)}%`;
}

function returnCls(v: number) {
  const n = safeNum(v);
  if (n >= 15) return 'text-profit font-semibold';
  if (n >= 0)  return 'text-profit/80';
  if (n >= -10) return 'text-warning';
  return 'text-loss font-semibold';
}

function probCls(v: number) {
  const n = safeNum(v);
  if (n >= 75) return 'text-profit font-semibold';
  if (n >= 55) return 'text-warning';
  return 'text-loss';
}

function bucketFill(bucket: string) {
  if (bucket.startsWith('<') || bucket.startsWith('-')) return '#EF4444';
  if (bucket === '0 to 10%') return '#94A3B8';
  return '#16A34A';
}

// ── algo color palette ────────────────────────────────────────────────────────
const ALGO_COLOR: Record<string, string> = {
  PPO:      '#C15F3C',
  SAC:      '#6366F1',
  TD3:      '#0D9488',
  A2C:      '#F59E0B',
  DDPG:     '#EC4899',
  Ensemble: '#16A34A',
};

const HORIZON_OPTIONS = [
  { label: '3 Months', days: 63 },
  { label: '6 Months', days: 126 },
  { label: '1 Year',   days: 252 },
  { label: '2 Years',  days: 504 },
];

const SIM_OPTIONS = [
  { label: '500 (Fast)',       value: 500  },
  { label: '1,000',           value: 1000 },
  { label: '3,000 (Precise)', value: 3000 },
];

const PAGE_INFO = {
  title: 'Future Prediction — How This Works',
  sections: [
    { heading: '1. What is Block Bootstrap?', text: 'We sample 20-day return blocks from the last 2 years of NIFTY 50 data and chain them into forward paths. This preserves real autocorrelation (momentum, reversal patterns) unlike a simple Gaussian model.' },
    { heading: '2. The fan chart', text: 'Each colored line is a percentile of the Ensemble strategy\'s outcome across all simulated paths. p50 = median outcome. p5 / p95 = worst and best 5% of scenarios. Break-even line at ×1.0.' },
    { heading: '3. Gray sample paths', text: '10 representative individual scenario paths (spread from worst to best final value) are shown as faint gray lines so you can see how individual paths evolve, not just aggregate bands.' },
    { heading: '4. Algorithm comparison', text: 'All 6 RL strategies (PPO, SAC, TD3, A2C, DDPG, Ensemble) are applied to every simulated path. The table compares expected return, best/worst case, Sharpe ratio, and probability of ending above breakeven.' },
    { heading: '5. Return distribution', text: 'Histogram of annualized Ensemble returns across all N scenarios. Green = positive return, gray = 0–10% (breakeven zone), red = negative return. The % of green scenarios = Probability of Profit.' },
    { heading: '6. Reproducibility', text: 'Seed is fixed (rng=42) — same horizon + same N scenarios always produces the same result. Larger N (3,000) gives smoother bands but takes longer. The 1 Year / 1,000 default is a good balance.' },
  ],
};

// ── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({
  label, value, sub, accent = false,
}: { label: string; value: string; sub: string; accent?: boolean }) {
  return (
    <div className={`rounded-xl border p-4 bg-white shadow-[0_1px_3px_rgba(0,0,0,0.05)] ${
      accent ? 'ring-1 ring-primary/25 border-primary/30' : 'border-border'
    }`}>
      <p className="text-xs text-text-secondary mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${accent ? 'text-primary' : 'text-text'}`}>{value}</p>
      <p className="text-[11px] text-text-muted mt-1">{sub}</p>
    </div>
  );
}

// ── Custom fan-chart tooltip ──────────────────────────────────────────────────

function FanTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const p: Record<string, number> = Object.fromEntries(payload.map((e: any) => [e.dataKey, e.value]));
  return (
    <div className="bg-white border border-border rounded-xl shadow-lg p-3 text-xs">
      <p className="font-semibold text-text mb-1.5">Day {label}</p>
      {(['p95', 'p75', 'p50', 'p25', 'p5'] as const).map(k => (
        p[k] != null && (
          <div key={k} className="flex justify-between gap-4">
            <span className="text-text-muted">{k.toUpperCase()}</span>
            <span className="font-mono font-medium">×{safeNum(p[k]).toFixed(3)}</span>
          </div>
        )
      ))}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function FuturePrediction() {
  const [loading, setLoading]         = useState(false);
  const [data, setData]               = useState<FuturePredictionResponse | null>(null);
  const [error, setError]             = useState<string | null>(null);
  const [horizonDays, setHorizonDays] = useState(252);
  const [nScenarios, setNScenarios]   = useState(1000);

  async function runSimulation() {
    setLoading(true);
    setError(null);
    toast.info(`Simulating ${nScenarios.toLocaleString()} paths × ${horizonDays} days…`);
    try {
      const res = await api.futurePrediction(nScenarios, horizonDays);
      setData(res);
      toast.success('Simulation complete');
    } catch (e: any) {
      const msg = e?.message ?? 'Simulation failed';
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }

  // Fan chart data — percentile bands + 10 sample paths merged by day index
  const fanData = useMemo(() => {
    if (!data) return [];
    return data.percentile_bands.map((band, i) => {
      const pt: Record<string, number> = {
        day: band.day,
        p5: band.p5, p25: band.p25, p50: band.p50, p75: band.p75, p95: band.p95,
      };
      data.sample_paths.forEach((path, pi) => {
        if (path[i]) pt[`path${pi}`] = path[i].value;
      });
      return pt;
    });
  }, [data]);

  const bestAlgo: AlgoFutureStat | undefined = useMemo(() => {
    if (!data) return undefined;
    return [...data.algo_stats].sort((a, b) => b.sharpe - a.sharpe)[0];
  }, [data]);

  return (
    <div className="flex flex-col gap-5 pb-10">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between">
        <PageHeader
          title="Future Prediction"
          subtitle="Forward portfolio simulation · Block Bootstrap · 10 years NIFTY 50 data"
          icon={<TrendingUp size={18} />}
          badge="SIMULATION"
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* ── Config panel ────────────────────────────────────────────────── */}
      <Card>
        <div className="flex flex-wrap items-end gap-6">
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Horizon</label>
            <div className="flex gap-2">
              {HORIZON_OPTIONS.map(opt => (
                <button
                  key={opt.days}
                  onClick={() => setHorizonDays(opt.days)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
                    horizonDays === opt.days
                      ? 'bg-primary text-white border-primary'
                      : 'border-border text-text-secondary hover:border-primary/40 hover:text-text'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Scenarios</label>
            <div className="flex gap-2">
              {SIM_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setNScenarios(opt.value)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors ${
                    nScenarios === opt.value
                      ? 'bg-primary text-white border-primary'
                      : 'border-border text-text-secondary hover:border-primary/40 hover:text-text'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <motion.button
            onClick={runSimulation}
            disabled={loading}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="ml-auto flex items-center gap-2 px-5 py-2 bg-primary text-white rounded-lg text-sm font-semibold shadow-md shadow-primary/30 hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading
              ? <span className="animate-spin w-4 h-4 border-2 border-white/30 border-t-white rounded-full" />
              : <Play size={15} />}
            {loading ? 'Simulating…' : 'Run Simulation'}
          </motion.button>
        </div>
      </Card>

      {/* ── Error banner ─────────────────────────────────────────────────── */}
      {error && (
        <div className="rounded-xl border border-loss/30 bg-loss-light/40 p-4 flex items-center gap-3">
          <AlertCircle size={16} className="text-loss shrink-0" />
          <p className="text-sm text-loss font-medium">{error}</p>
        </div>
      )}

      {/* ── Empty state ───────────────────────────────────────────────────── */}
      {!data && !loading && !error && (
        <Card>
          <div className="flex flex-col items-center justify-center gap-4 py-20 text-center">
            <div className="w-14 h-14 rounded-xl bg-primary-subtle flex items-center justify-center">
              <TrendingUp size={28} className="text-primary" />
            </div>
            <div>
              <p className="text-base font-semibold text-text">Run Your First Simulation</p>
              <p className="text-sm text-text-muted mt-1 max-w-sm mx-auto leading-relaxed">
                Pick a horizon and scenario count, then click "Run Simulation" to generate
                a forward fan chart comparing all 6 RL strategies.
              </p>
            </div>
            <div className="flex gap-3 mt-1 text-[11px] text-text-muted">
              <span className="px-2.5 py-1 rounded-lg bg-bg-card border border-border-light">Block Bootstrap</span>
              <span className="px-2.5 py-1 rounded-lg bg-bg-card border border-border-light">6 RL Strategies</span>
              <span className="px-2.5 py-1 rounded-lg bg-bg-card border border-border-light">Seed rng=42</span>
            </div>
          </div>
        </Card>
      )}

      {/* ── Loading state ─────────────────────────────────────────────────── */}
      {loading && (
        <Card>
          <div className="flex flex-col items-center justify-center gap-4 py-20">
            <span className="animate-spin w-10 h-10 border-4 border-primary/20 border-t-primary rounded-full" />
            <div className="text-center">
              <p className="text-sm font-medium text-text">Running simulation…</p>
              <p className="text-xs text-text-muted mt-0.5">
                {nScenarios.toLocaleString()} scenarios × {horizonDays} trading days
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* ── Results ───────────────────────────────────────────────────────── */}
      {data && !loading && (
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="flex flex-col gap-5"
        >
          {/* Summary stat cards */}
          <motion.div variants={fadeSlideUp} className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              label="Median Return (p50)"
              value={pctFmt(data.median_return)}
              sub="Annualized · 50th percentile"
              accent={safeNum(data.median_return) >= 0}
            />
            <StatCard
              label="Best Case (p95)"
              value={pctFmt(data.best_case_return)}
              sub="Annualized · 95th percentile"
            />
            <StatCard
              label="Worst Case (p5)"
              value={pctFmt(data.worst_case_return)}
              sub="Annualized · 5th percentile"
            />
            <StatCard
              label="Probability of Profit"
              value={`${safeNum(data.probability_profit).toFixed(1)}%`}
              sub={`Out of ${data.n_scenarios.toLocaleString()} scenarios`}
              accent={safeNum(data.probability_profit) >= 60}
            />
          </motion.div>

          {/* Fan chart */}
          <motion.div variants={fadeSlideUp}>
            <Card>
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-semibold text-text text-sm">Ensemble Portfolio — Forward Fan Chart</h3>
                  <p className="text-xs text-text-muted mt-0.5">
                    {data.method} · {data.n_scenarios.toLocaleString()} paths · {data.horizon_days} trading days
                  </p>
                </div>
                <span className="text-[10px] font-semibold bg-primary-subtle text-primary px-2 py-1 rounded-full">
                  P5 / P50 / P95
                </span>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={fanData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 11, fill: '#94A3B8' }}
                    tickFormatter={v => `D${v}`}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#94A3B8' }}
                    tickFormatter={v => `×${Number(v).toFixed(2)}`}
                    domain={['auto', 'auto']}
                    width={54}
                  />
                  <Tooltip content={<FanTooltip />} />
                  <ReferenceLine
                    y={1.0} stroke="#94A3B8" strokeDasharray="4 4" strokeWidth={1.5}
                    label={{ value: 'Break-even', position: 'insideTopRight', fontSize: 10, fill: '#94A3B8' }}
                  />

                  {/* 10 sample paths — faint gray background lines */}
                  {data.sample_paths.map((_, pi) => (
                    <Line
                      key={`path${pi}`}
                      dataKey={`path${pi}`}
                      stroke="#CBD5E1"
                      strokeWidth={0.8}
                      dot={false}
                      isAnimationActive={false}
                      legendType="none"
                    />
                  ))}

                  {/* Percentile bands */}
                  <Line dataKey="p95" stroke="#16A34A" strokeWidth={1.5} dot={false} strokeDasharray="5 3" isAnimationActive={false} name="p95 (Best 5%)" />
                  <Line dataKey="p75" stroke="#22C55E" strokeWidth={1}   dot={false} isAnimationActive={false} name="p75" />
                  <Line dataKey="p50" stroke="#0F172A" strokeWidth={2.5} dot={false} isAnimationActive={false} name="Median" />
                  <Line dataKey="p25" stroke="#F59E0B" strokeWidth={1}   dot={false} isAnimationActive={false} name="p25" />
                  <Line dataKey="p5"  stroke="#EF4444" strokeWidth={1.5} dot={false} strokeDasharray="5 3" isAnimationActive={false} name="p5 (Worst 5%)" />

                  <Legend wrapperStyle={{ fontSize: 11 }} formatter={val => <span className="text-text-secondary">{val}</span>} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </motion.div>

          {/* Algorithm table + Return distribution */}
          <motion.div variants={fadeSlideUp} className="grid lg:grid-cols-5 gap-5">

            {/* Algorithm comparison — 3 cols */}
            <Card className="lg:col-span-3">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-text text-sm">Algorithm Comparison</h3>
                {bestAlgo && (
                  <span className="text-[10px] bg-profit-light text-profit border border-profit/20 px-2 py-0.5 rounded-full font-semibold">
                    Best Sharpe: {bestAlgo.algo}
                  </span>
                )}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      {['Algorithm', 'Expected', 'Best (p95)', 'Worst (p5)', 'Sharpe', 'P(Profit)'].map(h => (
                        <th key={h} className={`pb-2 font-semibold text-text-muted ${h === 'Algorithm' ? 'text-left' : 'text-right'}`}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.algo_stats.map(algo => {
                      const isEns  = algo.algo === 'Ensemble';
                      const isBest = algo.algo === bestAlgo?.algo;
                      return (
                        <tr key={algo.algo} className={`border-b border-border/50 last:border-0 ${isEns ? 'bg-profit-light/20' : ''}`}>
                          <td className="py-2.5">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: ALGO_COLOR[algo.algo] }} />
                              <span className={`font-semibold ${isEns ? 'text-profit' : 'text-text'}`}>{algo.algo}</span>
                              {isEns && (
                                <span className="text-[9px] bg-profit-light text-profit px-1.5 py-0.5 rounded-full font-bold border border-profit/20">★ REC</span>
                              )}
                            </div>
                          </td>
                          <td className={`text-right py-2.5 ${returnCls(algo.expected_return)}`}>{pctFmt(algo.expected_return)}</td>
                          <td className="text-right py-2.5 text-profit/80">{pctFmt(algo.best_case)}</td>
                          <td className="text-right py-2.5 text-loss/80">{pctFmt(algo.worst_case)}</td>
                          <td className={`text-right py-2.5 font-mono ${isBest ? 'text-primary font-semibold' : 'text-text-secondary'}`}>
                            {safeNum(algo.sharpe).toFixed(3)}
                          </td>
                          <td className={`text-right py-2.5 ${probCls(algo.probability_profit)}`}>
                            {safeNum(algo.probability_profit).toFixed(1)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 p-3 bg-profit-light/30 rounded-xl border border-profit/15 text-xs text-text-secondary leading-relaxed">
                <span className="font-semibold text-profit">Why Ensemble?</span>{' '}
                Averages weight predictions from all 5 algorithms, reducing individual model bias.
                Produces more stable allocations — especially during market regime changes when single-strategy models break down.
              </div>
            </Card>

            {/* Return distribution histogram — 2 cols */}
            <Card className="lg:col-span-2">
              <h3 className="font-semibold text-text text-sm mb-0.5">Return Distribution</h3>
              <p className="text-xs text-text-muted mb-4">Ensemble · annualized return across all scenarios</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={data.return_distribution} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
                  <XAxis dataKey="bucket" tick={{ fontSize: 9, fill: '#94A3B8' }} interval={0} angle={-35} textAnchor="end" height={52} />
                  <YAxis tick={{ fontSize: 10, fill: '#94A3B8' }} tickFormatter={v => `${v}%`} />
                  <Tooltip
                    formatter={(v: any, _name: any, props: any) => [`${safeNum(v).toFixed(1)}% of scenarios`, props.payload.bucket]}
                    contentStyle={{ fontSize: 11 }}
                  />
                  <Bar dataKey="pct" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                    {data.return_distribution.map(entry => (
                      <Cell key={entry.bucket} fill={bucketFill(entry.bucket)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-3 flex items-center justify-between px-1">
                <span className="text-xs text-text-muted">Scenarios with positive return</span>
                <span className={`text-sm font-bold ${probCls(data.probability_profit)}`}>
                  {safeNum(data.probability_profit).toFixed(1)}%
                </span>
              </div>
            </Card>
          </motion.div>

          {/* Forward allocation */}
          <motion.div variants={fadeSlideUp}>
            <Card>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="font-semibold text-text text-sm">Forward Allocation (Ensemble)</h3>
                  <p className="text-xs text-text-muted mt-0.5">Top holdings used for forward simulation paths</p>
                </div>
                <span className="text-[10px] bg-primary-subtle text-primary px-2 py-1 rounded-full font-semibold">
                  TOP {data.forward_allocation.length} STOCKS
                </span>
              </div>

              {/* Grid of stock tiles */}
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
                {data.forward_allocation.slice(0, 16).map((stock, i) => (
                  <div
                    key={stock.ticker}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-bg-card border border-border-light"
                  >
                    <span className="text-xs text-text-muted w-5 text-right shrink-0">{i + 1}.</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-text truncate">{stock.ticker.replace('.NS', '')}</p>
                      <p className="text-[10px] text-text-muted truncate">{stock.sector}</p>
                    </div>
                    <span className="text-sm font-bold text-primary shrink-0">{safeNum(stock.weight).toFixed(1)}%</span>
                  </div>
                ))}
              </div>

              {/* Weight bar chart for top 10 */}
              <div className="space-y-1.5">
                {data.forward_allocation.slice(0, 10).map(stock => (
                  <div key={stock.ticker} className="flex items-center gap-3">
                    <span className="text-xs text-text-muted w-20 text-right shrink-0 font-mono">
                      {stock.ticker.replace('.NS', '')}
                    </span>
                    <div className="flex-1 h-1.5 bg-bg-card rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full"
                        style={{ width: `${Math.min(safeNum(stock.weight) / 15 * 100, 100)}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono text-text-secondary w-10 text-right shrink-0">
                      {safeNum(stock.weight).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </motion.div>

          {/* Metadata footer */}
          <motion.div variants={fadeSlideUp}>
            <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-text-muted px-1">
              <span>Method: <span className="text-text">{data.method}</span></span>
              <span>Seed data: <span className="text-text">{data.seed_days} trading days</span></span>
              <span>Block size: <span className="text-text">20 days</span></span>
              <span>Horizon: <span className="text-text">{data.horizon_days} trading days</span></span>
              <span>Scenarios: <span className="text-text">{data.n_scenarios.toLocaleString()}</span></span>
              <span className="ml-auto text-text-muted/50 italic">rng seed = 42 — deterministic</span>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* ── Portfolio Integration callout ─────────────────────────────── */}
      <div className="rounded-xl border border-primary/20 bg-primary/[0.03] px-4 py-3">
        <div className="flex items-start gap-3">
          <TrendingUp size={15} className="text-primary shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-[11px] font-bold text-primary uppercase tracking-wide mb-1.5">Where This Fits in the System</p>
            <div className="grid grid-cols-4 gap-2 text-[10px] mb-2.5">
              {[
                { label: 'RL Agents', sub: 'Train on history', active: false },
                { label: 'Ensemble', sub: 'Best-of-5 weights', active: false },
                { label: 'Block Bootstrap', sub: 'Simulate futures', active: true },
                { label: 'Risk Report', sub: 'Sharpe · P(profit)', active: false },
              ].map(item => (
                <div key={item.label}
                  className={`rounded-lg border px-2 py-1.5 text-center ${
                    item.active
                      ? 'border-primary/30 bg-primary/8 text-primary'
                      : 'border-border-light bg-white text-text-secondary'
                  }`}>
                  <p className="font-semibold leading-none mb-0.5">{item.label}</p>
                  <p className="opacity-60">{item.sub}</p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-text-secondary">
              This tab takes the <span className="font-semibold text-text">Ensemble portfolio weights</span> (already optimized by RL training)
              and applies them to thousands of statistically plausible future return paths sampled from real NIFTY 50 data.
              The result: given the RL agent's current allocation, what range of outcomes should an investor expect?
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
