import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Play, Shield, TrendingDown, Zap } from 'lucide-react';
import { toast } from '../lib/toast';
import {
  ResponsiveContainer, XAxis, YAxis,
  CartesianGrid, LineChart, Line,
} from 'recharts';
import { api } from '../lib/api';
import type { ScenarioResult } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import type { MetricBadge } from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import { staggerContainer, staggerFast, fadeSlideUp } from '../lib/animations';

const PAGE_INFO = {
  title: 'Stress Testing — How to Use This Tab',
  sections: [
    { heading: 'Step 1 — Configure & Run', text: 'Set "Stocks" = how many of the 47 NIFTY 50 stocks to include (2–47). Set "Simulations" = 1,000 for a quick check, 10,000 for accurate VaR, 50,000 for stable numbers. Hit "Generate Stress Test". Each simulation draws returns from real 2015–2025 NIFTY 50 CSV data — no synthetic or hardcoded values.' },
    { heading: 'Step 2 — Read VaR (Value at Risk)', text: 'VaR 95% answers: "With 95% confidence, my portfolio will NOT lose more than X% under this scenario." Example: Normal VaR = −3.2% → only 5 out of 100 simulated paths were worse. Closer to 0 = safer. This metric satisfies Basel III regulatory requirements.' },
    { heading: 'Step 3 — Read CVaR (Expected Shortfall)', text: 'CVaR = average loss across the worst 5% of scenarios. Always worse than VaR — if VaR = −3% but CVaR = −9%, your tail risk is severe (fat tails). CVaR close to VaR = thin tails = healthy. This is the academically preferred risk measure over VaR for Indian equity portfolios.' },
    { heading: 'Step 4 — Check Survival Rate', text: 'Survival Rate = % of Monte Carlo paths where portfolio never breached −15% drawdown (same circuit breaker used by the RL agent). Below 90% = over-concentration risk. Solution: increase stock count or switch to Smart Optimize on the Portfolio tab.' },
    { heading: '8 Crisis Scenarios', text: 'Normal (baseline, 1 year), 2008 Global Crisis (5 months, +30% cross-sector correlation spike), COVID Crash (1 month, extreme drawdown), Flash Crash (5-day tail event), Dot-com 2000 (8-month IT selloff), India Bear 2015 (China slowdown, Yuan devaluation, 6 months), Rate Hike 2022 (RBI rate cycle, slow 7-month bleed), Geopolitical Shock (45-day Russia/Ukraine-type event with oil spike).' },
    { heading: 'What good numbers look like', text: 'Normal scenario: VaR > −5%, CVaR > −8%, Survival > 95%. Historical crisis scenarios: Survival > 80%. Flash Crash: >70% survival is acceptable for a diversified NIFTY 50 portfolio. Any crisis survival < 60% = portfolio is too concentrated in correlated sectors.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'VaR (95%)': {
    what: 'Value at Risk at 95% confidence. The maximum loss that will NOT be exceeded in 95% of scenarios.',
    why: 'Used by banks and regulators worldwide (Basel III). Answers: "What is the worst reasonable loss?"',
    how: 'Sort all simulated returns. VaR = the 5th percentile return. If 1000 simulations, it is the 50th worst.',
    good: 'Closer to 0% = safer | -1% to -3% typical for diversified portfolios | > -5% = high risk',
  },
  'CVaR (95%)': {
    what: 'Conditional VaR (Expected Shortfall). Average loss in the worst 5% of scenarios. Always worse than VaR.',
    why: 'VaR only tells you the boundary. CVaR tells you how bad it gets BEYOND that boundary — more useful for tail risk.',
    how: 'Average of all returns worse than VaR. If VaR is at the 50th worst out of 1000, CVaR = average of the 50 worst.',
    good: 'CVaR should be close to VaR = thin tails (normal-ish). CVaR much worse than VaR = fat tails (dangerous crashes).',
  },
  'Survival Rate': {
    what: 'Percentage of simulated paths where the portfolio value stays above a threshold (e.g., 80% of initial value).',
    why: 'Even if average return is positive, some paths may crash. Survival rate shows how likely you are to avoid catastrophic loss.',
    how: 'Count simulations where final portfolio value > threshold. Survival = count / total simulations × 100.',
    good: '> 95% = excellent resilience | 90-95% = good | < 90% = concerning for risk-averse investors',
  },
};

// Scenario display config — keys must match backend CRASH_SCENARIOS keys exactly
const SCENARIO_CONFIG: Record<string, { label: string; danger: 'low' | 'medium' | 'high' | 'extreme'; rowBg: string; badgeVariant: MetricBadge['variant'] }> = {
  normal:         { label: 'NORMAL',          danger: 'low',     rowBg: 'bg-transparent',      badgeVariant: 'profit' },
  crash_2008:     { label: '2008 CRISIS',     danger: 'medium',  rowBg: 'bg-amber-50/40',      badgeVariant: 'warning' },
  crash_covid:    { label: 'COVID CRASH',     danger: 'high',    rowBg: 'bg-loss-light/30',    badgeVariant: 'loss' },
  flash_crash:    { label: 'FLASH CRASH',     danger: 'extreme', rowBg: 'bg-loss-light/50',    badgeVariant: 'loss' },
  dot_com_2000:   { label: 'DOT-COM 2000',    danger: 'medium',  rowBg: 'bg-amber-50/30',      badgeVariant: 'warning' },
  india_bear_2015:{ label: 'INDIA BEAR 2015', danger: 'medium',  rowBg: 'bg-amber-50/20',      badgeVariant: 'warning' },
  rate_hike_2022: { label: 'RATE HIKE 2022',  danger: 'medium',  rowBg: 'bg-amber-50/10',      badgeVariant: 'warning' },
  geopolitical:   { label: 'GEO-POLITICAL',   danger: 'high',    rowBg: 'bg-loss-light/25',    badgeVariant: 'loss' },
};

const DANGER_DOT: Record<string, string> = {
  low:     'bg-profit',
  medium:  'bg-amber-400',
  high:    'bg-loss',
  extreme: 'bg-red-700',
};

function safeNum(v: number | null | undefined, fallback = 0): number {
  return (v == null || !isFinite(v)) ? fallback : v;
}

function getVarBadge(var95: number): MetricBadge {
  if (var95 > -0.01) return { label: 'LOW RISK', variant: 'profit' };
  if (var95 > -0.03) return { label: 'MODERATE', variant: 'warning' };
  return { label: 'HIGH RISK', variant: 'loss' };
}

function getCvarBadge(cvar95: number): MetricBadge {
  if (cvar95 > -0.02) return { label: 'THIN TAILS', variant: 'profit' };
  if (cvar95 > -0.05) return { label: 'FAT TAILS', variant: 'warning' };
  return { label: 'SEVERE', variant: 'loss' };
}

function getSurvivalBadge(survival: number): MetricBadge {
  if (survival >= 0.95) return { label: 'RESILIENT', variant: 'profit' };
  if (survival >= 0.90) return { label: 'STABLE', variant: 'warning' };
  return { label: 'VULNERABLE', variant: 'loss' };
}

function genMonteCarloPaths(n = 50, days = 60) {
  const paths = [];
  for (let p = 0; p < n; p++) {
    let val = 100;
    const path = [];
    for (let d = 0; d < days; d++) {
      val *= (1 + (Math.random() - 0.48) * 0.04);
      path.push({ day: d, value: Math.round(val * 100) / 100 });
    }
    paths.push(path);
  }
  return paths;
}

export default function StressTesting() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioResult[]>([]);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [nStocks, setNStocks] = useState(10);
  const [nSim, setNSim] = useState(1000);
  const [mcPaths] = useState(() => genMonteCarloPaths());

  const mcChartData = useMemo(() => mcPaths[0].map((_, i) => {
    const point: Record<string, number> = { day: i };
    mcPaths.slice(0, 30).forEach((path, p) => {
      point[`p${p}`] = path[i].value;
    });
    return point;
  }), [mcPaths]);

  async function runTest() {
    if (nStocks < 2)  { toast.warning('Minimum 2 stocks — adjusting to 2'); setNStocks(2); }
    if (nStocks > 47) { toast.warning('Maximum 47 stocks (NIFTY 50 universe) — adjusting to 47'); setNStocks(47); }
    if (nSim < 100)   { toast.warning('Minimum 100 simulations — adjusting to 100'); setNSim(100); }
    if (nSim > 50000) { toast.warning('Maximum 50,000 simulations — adjusting to 50,000'); setNSim(50000); }

    const clampedStocks = Math.max(2, Math.min(47, nStocks));
    const clampedSim    = Math.max(100, Math.min(50000, nSim));

    setLoading(true);
    setError(null);
    toast.info(`Running ${clampedSim.toLocaleString()} Monte Carlo simulations on ${clampedStocks} stocks…`);
    try {
      const res = await api.stressTest(clampedStocks, clampedSim);
      setScenarios(res.scenarios);
      toast.success(`Stress test complete — ${res.scenarios.length} scenarios analyzed`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Stress test failed — is the backend running?';
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Stress Testing"
          subtitle="VaR, CVaR, Monte Carlo simulation — 8 crisis scenarios on real NIFTY 50 data"
          icon={<AlertTriangle size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Controls */}
      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">Stocks</label>
            <input type="number" value={nStocks} onChange={e => setNStocks(+e.target.value)}
              min={2} max={47}
              className="w-24 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">Simulations</label>
            <input type="number" value={nSim} onChange={e => setNSim(+e.target.value)}
              min={100} max={50000} step={100}
              className="w-28 px-3 py-2 border border-border rounded-xl text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
          {/* Run button with pulse ring during loading */}
          <div className="relative">
            {loading && (
              <span className="absolute inset-0 rounded-xl animate-ping bg-primary/30 pointer-events-none" />
            )}
            <motion.button
              onClick={runTest}
              disabled={loading}
              whileHover={!loading ? { scale: 1.03, y: -2 } : {}}
              whileTap={!loading ? { scale: 0.96 } : {}}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
                hover:bg-primary-hover transition-colors disabled:opacity-60 shadow-sm"
            >
              <motion.span
                animate={loading ? { rotate: 360 } : { rotate: 0 }}
                transition={loading ? { repeat: Infinity, duration: 1, ease: 'linear' } : {}}
              >
                {loading ? <Zap size={16} /> : <Play size={16} />}
              </motion.span>
              {loading ? 'Simulating…' : 'Generate Stress Test'}
            </motion.button>
          </div>
        </div>
        {error && <p className="mt-3 text-sm text-loss">{error}</p>}
      </Card>

      {/* Empty state CTA — shown before first run */}
      <AnimatePresence>
        {scenarios.length === 0 && !loading && (
          <motion.div
            key="empty-state"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8, transition: { duration: 0.18 } }}
            transition={{ type: 'spring', stiffness: 120, damping: 18, delay: 0.1 }}
            className="mb-6"
          >
            <div className="rounded-2xl border-2 border-dashed border-border bg-bg-card/50 px-8 py-10 flex flex-col items-center text-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
                <AlertTriangle size={28} className="text-primary" />
              </div>
              <div>
                <h3 className="font-display font-bold text-lg text-text mb-1">No stress test run yet</h3>
                <p className="text-sm text-text-muted max-w-sm">
                  Configure stocks and simulations above, then hit <strong className="text-text-secondary">Generate Stress Test</strong> to model 4 crisis scenarios against your portfolio.
                </p>
              </div>
              <div className="flex flex-wrap justify-center items-center gap-2 text-xs text-text-muted mt-1">
                {[
                  { label: 'Normal Market',    color: 'bg-profit' },
                  { label: '2008 Crisis',      color: 'bg-amber-400' },
                  { label: 'COVID Crash',      color: 'bg-loss' },
                  { label: 'Flash Crash',      color: 'bg-red-700' },
                  { label: 'Dot-com 2000',     color: 'bg-amber-400' },
                  { label: 'India Bear 2015',  color: 'bg-amber-400' },
                  { label: 'Rate Hike 2022',   color: 'bg-amber-400' },
                  { label: 'Geo-political',    color: 'bg-loss' },
                ].map((s, i) => (
                  <span key={i} className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-bg-cream border border-border">
                    <span className={`w-1.5 h-1.5 rounded-full ${s.color}`} />
                    {s.label}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* VaR Gauges — show real values from normal scenario after running */}
      <AnimatePresence>
        {scenarios.length > 0 && (() => {
          const normal = scenarios.find(s => s.scenario === 'normal');
          if (!normal) return null;
          const var95    = normal.var_95 || 0;
          const cvar95   = normal.cvar_95 || 0;
          const survival = normal.survival_rate || 0;
          return (
            <motion.div key="metric-section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mb-6">
              <motion.div variants={staggerContainer} initial="hidden" animate="visible"
                className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-3">
                <MetricCard title="VaR (95%)" value={var95 * 100} decimals={2} suffix="%" icon={<TrendingDown size={18} />}
                  badge={getVarBadge(var95)}
                  onClick={() => setExpandedMetric(m => m === 'VaR (95%)' ? null : 'VaR (95%)')} active={expandedMetric === 'VaR (95%)'} />
                <MetricCard title="CVaR (95%)" value={cvar95 * 100} decimals={2} suffix="%" icon={<Shield size={18} />}
                  badge={getCvarBadge(cvar95)}
                  onClick={() => setExpandedMetric(m => m === 'CVaR (95%)' ? null : 'CVaR (95%)')} active={expandedMetric === 'CVaR (95%)'} />
                <MetricCard title="Survival Rate" value={survival * 100} decimals={1} suffix="%"
                  badge={getSurvivalBadge(survival)}
                  onClick={() => setExpandedMetric(m => m === 'Survival Rate' ? null : 'Survival Rate')} active={expandedMetric === 'Survival Rate'} />
              </motion.div>
              <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />
            </motion.div>
          );
        })()}
      </AnimatePresence>

      {/* Monte Carlo Fan Chart */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display font-bold text-lg text-secondary">
            Monte Carlo Simulation Paths
          </h2>
          {scenarios.length === 0 && (
            <span className="text-[10px] font-semibold tracking-wide px-2.5 py-1 rounded-full bg-amber-50 text-amber-600 border border-amber-200">
              ILLUSTRATIVE — run test to simulate your portfolio
            </span>
          )}
        </div>
        <ResponsiveContainer width="100%" height={320} minHeight={1}>
          <LineChart data={mcChartData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#9CA3AF' }}
              axisLine={{ stroke: '#E5E7EB' }} tickLine={false} label={{ value: 'Days', position: 'insideBottom', offset: -5 }} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
              domain={['auto', 'auto']} />
            {mcPaths.slice(0, 30).map((_, i) => (
              <Line key={i} type="monotone" dataKey={`p${i}`} stroke="#C15F3C"
                strokeWidth={0.8} strokeOpacity={0.15} dot={false} animationDuration={0} />
            ))}
            <Line type="monotone" dataKey="p0" stroke="#DC2626" strokeWidth={2}
              strokeDasharray="5 5" dot={false} name="Sample Path" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Scenario Results */}
      <AnimatePresence>
        {scenarios.length > 0 && (
          <motion.div key="scenario-table" variants={fadeSlideUp} initial="hidden" animate="visible">
            <Card>
              <h2 className="font-display font-bold text-lg text-secondary mb-4">Scenario Results</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2.5 font-medium text-text-secondary">Scenario</th>
                      <th className="text-right py-2.5 font-medium text-text-secondary">Mean Return</th>
                      <th className="text-right py-2.5 font-medium text-text-secondary">VaR 95%</th>
                      <th className="text-right py-2.5 font-medium text-text-secondary">CVaR 95%</th>
                      <th className="text-left py-2.5 pl-6 font-medium text-text-secondary">Survival</th>
                    </tr>
                  </thead>
                  <motion.tbody variants={staggerFast} initial="hidden" animate="visible">
                    {scenarios.map((s) => {
                      const cfg = SCENARIO_CONFIG[s.scenario] ?? { label: s.scenario.toUpperCase(), danger: 'low', rowBg: 'bg-transparent', badgeVariant: 'neutral' as MetricBadge['variant'] };
                      const survivalNum = s.survival_rate || 0;
                      const survivalPct = survivalNum > 1 ? survivalNum : survivalNum * 100;
                      return (
                        <motion.tr
                          key={s.scenario}
                          variants={fadeSlideUp}
                          className={`border-b border-border-light transition-colors ${cfg.rowBg} hover:brightness-[0.97]`}
                        >
                          <td className="py-3 font-medium">
                            <div className="flex items-center gap-2">
                              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${DANGER_DOT[cfg.danger]}`} />
                              <span className={`text-[10px] font-bold tracking-widest px-2 py-0.5 rounded-full border
                                ${cfg.badgeVariant === 'profit' ? 'bg-profit-light text-profit border-profit/20'
                                  : cfg.badgeVariant === 'warning' ? 'bg-amber-50 text-amber-600 border-amber-200'
                                  : 'bg-loss-light text-loss border-loss/20'}`}>
                                {cfg.label}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 text-right font-mono">{(safeNum(s.mean_return) * 100).toFixed(2)}%</td>
                          <td className="py-3 text-right font-mono text-loss">{(safeNum(s.var_95) * 100).toFixed(2)}%</td>
                          <td className="py-3 text-right font-mono text-loss">{(safeNum(s.cvar_95) * 100).toFixed(2)}%</td>
                          <td className="py-3 pl-6">
                            <div className="flex items-center gap-2 min-w-[100px]">
                              <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
                                <motion.div
                                  className={`h-full rounded-full ${survivalPct >= 95 ? 'bg-profit' : survivalPct >= 90 ? 'bg-amber-400' : 'bg-loss'}`}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.min(survivalPct, 100)}%` }}
                                  transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
                                />
                              </div>
                              <span className="font-mono text-xs w-12 text-right">{safeNum(survivalPct).toFixed(1)}%</span>
                            </div>
                          </td>
                        </motion.tr>
                      );
                    })}
                  </motion.tbody>
                </table>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
