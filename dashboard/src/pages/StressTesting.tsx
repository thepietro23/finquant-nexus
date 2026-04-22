import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Play, Shield, TrendingDown } from 'lucide-react';
import { toast } from '../lib/toast';
import {
  ResponsiveContainer, XAxis, YAxis,
  CartesianGrid, LineChart, Line,
} from 'recharts';
import { api } from '../lib/api';
import type { ScenarioResult } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import Badge from '../components/ui/Badge';
import { staggerContainer } from '../lib/animations';

const PAGE_INFO = {
  title: 'Stress Testing — What Does This Page Show?',
  sections: [
    { heading: 'What is stress testing?', text: 'Tests your portfolio against extreme market conditions using Monte Carlo simulation. Generates thousands of possible future scenarios to estimate risk metrics like VaR and CVaR.' },
    { heading: 'Monte Carlo simulation', text: 'Randomly generates thousands of future price paths based on historical statistics (mean, volatility, correlations). Like rolling dice 10,000 times to see what could happen.' },
    { heading: '4 crash scenarios', text: 'Normal (typical market), 2008 Crisis (3.5% daily vol, +30% correlation), COVID Crash (5% vol, +40% corr), Flash Crash (8% vol, 5 days). Each models a real historical pattern.' },
    { heading: 'VaR (Value at Risk)', text: '95% VaR = "with 95% confidence, the portfolio will NOT lose more than this amount." It is a threshold — the boundary of the worst 5% outcomes.' },
    { heading: 'CVaR (Conditional VaR)', text: 'Also called Expected Shortfall. "If you ARE in the worst 5%, what is the average loss?" CVaR is always worse than VaR and is considered more conservative.' },
    { heading: 'Survival rate', text: 'Percentage of simulations where the portfolio stays above a minimum threshold (does not get wiped out). Higher = more resilient portfolio.' },
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

// Mock Monte Carlo paths
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

  // Build chart data from first 30 paths — memoized since mcPaths never changes
  const mcChartData = useMemo(() => mcPaths[0].map((_, i) => {
    const point: Record<string, number> = { day: i };
    mcPaths.slice(0, 30).forEach((path, p) => {
      point[`p${p}`] = path[i].value;
    });
    return point;
  }), [mcPaths]);

  async function runTest() {
    setLoading(true);
    setError(null);
    toast.info(`Running ${nSim.toLocaleString()} Monte Carlo simulations on ${nStocks} stocks…`);
    try {
      const res = await api.stressTest(
        Math.max(2, Math.min(47, nStocks)),
        Math.max(100, Math.min(50000, nSim)),
      );
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
          subtitle="VaR, CVaR, Monte Carlo simulation — 4 crash scenarios"
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
          <button onClick={runTest} disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary text-white rounded-xl text-sm font-medium
              hover:bg-primary-hover transition-colors disabled:opacity-50">
            <Play size={16} />
            {loading ? 'Running...' : 'Generate Stress Test'}
          </button>
        </div>
        {error && <p className="mt-3 text-sm text-loss">{error}</p>}
      </Card>

      {/* VaR Gauges — show real values from normal scenario after running */}
      {scenarios.length > 0 && (() => {
        const normal = scenarios.find(s => s.scenario === 'normal');
        if (!normal) return null;
        const var95 = parseFloat(normal.var_95) || 0;
        const cvar95 = parseFloat(normal.cvar_95) || 0;
        const survival = parseFloat(normal.survival_rate) || 0;
        return (
          <>
            <motion.div variants={staggerContainer} initial="hidden" animate="visible"
              className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-2">
              <MetricCard title="VaR (95%)" value={var95 * 100} decimals={2} suffix="%" icon={<TrendingDown size={18} />}
                onClick={() => setExpandedMetric(m => m === 'VaR (95%)' ? null : 'VaR (95%)')} active={expandedMetric === 'VaR (95%)'} />
              <MetricCard title="CVaR (95%)" value={cvar95 * 100} decimals={2} suffix="%" icon={<Shield size={18} />}
                onClick={() => setExpandedMetric(m => m === 'CVaR (95%)' ? null : 'CVaR (95%)')} active={expandedMetric === 'CVaR (95%)'} />
              <MetricCard title="Survival Rate" value={survival * 100} decimals={1} suffix="%"
                onClick={() => setExpandedMetric(m => m === 'Survival Rate' ? null : 'Survival Rate')} active={expandedMetric === 'Survival Rate'} />
            </motion.div>
            <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />
          </>
        );
      })()}

      {/* Monte Carlo Fan Chart */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">
          Monte Carlo Simulation Paths
        </h2>
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
            {/* VaR line */}
            <Line type="monotone" dataKey="p0" stroke="#DC2626" strokeWidth={2}
              strokeDasharray="5 5" dot={false} name="Sample Path" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Scenario Results */}
      {scenarios.length > 0 && (
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
                  <th className="text-right py-2.5 font-medium text-text-secondary">Survival</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map(s => (
                  <tr key={s.scenario} className="border-b border-border-light hover:bg-bg-card transition-colors">
                    <td className="py-3 font-medium">
                      <Badge variant={s.scenario === 'normal' ? 'profit' : 'loss'}>
                        {s.scenario.replace(/_/g, ' ').toUpperCase()}
                      </Badge>
                    </td>
                    <td className="py-3 text-right font-mono">{s.mean_return}</td>
                    <td className="py-3 text-right font-mono text-loss">{s.var_95}</td>
                    <td className="py-3 text-right font-mono text-loss">{s.cvar_95}</td>
                    <td className="py-3 text-right font-mono">{s.survival_rate}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
