import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Network, TrendingUp, GitBranch, Zap, BarChart3 } from 'lucide-react';
import {
  ResponsiveContainer, XAxis, YAxis, CartesianGrid,
  BarChart, Bar, Tooltip, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { GNNSummaryResponse } from '../lib/api';
import Card from '../components/ui/Card';
import MetricCard from '../components/ui/MetricCard';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import MetricInfoPanel from '../components/ui/MetricInfoPanel';
import Badge from '../components/ui/Badge';
import { staggerContainer, fadeSlideUp } from '../lib/animations';

const PAGE_INFO = {
  title: 'GNN Insights — What Does This Page Show?',
  sections: [
    { heading: 'What is T-GAT?', text: 'Temporal Graph Attention Network — a GNN that learns relationships between stocks. Each stock is a node, connections (edges) represent sector, supply chain, and correlation relationships.' },
    { heading: 'Why use a Graph Neural Network?', text: 'Traditional models treat each stock independently. But stocks are interconnected — when TCS drops, Infosys often follows. GNN captures these inter-stock dependencies for better predictions.' },
    { heading: 'Attention mechanism', text: 'Not all connections are equally important. Attention weights learn which neighbors matter most for predicting each stock\'s behavior. The heatmap shows real correlation-based attention weights.' },
    { heading: '3 types of edges', text: 'Sector (same industry), Supply Chain (business relationships like Tata Steel → Maruti), Correlation (price co-movement > 0.4). Together they form a rich stock network.' },
    { heading: 'Graph density', text: 'Density = edges / max possible edges. 0 = no connections, 1 = fully connected. Higher density = more information flow. Our graph is sparse (density ~0.15) — this is typical and good for GNNs.' },
    { heading: 'How is it used?', text: 'T-GAT produces 64-dimensional embeddings per stock. These embeddings are fed to the RL agent as features, giving it "awareness" of inter-stock dynamics and sector structure.' },
  ],
};

const METRIC_DETAILS: Record<string, { what: string; why: string; how: string; good: string }> = {
  'Total Nodes': {
    what: 'Number of stocks in the graph. Each NIFTY 50 stock is represented as a node.',
    why: 'More nodes = richer network. Our graph covers the most liquid Indian large-cap stocks.',
    how: 'Count of all stocks with valid price data in our dataset. Some NIFTY 50 stocks may be excluded if data is insufficient.',
    good: '44-50 nodes is ideal for NIFTY 50. Fewer means missing stocks. Too many means including illiquid stocks.',
  },
  'Total Edges': {
    what: 'Number of connections between stocks. Each edge represents a relationship (sector, supply chain, or correlation).',
    why: 'Edges carry information between stocks in the GNN. More edges = more information flow, but too many = noise.',
    how: 'Sum of sector edges (same sector), supply chain edges (business relationships), and correlation edges (|corr| > 0.4).',
    good: '100-200 edges is healthy for 44 stocks. This gives avg degree 4-9, similar to real financial networks.',
  },
  'Graph Density': {
    what: 'Ratio of actual edges to maximum possible edges. Measures how "connected" the network is.',
    why: 'Dense graphs share too much information (everything affects everything). Sparse graphs capture meaningful relationships only.',
    how: 'Density = 2 × edges / (nodes × (nodes − 1)). For undirected graphs, max edges = n(n−1)/2.',
    good: '0.05-0.20 is typical for financial networks. >0.3 = too dense (noisy). <0.05 = too sparse (missing relationships).',
  },
  'Avg Degree': {
    what: 'Average number of connections per stock. "How many neighbors does a typical stock have?"',
    why: 'High degree stocks are network hubs — they influence many others. Banks and large IT firms tend to be hubs.',
    how: 'Sum of all node degrees / number of nodes. Degree = number of edges connected to a node.',
    good: '4-10 is healthy. <3 = disconnected graph. >15 = almost fully connected (GNN becomes like a regular neural net).',
  },
};

const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Others': '#9CA3AF', 'Infra': '#14B8A6',
  'Unknown': '#9CA3AF',
};

const EDGE_TYPE_COLORS: Record<string, string> = {
  sector: '#C15F3C',
  supply: '#6366F1',
  correlation: '#0D9488',
};

export default function GnnInsights() {
  const [data, setData] = useState<GNNSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; val: number } | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    api.gnnSummary()
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e instanceof Error ? e.message : 'Failed to load GNN data'); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
    </div>
  );
  if (error || !data) return (
    <Card className="text-center py-12">
      <p className="text-loss font-medium">{error || 'No data'}</p>
      <p className="text-sm text-text-secondary mt-1">Is the backend running?</p>
    </Card>
  );

  // Degree distribution chart data
  const degreeChartData = Object.entries(data.degree_distribution)
    .map(([deg, count]) => ({ degree: Number(deg), count }))
    .sort((a, b) => a.degree - b.degree);

  // Selected node info
  const selectedNodeData = selectedNode ? data.nodes.find(n => n.ticker === selectedNode) : null;
  const selectedNodeEdges = selectedNode
    ? data.edges.filter(e => e.source === selectedNode || e.target === selectedNode)
    : [];

  // SVG mini-graph: top-15 stocks by degree, positioned in a circle
  const top15Nodes = [...data.nodes].sort((a, b) => b.degree - a.degree).slice(0, 15);
  const top15Set = new Set(top15Nodes.map(n => n.ticker));
  const top15Edges = data.edges.filter(e => top15Set.has(e.source) && top15Set.has(e.target));

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="GNN Insights"
          subtitle="Temporal Graph Attention Network — real correlation data, 3 edge types"
          icon={<Network size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Metric Cards */}
      <motion.div variants={staggerContainer} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
        <MetricCard title="Total Nodes" value={data.n_nodes} decimals={0} icon={<Network size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Total Nodes' ? null : 'Total Nodes')} active={expandedMetric === 'Total Nodes'} />
        <MetricCard title="Total Edges" value={data.n_edges} decimals={0} icon={<GitBranch size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Total Edges' ? null : 'Total Edges')} active={expandedMetric === 'Total Edges'} />
        <MetricCard title="Graph Density" value={data.density} decimals={4} icon={<Zap size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Graph Density' ? null : 'Graph Density')} active={expandedMetric === 'Graph Density'} />
        <MetricCard title="Avg Degree" value={data.avg_degree} decimals={1} icon={<BarChart3 size={18} />}
          onClick={() => setExpandedMetric(m => m === 'Avg Degree' ? null : 'Avg Degree')} active={expandedMetric === 'Avg Degree'} />
      </motion.div>
      <MetricInfoPanel expandedMetric={expandedMetric} onClose={() => setExpandedMetric(null)} details={METRIC_DETAILS} />

      {/* Edge Type Breakdown */}
      <Card className="mb-6">
        <h2 className="font-display font-bold text-lg text-secondary mb-4">Edge Type Breakdown</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { type: 'Sector', color: EDGE_TYPE_COLORS.sector, count: data.sector_edges,
              desc: 'Same-sector stocks connected bidirectionally. Banking stocks connect to other banks, IT connects to IT.' },
            { type: 'Supply Chain', color: EDGE_TYPE_COLORS.supply, count: data.supply_chain_edges,
              desc: 'Business relationships (e.g., TATASTEEL → MARUTI for steel supply, BHARTIARTL → TCS for telecom-IT).' },
            { type: 'Correlation', color: EDGE_TYPE_COLORS.correlation, count: data.correlation_edges,
              desc: 'Cross-sector pairs with |60-day rolling correlation| > 0.4. These are dynamic — they change as market conditions shift.' },
          ].map(e => (
            <div key={e.type} className="flex items-start gap-3 p-4 rounded-xl bg-bg-card border border-border-light">
              <div className="w-4 h-4 rounded-full mt-0.5 shrink-0" style={{ backgroundColor: e.color }} />
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-semibold text-sm text-text">{e.type}</p>
                  <span className="font-mono text-xs text-text-muted bg-white px-2 py-0.5 rounded-md border border-border">
                    {e.count}
                  </span>
                </div>
                <p className="text-xs text-text-secondary leading-relaxed">{e.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Interactive Mini-Graph (Top 15 by degree) */}
        <Card className="min-h-[420px] flex flex-col">
          <h2 className="font-display font-bold text-lg text-secondary mb-1">
            Stock Network <span className="text-xs text-text-muted font-normal">(top 15 by connections)</span>
          </h2>
          <p className="text-xs text-text-secondary mb-3">Click a node to see its connections and details</p>
          <div className="flex-1 relative bg-bg-card rounded-xl overflow-hidden">
            <svg viewBox="0 0 500 420" className="w-full h-full">
              {/* Edges */}
              {top15Edges.map((e, idx) => {
                const si = top15Nodes.findIndex(n => n.ticker === e.source);
                const ti = top15Nodes.findIndex(n => n.ticker === e.target);
                if (si < 0 || ti < 0) return null;
                const angle1 = (si / top15Nodes.length) * Math.PI * 2 - Math.PI / 2;
                const angle2 = (ti / top15Nodes.length) * Math.PI * 2 - Math.PI / 2;
                const cx = 250, cy = 210, r = 150;
                const x1 = cx + Math.cos(angle1) * r, y1 = cy + Math.sin(angle1) * r;
                const x2 = cx + Math.cos(angle2) * r, y2 = cy + Math.sin(angle2) * r;
                const isHighlighted = selectedNode && (e.source === selectedNode || e.target === selectedNode);
                const dimmed = selectedNode && !isHighlighted;
                return (
                  <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={EDGE_TYPE_COLORS[e.type as keyof typeof EDGE_TYPE_COLORS] || '#E5E7EB'}
                    strokeWidth={isHighlighted ? 2.5 : 1}
                    opacity={dimmed ? 0.06 : isHighlighted ? 0.7 : 0.2} />
                );
              })}
              {/* Nodes */}
              {top15Nodes.map((n, i) => {
                const angle = (i / top15Nodes.length) * Math.PI * 2 - Math.PI / 2;
                const cx = 250 + Math.cos(angle) * 150;
                const cy = 210 + Math.sin(angle) * 150;
                const radius = 12 + n.degree * 0.8;
                const color = SECTOR_COLORS[n.sector] || '#9CA3AF';
                const isSelected = n.ticker === selectedNode;
                const isConnected = selectedNode
                  ? selectedNodeEdges.some(e => e.source === n.ticker || e.target === n.ticker)
                  : false;
                const dimmed = selectedNode && !isSelected && !isConnected;
                return (
                  <g key={n.ticker} className="cursor-pointer"
                    onClick={() => setSelectedNode(s => s === n.ticker ? null : n.ticker)}>
                    {isSelected && (
                      <circle cx={cx} cy={cy} r={radius + 5} fill="none"
                        stroke={color} strokeWidth={2} opacity={0.5} />
                    )}
                    <circle cx={cx} cy={cy} r={radius}
                      fill={color} opacity={dimmed ? 0.15 : 0.85}
                      stroke={isSelected ? '#1F2937' : 'none'} strokeWidth={2} />
                    <text x={cx} y={cy + 3.5} textAnchor="middle" fill="white"
                      fontSize={8} fontWeight={700} fontFamily="Inter">
                      {n.ticker.slice(0, 5)}
                    </text>
                    {/* Degree label */}
                    <text x={cx} y={cy - radius - 4} textAnchor="middle"
                      fill={dimmed ? '#D1D5DB' : '#6B7280'} fontSize={7} fontFamily="Inter">
                      {n.degree}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </Card>

        {/* Attention Heatmap (Real Correlation Data) */}
        <Card className="min-h-[420px]">
          <h2 className="font-display font-bold text-lg text-secondary mb-1">
            Attention Heatmap <span className="text-xs text-text-muted font-normal">(correlation-derived)</span>
          </h2>
          <p className="text-xs text-text-secondary mb-3">
            60-day rolling correlation between top-15 stocks. GNN attention weights learn from these patterns.
          </p>
          {hoveredCell && (
            <div className="text-xs font-mono bg-bg-card rounded-lg px-3 py-1.5 mb-2 border border-border-light inline-block">
              {data.attention_tickers[hoveredCell.i]} × {data.attention_tickers[hoveredCell.j]}
              {' = '}
              <span className={hoveredCell.val > 0.5 ? 'text-primary font-bold' : 'text-text-muted'}>
                {hoveredCell.val.toFixed(4)}
              </span>
            </div>
          )}
          <div className="overflow-auto">
            <div className="inline-grid gap-0.5" style={{
              gridTemplateColumns: `56px repeat(${data.attention_tickers.length}, 26px)`,
            }}>
              {/* Header row */}
              <div />
              {data.attention_tickers.map(t => (
                <div key={t} className="text-[7px] text-text-muted font-mono text-center rotate-[-45deg] origin-center h-8 flex items-end justify-center">
                  {t.slice(0, 5)}
                </div>
              ))}
              {/* Data rows */}
              {data.attention_matrix.map((row, i) => (
                <div key={i} className="contents">
                  <div className="text-[7px] text-text-muted font-mono pr-1 flex items-center justify-end">
                    {data.attention_tickers[i]?.slice(0, 5)}
                  </div>
                  {row.map((v, j) => {
                    const intensity = Math.min(v, 1);
                    return (
                      <div key={j}
                        className="w-[26px] h-[26px] rounded-sm cursor-crosshair transition-transform hover:scale-125 hover:z-10"
                        style={{
                          backgroundColor: i === j
                            ? '#F3F4F6'
                            : `rgba(193, 95, 60, ${intensity * 0.85 + 0.05})`,
                        }}
                        onMouseEnter={() => setHoveredCell({ i, j, val: v })}
                        onMouseLeave={() => setHoveredCell(null)}
                        title={`${data.attention_tickers[i]} × ${data.attention_tickers[j]} = ${v.toFixed(4)}`}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2 mt-3">
            <span className="text-[10px] text-text-muted">Low</span>
            <div className="flex h-3 rounded-sm overflow-hidden flex-1 max-w-[160px]">
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} className="flex-1" style={{ backgroundColor: `rgba(193, 95, 60, ${i * 0.1 + 0.05})` }} />
              ))}
            </div>
            <span className="text-[10px] text-text-muted">High</span>
          </div>
        </Card>
      </div>

      {/* Selected Node Detail Panel */}
      {selectedNodeData && (
        <Card className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 rounded-full" style={{ backgroundColor: SECTOR_COLORS[selectedNodeData.sector] || '#9CA3AF' }} />
              <h2 className="font-display font-bold text-lg text-secondary">{selectedNodeData.ticker}</h2>
              <Badge variant="neutral">{selectedNodeData.sector}</Badge>
            </div>
            <button onClick={() => setSelectedNode(null)} className="text-text-muted hover:text-text text-sm">
              Close
            </button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            <div className="bg-bg-card rounded-xl p-3">
              <p className="text-xs text-text-secondary">Connections</p>
              <p className="text-xl font-bold font-mono text-primary">{selectedNodeData.degree}</p>
            </div>
            <div className="bg-bg-card rounded-xl p-3">
              <p className="text-xs text-text-secondary">Portfolio Weight</p>
              <p className="text-xl font-bold font-mono">{selectedNodeData.weight}%</p>
            </div>
            <div className="bg-bg-card rounded-xl p-3">
              <p className="text-xs text-text-secondary">Daily Return</p>
              <p className={`text-xl font-bold font-mono ${selectedNodeData.daily_return >= 0 ? 'text-profit' : 'text-loss'}`}>
                {selectedNodeData.daily_return > 0 ? '+' : ''}{selectedNodeData.daily_return}%
              </p>
            </div>
            <div className="bg-bg-card rounded-xl p-3">
              <p className="text-xs text-text-secondary">Edge Types</p>
              <div className="flex gap-1 mt-1">
                {['sector', 'supply', 'correlation'].map(type => {
                  const count = selectedNodeEdges.filter(e => e.type === type).length;
                  if (count === 0) return null;
                  return (
                    <span key={type} className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                      style={{ backgroundColor: EDGE_TYPE_COLORS[type] + '15', color: EDGE_TYPE_COLORS[type] }}>
                      {type[0].toUpperCase()}: {count}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
          {/* Connected stocks list */}
          <h3 className="text-sm font-medium text-text-secondary mb-2">Connected Stocks (sorted by weight)</h3>
          <div className="flex flex-wrap gap-2">
            {selectedNodeEdges
              .map(e => ({ ticker: e.source === selectedNode ? e.target : e.source, weight: e.weight, type: e.type }))
              .sort((a, b) => b.weight - a.weight)
              .map((conn, i) => {
                const node = data.nodes.find(n => n.ticker === conn.ticker);
                return (
                  <div key={i}
                    className="flex items-center gap-1.5 text-xs font-mono px-2.5 py-1 rounded-lg bg-white border border-border-light hover:border-primary/30 transition-colors cursor-pointer"
                    onClick={() => setSelectedNode(conn.ticker)}
                    style={{ borderLeftColor: EDGE_TYPE_COLORS[conn.type] || '#9CA3AF', borderLeftWidth: 3 }}>
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: SECTOR_COLORS[node?.sector || ''] || '#9CA3AF' }} />
                    <span className="font-semibold">{conn.ticker}</span>
                    <span className="text-text-muted">({conn.weight.toFixed(2)})</span>
                  </div>
                );
              })}
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Top Connections Table */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-1">
            Strongest Connections
          </h2>
          <p className="text-xs text-text-secondary mb-3">
            Top 20 stock pairs by correlation strength. Higher = more co-movement.
          </p>
          <div className="overflow-auto max-h-[380px]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-border">
                  <th className="text-left py-2 font-medium text-text-secondary">Stock A</th>
                  <th className="text-left py-2 font-medium text-text-secondary">Stock B</th>
                  <th className="text-right py-2 font-medium text-text-secondary">Correlation</th>
                  <th className="text-right py-2 font-medium text-text-secondary">Type</th>
                </tr>
              </thead>
              <tbody>
                {data.top_connections.map((c, i) => (
                  <tr key={i} className="border-b border-border-light hover:bg-bg-card transition-colors cursor-pointer"
                    onClick={() => setSelectedNode(c.stock_a)}>
                    <td className="py-2 font-mono font-medium">{c.stock_a}</td>
                    <td className="py-2 font-mono font-medium">{c.stock_b}</td>
                    <td className="py-2 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{
                            width: `${c.correlation * 100}%`,
                            backgroundColor: c.correlation > 0.6 ? '#C15F3C' : c.correlation > 0.4 ? '#F59E0B' : '#9CA3AF',
                          }} />
                        </div>
                        <span className="font-mono text-xs w-10 text-right">{c.correlation.toFixed(3)}</span>
                      </div>
                    </td>
                    <td className="py-2 text-right">
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded-full"
                        style={{ backgroundColor: (EDGE_TYPE_COLORS[c.type] || '#9CA3AF') + '15', color: EDGE_TYPE_COLORS[c.type] || '#9CA3AF' }}>
                        {c.type}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Degree Distribution */}
        <Card>
          <h2 className="font-display font-bold text-lg text-secondary mb-1">
            Degree Distribution
          </h2>
          <p className="text-xs text-text-secondary mb-3">
            How many connections each stock has. Hub stocks (high degree) are influential in the network.
          </p>
          <ResponsiveContainer width="100%" height={280} minHeight={1}>
            <BarChart data={degreeChartData} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
              <CartesianGrid stroke="#F3F4F6" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="degree" tick={{ fontSize: 11, fill: '#9CA3AF' }}
                axisLine={{ stroke: '#E5E7EB' }} tickLine={false}
                label={{ value: 'Connections per Stock', position: 'insideBottom', offset: -10, style: { fontSize: 11, fill: '#9CA3AF' } }} />
              <YAxis tick={{ fontSize: 11, fill: '#9CA3AF' }} axisLine={false} tickLine={false}
                label={{ value: 'Stocks', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9CA3AF' } }} />
              <Tooltip
                contentStyle={{ borderRadius: 12, border: '1px solid #E5E7EB', fontSize: 12 }}
                formatter={(value: number) => [`${value} stocks`, 'Count']}
                labelFormatter={(label) => `${label} connections`}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {degreeChartData.map((d, i) => (
                  <Cell key={i} fill={d.degree >= 8 ? '#C15F3C' : d.degree >= 5 ? '#E8A87C' : '#F3E4D7'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-2 text-[10px] text-text-muted">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-[#F3E4D7]" />Low (1-4)</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-[#E8A87C]" />Medium (5-7)</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-[#C15F3C]" />Hub (8+)</span>
          </div>
        </Card>
      </div>

      {/* Sector Connectivity Matrix */}
      <Card>
        <h2 className="font-display font-bold text-lg text-secondary mb-1">
          Sector Connectivity
        </h2>
        <p className="text-xs text-text-secondary mb-4">
          How sectors connect to each other. Same-sector connections are strongest (sector edges).
          Cross-sector connections reveal supply chain and correlation dependencies.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 font-medium text-text-secondary">Sector A</th>
                <th className="text-left py-2 font-medium text-text-secondary">Sector B</th>
                <th className="text-right py-2 font-medium text-text-secondary">Edges</th>
                <th className="text-right py-2 font-medium text-text-secondary">Avg Correlation</th>
                <th className="text-left py-2 pl-4 font-medium text-text-secondary">Strength</th>
              </tr>
            </thead>
            <tbody>
              {data.sector_connectivity.slice(0, 15).map((sc, i) => (
                <motion.tr key={i} variants={fadeSlideUp}
                  className="border-b border-border-light hover:bg-bg-card transition-colors">
                  <td className="py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: SECTOR_COLORS[sc.sector_a] || '#9CA3AF' }} />
                      <span className="font-medium text-sm">{sc.sector_a}</span>
                    </div>
                  </td>
                  <td className="py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: SECTOR_COLORS[sc.sector_b] || '#9CA3AF' }} />
                      <span className="font-medium text-sm">{sc.sector_b}</span>
                    </div>
                  </td>
                  <td className="py-2.5 text-right font-mono">{sc.n_edges}</td>
                  <td className="py-2.5 text-right font-mono">{sc.avg_weight.toFixed(3)}</td>
                  <td className="py-2.5 pl-4">
                    <div className="w-24 h-2.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all" style={{
                        width: `${Math.min(sc.avg_weight * 130, 100)}%`,
                        backgroundColor: sc.sector_a === sc.sector_b ? '#C15F3C' : '#6366F1',
                      }} />
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
