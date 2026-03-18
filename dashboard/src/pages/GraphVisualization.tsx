/**
 * Graph Visualization — Interactive stock connectivity network
 *
 * Shows NIFTY 50 stocks as nodes connected by:
 *   - Sector edges (same sector)
 *   - Supply chain edges (business relationships)
 *   - Correlation edges (|corr| > 0.4)
 *
 * Node size = degree (connections)
 * Node color = sector
 * Edge color = relationship type
 * All data from real backend (no Math.random)
 */
import { useEffect, useState, useMemo, useRef } from 'react';
import { GitGraph } from 'lucide-react';
import { api } from '../lib/api';
import type { GNNSummaryResponse } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import Badge from '../components/ui/Badge';

const PAGE_INFO = {
  title: 'Graph Visualization — What Does This Page Show?',
  sections: [
    { heading: 'What is this?', text: 'Interactive force-directed graph showing NIFTY 50 stocks as nodes and their relationships as edges. The layout is physics-based — connected stocks cluster together naturally.' },
    { heading: 'Node properties', text: 'Size = number of connections (degree). Color = sector (Banking=orange, IT=purple, etc.). Click any node to see detailed info in the right panel.' },
    { heading: 'Edge types', text: 'Sector (same industry), Supply Chain (business relationships), Correlation (60-day price co-movement > 0.4). Toggle each type on/off.' },
    { heading: 'Force simulation', text: 'Physics engine with repulsion (nodes push apart), attraction (connected nodes pull together), and gravity (keeps everything centered). Runs 150 frames then stabilizes.' },
    { heading: 'What to look for', text: 'Banking stocks cluster together, IT stocks form another cluster — this shows GNN correctly identifies sector relationships. Cross-sector correlation edges connect clusters.' },
  ],
};

// ── Sector colors ──
const SECTOR_COLORS: Record<string, string> = {
  'Banking': '#C15F3C', 'Finance': '#A34E30', 'IT': '#6366F1',
  'Telecom': '#8B5CF6', 'Pharma': '#0D9488', 'FMCG': '#16A34A',
  'Energy': '#F59E0B', 'Auto': '#3B82F6', 'Metals': '#EC4899',
  'Infrastructure': '#14B8A6', 'Infra': '#14B8A6', 'Others': '#9CA3AF',
  'Unknown': '#9CA3AF',
};

const EDGE_COLORS: Record<string, string> = {
  sector: '#C15F3C',
  supply: '#6366F1',
  correlation: '#0D9488',
};

interface GraphNode {
  id: string;
  ticker: string;
  sector: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
  dailyReturn: number;
  weight: number;
}

interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

// ── Simple force simulation ──
function applyForces(nodes: GraphNode[], edges: GraphEdge[], width: number, height: number) {
  const alpha = 0.3;
  const repulsion = 800;
  const attraction = 0.005;
  const centerForce = 0.01;

  // Center gravity
  nodes.forEach(n => {
    n.vx += (width / 2 - n.x) * centerForce;
    n.vy += (height / 2 - n.y) * centerForce;
  });

  // Repulsion between all nodes
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = repulsion / (dist * dist);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      nodes[i].vx -= fx;
      nodes[i].vy -= fy;
      nodes[j].vx += fx;
      nodes[j].vy += fy;
    }
  }

  // Attraction along edges
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(e => {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    if (!s || !t) return;
    const dx = t.x - s.x;
    const dy = t.y - s.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    // Stronger attraction for higher correlation weight
    const force = dist * attraction * (0.5 + e.weight * 0.5);
    s.vx += dx * force;
    s.vy += dy * force;
    t.vx -= dx * force;
    t.vy -= dy * force;
  });

  // Apply velocity
  nodes.forEach(n => {
    n.vx *= 0.8; // damping
    n.vy *= 0.8;
    n.x += n.vx * alpha;
    n.y += n.vy * alpha;
    // Keep in bounds
    n.x = Math.max(40, Math.min(width - 40, n.x));
    n.y = Math.max(40, Math.min(height - 40, n.y));
  });
}

export default function GraphVisualization() {
  const [gnnData, setGnnData] = useState<GNNSummaryResponse | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showEdgeType, setShowEdgeType] = useState({ sector: true, supply: true, correlation: true });
  const [loading, setLoading] = useState(true);
  const svgRef = useRef<SVGSVGElement>(null);
  const animRef = useRef<number>(0);
  const [, setTick] = useState(0);

  const W = 900, H = 600;

  // Load GNN data from backend (real data)
  useEffect(() => {
    api.gnnSummary()
      .then(d => { setGnnData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  // Build graph when data loads
  useEffect(() => {
    if (!gnnData) return;

    // Deterministic initial positions: arrange by sector in circle segments
    const sectorList = [...new Set(gnnData.nodes.map(n => n.sector))];
    const sectorAngle: Record<string, number> = {};
    sectorList.forEach((s, i) => { sectorAngle[s] = (i / sectorList.length) * Math.PI * 2; });

    // Seed a simple LCG for deterministic "jitter" without Math.random
    let seed = 42;
    const nextRand = () => { seed = (seed * 1664525 + 1013904223) & 0x7fffffff; return seed / 0x7fffffff; };

    const sectorCounters: Record<string, number> = {};
    const newNodes: GraphNode[] = gnnData.nodes.map(n => {
      if (!sectorCounters[n.sector]) sectorCounters[n.sector] = 0;
      const idx = sectorCounters[n.sector]++;
      const baseAngle = sectorAngle[n.sector] || 0;
      const spread = 0.4; // radians spread within sector
      const angle = baseAngle + (idx - 2) * spread * 0.3 + (nextRand() - 0.5) * 0.3;
      const r = 150 + nextRand() * 100;
      return {
        id: n.ticker,
        ticker: n.ticker,
        sector: n.sector,
        x: W / 2 + Math.cos(angle) * r,
        y: H / 2 + Math.sin(angle) * r,
        vx: 0, vy: 0,
        degree: n.degree,
        dailyReturn: n.daily_return,
        weight: n.weight,
      };
    });

    const newEdges: GraphEdge[] = gnnData.edges.map(e => ({
      source: e.source,
      target: e.target,
      type: e.type,
      weight: e.weight,
    }));

    setNodes(newNodes);
    setEdges(newEdges);
  }, [gnnData]);

  // Force simulation animation
  useEffect(() => {
    if (nodes.length === 0) return;
    let frame = 0;
    const maxFrames = 150;

    function step() {
      applyForces(nodes, edges, W, H);
      setTick(t => t + 1);
      frame++;
      if (frame < maxFrames) {
        animRef.current = requestAnimationFrame(step);
      }
    }
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [nodes.length]);

  // Filter edges by type
  const visibleEdges = edges.filter(e => showEdgeType[e.type as keyof typeof showEdgeType]);

  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  // Highlighted edges (connected to hovered/selected node)
  const highlightId = hovered || selected;
  const connectedNodes = useMemo(() => {
    if (!highlightId) return new Set<string>();
    const connected = new Set<string>();
    visibleEdges.forEach(e => {
      if (e.source === highlightId) connected.add(e.target);
      if (e.target === highlightId) connected.add(e.source);
    });
    return connected;
  }, [highlightId, visibleEdges]);

  const selectedNode = selected ? nodeMap.get(selected) : null;

  // Stats
  const edgeCounts = {
    sector: edges.filter(e => e.type === 'sector').length,
    supply: edges.filter(e => e.type === 'supply').length,
    correlation: edges.filter(e => e.type === 'correlation').length,
  };

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
    </div>
  );

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Graph Visualization"
          subtitle="Interactive NIFTY 50 stock network — real correlations, 3 edge types"
          icon={<GitGraph size={24} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      {/* Edge type filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <span className="text-sm font-medium text-text-secondary">Show edges:</span>
        {(['sector', 'supply', 'correlation'] as const).map(type => (
          <button key={type}
            onClick={() => setShowEdgeType(p => ({ ...p, [type]: !p[type] }))}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              showEdgeType[type]
                ? 'border-current opacity-100'
                : 'border-border opacity-40'
            }`}
            style={{ color: EDGE_COLORS[type] }}
          >
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: EDGE_COLORS[type] }} />
            {type.charAt(0).toUpperCase() + type.slice(1)} ({edgeCounts[type]})
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6">
        {/* Main Graph */}
        <Card noPad className="overflow-hidden">
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`}
            className="w-full bg-bg-card cursor-grab active:cursor-grabbing"
            style={{ minHeight: 500 }}>
            {/* Edges */}
            {visibleEdges.map((e, i) => {
              const s = nodeMap.get(e.source);
              const t = nodeMap.get(e.target);
              if (!s || !t) return null;
              const isHighlighted = highlightId && (e.source === highlightId || e.target === highlightId);
              return (
                <line key={i}
                  x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                  stroke={EDGE_COLORS[e.type] || '#E5E7EB'}
                  strokeWidth={isHighlighted ? 2 + e.weight : 0.6 + e.weight * 0.5}
                  opacity={highlightId ? (isHighlighted ? 0.8 : 0.08) : 0.25}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map(n => {
              const r = 6 + n.degree * 1.2; // Node size by degree
              const color = SECTOR_COLORS[n.sector] || '#9CA3AF';
              const isHighlighted = !highlightId || n.id === highlightId || connectedNodes.has(n.id);
              const isSelected = n.id === selected;
              return (
                <g key={n.id}
                  onMouseEnter={() => setHovered(n.id)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => setSelected(s => s === n.id ? null : n.id)}
                  className="cursor-pointer"
                >
                  {/* Glow ring for selected */}
                  {isSelected && (
                    <circle cx={n.x} cy={n.y} r={r + 6} fill="none"
                      stroke={color} strokeWidth={2} opacity={0.4} />
                  )}
                  {/* Node circle */}
                  <circle cx={n.x} cy={n.y} r={r}
                    fill={color} opacity={isHighlighted ? 0.9 : 0.2}
                    stroke={isSelected ? '#111827' : 'none'} strokeWidth={2}
                  />
                  {/* Label */}
                  {(r > 10 || hovered === n.id) && (
                    <text x={n.x} y={n.y - r - 4} textAnchor="middle"
                      fontSize={9} fontWeight={600} fill={isHighlighted ? '#374151' : '#D1D5DB'}
                      fontFamily="Inter">
                      {n.ticker}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </Card>

        {/* Right Panel — Details */}
        <div className="space-y-4">
          {/* Selected Node Info */}
          {selectedNode ? (
            <Card>
              <h3 className="font-display font-bold text-lg text-secondary mb-2">{selectedNode.ticker}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-secondary">Sector</span>
                  <Badge variant="neutral">{selectedNode.sector}</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">Connections</span>
                  <span className="font-mono font-semibold">{selectedNode.degree}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">Portfolio Weight</span>
                  <span className="font-mono font-semibold">{selectedNode.weight}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">Daily Return</span>
                  <span className={`font-mono font-semibold ${
                    selectedNode.dailyReturn >= 0 ? 'text-profit' : 'text-loss'
                  }`}>
                    {selectedNode.dailyReturn > 0 ? '+' : ''}{selectedNode.dailyReturn}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-secondary">Neighbors</span>
                  <span className="font-mono">{connectedNodes.size}</span>
                </div>
              </div>

              {/* Connected stocks by edge type */}
              <h4 className="text-xs font-medium text-text-secondary mt-4 mb-2">Connected Stocks</h4>
              <div className="flex flex-wrap gap-1.5">
                {[...connectedNodes].slice(0, 15).map(id => {
                  const node = nodeMap.get(id);
                  if (!node) return null;
                  const edge = visibleEdges.find(
                    e => (e.source === selectedNode.id && e.target === id) ||
                         (e.target === selectedNode.id && e.source === id)
                  );
                  return (
                    <span key={id}
                      className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-bg-card border border-border-light"
                      style={{
                        borderLeftColor: edge ? (EDGE_COLORS[edge.type] || '#9CA3AF') : '#9CA3AF',
                        borderLeftWidth: 2,
                      }}>
                      {node.ticker}
                      {edge && <span className="text-text-muted ml-1">({edge.weight.toFixed(2)})</span>}
                    </span>
                  );
                })}
              </div>
            </Card>
          ) : (
            <Card cream>
              <p className="text-sm text-text-secondary text-center py-4">
                Click a node to view stock details and connections
              </p>
            </Card>
          )}

          {/* Legend */}
          <Card>
            <h3 className="font-semibold text-sm text-secondary mb-3">Sector Legend</h3>
            <div className="space-y-1.5">
              {Object.entries(SECTOR_COLORS).filter(([k]) => !['Unknown', 'Infra'].includes(k)).map(([sector, color]) => {
                const count = nodes.filter(n => n.sector === sector).length;
                if (count === 0) return null;
                return (
                  <div key={sector} className="flex items-center gap-2 text-xs">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <span className="text-text-secondary flex-1">{sector}</span>
                    <span className="font-mono text-text-muted">{count}</span>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Graph Stats */}
          <Card>
            <h3 className="font-semibold text-sm text-secondary mb-3">Graph Stats</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-text-secondary">Total Nodes</span>
                <span className="font-mono font-semibold">{nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Total Edges</span>
                <span className="font-mono font-semibold">{edges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Visible Edges</span>
                <span className="font-mono font-semibold">{visibleEdges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Avg Degree</span>
                <span className="font-mono font-semibold">
                  {gnnData ? gnnData.avg_degree : 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-secondary">Density</span>
                <span className="font-mono font-semibold">
                  {gnnData ? gnnData.density.toFixed(4) : 0}
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
