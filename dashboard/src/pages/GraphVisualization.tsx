import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { GitGraph, MousePointer2, Network, AlertCircle, RefreshCw, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Skeleton } from '../components/ui/Skeleton';
import { api } from '../lib/api';
import { toast } from '../lib/toast';
import type { GNNSummaryResponse } from '../lib/api';
import Card from '../components/ui/Card';
import PageHeader from '../components/ui/PageHeader';
import PageInfoPanel from '../components/ui/PageInfoPanel';
import { fadeSlideUp, staggerFast } from '../lib/animations';

function safeNum(v: unknown, fallback = 0): number {
  const n = Number(v);
  return isFinite(n) ? n : fallback;
}

const PAGE_INFO = {
  title: 'Graph Visualization — How to Use This Tab',
  sections: [
    { heading: '1. What is this tab?', text: 'Interactive force-directed graph of all 47 NIFTY 50 stocks connected by 3 relationship types. Physics simulation clusters related stocks — Banking stocks pull toward each other, IT stocks form their own cluster, Energy nodes link via correlation edges. This is the Graph Neural Network (GNN) input graph visualized.' },
    { heading: '2. GNN — Graph Neural Network in FINQUANT-NEXUS', text: 'The GNN (T-GAT: Temporal Graph Attention Network) learns 32-dimensional embeddings for each stock by aggregating information from its graph neighbours. These embeddings feed into the RL agent as extra features — so the agent "sees" which stocks are correlated and avoids doubling up on the same cluster (natural diversification).' },
    { heading: '3. Node properties', text: 'Node size = degree (number of connections — more connections = bigger node = more integrated into market). Node color = sector. Click any node to inspect: ticker, sector, portfolio weight %, latest daily return %, and top-5 connections by correlation strength.' },
    { heading: '4. Three edge types', text: 'Sector edges (orange): both stocks in the same industry — static structural relationship. Supply chain edges (purple): known business dependencies (e.g., Reliance → BPCL via energy supply). Correlation edges (teal): 60-day rolling price co-movement above threshold — dynamic, changes with market conditions. Toggle each type on/off using the filter buttons.' },
    { heading: '5. What to look for', text: 'Tight, dense clusters = high intra-sector correlation (risk: one negative event hits all). Cross-sector teal edges = hidden dependencies you wouldn\'t expect from fundamentals. Isolated nodes (few edges) = good diversifiers — they move independently of market and reduce portfolio variance. GNN attention matrix (heatmap) shows which stock-pairs the network considers most influential.' },
    { heading: '6. How this improves the RL portfolio', text: 'Highly connected stock pairs (HDFC Bank ↔ ICICI Bank correlation often >0.85) get Sharpe-penalized when both overweighted — it is a correlated bet. GNN makes this visible to the RL agent via embeddings: agent learns to diversify across low-edge-count stocks. Result: better sector spread vs. a naive equal-weight approach.' },
  ],
};

const SECTOR_COLORS: Record<string, string> = {
  Banking: '#C15F3C', Finance: '#A34E30', IT: '#6366F1',
  Telecom: '#8B5CF6', Pharma: '#0D9488', FMCG: '#16A34A',
  Energy: '#F59E0B', Auto: '#3B82F6', Metals: '#EC4899',
  Infrastructure: '#14B8A6', Infra: '#14B8A6', Others: '#9CA3AF',
  Unknown: '#9CA3AF',
};

const EDGE_COLORS: Record<string, string> = {
  sector: '#C15F3C',
  supply: '#6366F1',
  correlation: '#0D9488',
};

const EDGE_LABELS: Record<string, string> = {
  sector: 'Sector',
  supply: 'Supply Chain',
  correlation: 'Correlation',
};

interface GraphNode {
  id: string; ticker: string; sector: string;
  x: number; y: number; vx: number; vy: number;
  degree: number; dailyReturn: number; weight: number;
}

interface GraphEdge {
  source: string; target: string; type: string; weight: number;
}

// ── Improved force simulation with cooling schedule ──
function applyForces(
  nodes: GraphNode[], edges: GraphEdge[],
  W: number, H: number, alpha: number,
) {
  const repulsion  = 1500;
  const attraction = 0.004;
  const centerForce = 0.006;
  const minDist    = 30;

  // Center gravity
  nodes.forEach(n => {
    n.vx += (W / 2 - n.x) * centerForce;
    n.vy += (H / 2 - n.y) * centerForce;
  });

  // Repulsion with soft minimum distance
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const distSq = Math.max(dx * dx + dy * dy, 1);
      const dist = Math.sqrt(distSq);
      const clampedDist = Math.max(dist, minDist);
      const force = repulsion / (clampedDist * clampedDist);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      nodes[i].vx -= fx;
      nodes[i].vy -= fy;
      nodes[j].vx += fx;
      nodes[j].vy += fy;
    }
  }

  // Edge attraction
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(e => {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    if (!s || !t) return;
    const dx = t.x - s.x;
    const dy = t.y - s.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const force = dist * attraction * (0.4 + e.weight * 0.6);
    s.vx += dx * force;
    s.vy += dy * force;
    t.vx -= dx * force;
    t.vy -= dy * force;
  });

  // Velocity + bounds
  nodes.forEach(n => {
    n.vx *= 0.84;
    n.vy *= 0.84;
    n.x += n.vx * alpha;
    n.y += n.vy * alpha;
    n.x = Math.max(55, Math.min(W - 55, n.x));
    n.y = Math.max(45, Math.min(H - 45, n.y));
  });
}

// ── Deterministic per-ticker hash for stable initial positions ──
function nodeHash(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h) ^ s.charCodeAt(i);
  return Math.abs(h);
}
function nodeRand(ticker: string, salt: number): number {
  return (nodeHash(ticker + salt) % 10000) / 10000;
}

export default function GraphVisualization() {
  const [gnnData, setGnnData]   = useState<GNNSummaryResponse | null>(null);
  const [nodes,   setNodes]     = useState<GraphNode[]>([]);
  const [edges,   setEdges]     = useState<GraphEdge[]>([]);
  const [hovered, setHovered]   = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showEdgeType, setShowEdgeType] = useState({ sector: true, supply: true, correlation: true });
  const [loading,   setLoading]   = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [simDone,   setSimDone]   = useState(false);

  const animRef = useRef<number>(0);
  const [, setTick] = useState(0);

  const W = 960, H = 620;
  const MAX_FRAMES = 300;

  // ── Load data ──
  const load = useCallback(() => {
    setLoading(true); setLoadError(null); setSimDone(false);
    api.gnnSummary()
      .then(d  => { setGnnData(d); setLoading(false); })
      .catch(e => {
        const msg = e instanceof Error ? e.message : 'Failed to load graph data';
        setLoadError(msg); setLoading(false);
        toast.error(msg);
      });
  }, []);

  useEffect(() => { load(); }, [load]);

  // ── Build nodes with sector-cluster initial positions ──
  useEffect(() => {
    if (!gnnData) return;

    const sectorList = [...new Set(gnnData.nodes.map(n => n.sector))];
    // Place sector centers in a ring, starting from top (-π/2)
    const sectorCenter: Record<string, [number, number]> = {};
    sectorList.forEach((s, i) => {
      const angle = (i / sectorList.length) * Math.PI * 2 - Math.PI / 2;
      const r = Math.min(W, H) * 0.31;
      sectorCenter[s] = [W / 2 + Math.cos(angle) * r, H / 2 + Math.sin(angle) * r];
    });

    // Count stocks per sector for sub-circle radius
    const sectorSize: Record<string, number> = {};
    gnnData.nodes.forEach(n => { sectorSize[n.sector] = (sectorSize[n.sector] ?? 0) + 1; });
    const sectorCounters: Record<string, number> = {};

    const newNodes: GraphNode[] = gnnData.nodes.map(n => {
      const idx = sectorCounters[n.sector] ?? 0;
      sectorCounters[n.sector] = idx + 1;
      const total = sectorSize[n.sector];
      const [cx, cy] = sectorCenter[n.sector] ?? [W / 2, H / 2];
      const angle = (idx / Math.max(total, 1)) * Math.PI * 2;
      const subR  = 18 + total * 5;
      return {
        id: n.ticker, ticker: n.ticker, sector: n.sector,
        x: cx + Math.cos(angle) * subR + (nodeRand(n.ticker, 0) - 0.5) * 12,
        y: cy + Math.sin(angle) * subR + (nodeRand(n.ticker, 1) - 0.5) * 12,
        vx: 0, vy: 0,
        degree: n.degree, dailyReturn: n.daily_return, weight: n.weight,
      };
    });

    setNodes(newNodes);
    setEdges(gnnData.edges.map(e => ({ source: e.source, target: e.target, type: e.type, weight: e.weight })));
    setSimDone(false);
  }, [gnnData]);

  // ── Force simulation with cooling ──
  useEffect(() => {
    if (nodes.length === 0) return;
    let frame = 0;
    cancelAnimationFrame(animRef.current);

    function step() {
      // Cooling: starts at 1.0, eases to 0.04
      const alpha = Math.max(0.04, 1 - frame / MAX_FRAMES);
      applyForces(nodes, edges, W, H, alpha);
      setTick(t => t + 1);
      frame++;
      if (frame < MAX_FRAMES) {
        animRef.current = requestAnimationFrame(step);
      } else {
        setSimDone(true);
      }
    }
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [nodes.length]);

  // ── Derived state ──
  const visibleEdges = edges.filter(e => showEdgeType[e.type as keyof typeof showEdgeType]);
  const nodeMap      = new Map(nodes.map(n => [n.id, n]));
  const highlightId  = hovered || selected;

  const connectedNodes = useMemo(() => {
    if (!highlightId) return new Set<string>();
    const s = new Set<string>();
    visibleEdges.forEach(e => {
      if (e.source === highlightId) s.add(e.target);
      if (e.target === highlightId) s.add(e.source);
    });
    return s;
  }, [highlightId, visibleEdges]);

  const selectedNode = selected ? nodeMap.get(selected) : null;

  // Median degree for label threshold
  const medianDegree = useMemo(() => {
    if (nodes.length === 0) return 0;
    const sorted = [...nodes].map(n => n.degree).sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length * 0.75)] ?? 0; // top quartile
  }, [nodes]);

  const edgeCounts = {
    sector:      edges.filter(e => e.type === 'sector').length,
    supply:      edges.filter(e => e.type === 'supply').length,
    correlation: edges.filter(e => e.type === 'correlation').length,
  };

  // ── Loading / Error states ──
  if (loading) return (
    <div className="space-y-4">
      <Skeleton className="h-7 w-48" rounded="lg" />
      <Skeleton className="h-[560px] w-full" rounded="xl" />
    </div>
  );

  if (loadError && nodes.length === 0) return (
    <Card>
      <div className="flex flex-col items-center gap-3 py-14 text-center">
        <AlertCircle size={34} className="text-loss" />
        <p className="font-semibold text-text">Graph data failed to load</p>
        <p className="text-sm text-text-muted max-w-sm">{loadError}</p>
        <button onClick={load}
          className="mt-2 px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-hover transition-colors">
          Retry
        </button>
      </div>
    </Card>
  );

  return (
    <div>
      {/* Header row */}
      <div className="flex items-start justify-between mb-5">
        <PageHeader
          title="Graph Visualization"
          subtitle={`${nodes.length} nodes · ${edges.length} edges · sector clusters · 60-day rolling correlations`}
          icon={<GitGraph size={20} />}
        />
        <PageInfoPanel title={PAGE_INFO.title} sections={PAGE_INFO.sections} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_272px] gap-4 items-start">

        {/* ── Main Graph Card ── */}
        <div className="rounded-xl border border-border bg-white shadow-[0_1px_3px_rgba(0,0,0,0.05)] overflow-hidden">

          {/* Control strip */}
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border-light bg-[#FAFBFC]">
            <span className="text-[11px] font-semibold text-text-muted uppercase tracking-wide mr-1">Edges</span>
            {(['sector', 'supply', 'correlation'] as const).map(type => (
              <button key={type}
                onClick={() => setShowEdgeType(p => ({ ...p, [type]: !p[type] }))}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium transition-all border ${
                  showEdgeType[type]
                    ? 'border-current shadow-sm'
                    : 'border-transparent bg-bg-card text-text-muted opacity-50'
                }`}
                style={showEdgeType[type] ? { color: EDGE_COLORS[type], borderColor: EDGE_COLORS[type] + '60', background: EDGE_COLORS[type] + '10' } : undefined}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: showEdgeType[type] ? EDGE_COLORS[type] : '#9CA3AF' }} />
                {EDGE_LABELS[type]}
                <span className="opacity-60">({edgeCounts[type]})</span>
              </button>
            ))}

            <div className="flex-1" />

            {/* Sim status */}
            <span className={`flex items-center gap-1.5 text-[10px] font-medium transition-colors ${simDone ? 'text-profit' : 'text-text-muted'}`}>
              {simDone
                ? <><span className="w-1.5 h-1.5 rounded-full bg-profit" /> Stable</>
                : <><RefreshCw size={10} className="animate-spin" /> Simulating…</>}
            </span>

            {/* Reset selection */}
            {selected && (
              <button onClick={() => setSelected(null)}
                className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text px-2 py-1 rounded-md hover:bg-bg-card transition-colors border border-border-light">
                <X size={10} /> Clear
              </button>
            )}

            {/* Rerun layout */}
            <button
              onClick={() => { setSimDone(false); setGnnData(d => d ? { ...d } : d); }}
              title="Rerun layout"
              className="p-1.5 rounded-md text-text-muted hover:text-primary hover:bg-primary-subtle transition-colors">
              <RefreshCw size={12} />
            </button>
          </div>

          {/* SVG canvas */}
          <div className="relative">
            <svg viewBox={`0 0 ${W} ${H}`}
              className="w-full cursor-default"
              style={{ minHeight: 480, display: 'block' }}>

              {/* Dot grid background */}
              <defs>
                <pattern id="gnn-dots" x="0" y="0" width="22" height="22" patternUnits="userSpaceOnUse">
                  <circle cx="1" cy="1" r="0.9" fill="#E2E4E9" />
                </pattern>
              </defs>
              <rect width={W} height={H} fill="url(#gnn-dots)" />

              {/* Edges */}
              {visibleEdges.map((e, i) => {
                const s = nodeMap.get(e.source);
                const t = nodeMap.get(e.target);
                if (!s || !t) return null;
                const isLit = highlightId && (e.source === highlightId || e.target === highlightId);
                const baseOpacity = e.type === 'sector' ? 0.22 : e.type === 'supply' ? 0.28 : 0.18;
                return (
                  <line key={i}
                    x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                    stroke={EDGE_COLORS[e.type] ?? '#E5E7EB'}
                    strokeWidth={isLit ? 1.8 + e.weight * 1.2 : 0.8 + e.weight * 0.4}
                    strokeLinecap="round"
                    opacity={highlightId ? (isLit ? 0.75 : 0.05) : baseOpacity}
                  />
                );
              })}

              {/* Nodes */}
              {nodes.map(n => {
                const r     = 5 + n.degree * 1.1;
                const color = SECTOR_COLORS[n.sector] ?? '#9CA3AF';
                const isLit = !highlightId || n.id === highlightId || connectedNodes.has(n.id);
                const isSel = n.id === selected;
                const isHov = n.id === hovered;

                // Label visibility: top-quartile hubs, hovered, selected, or neighbor of selected
                const showLabel = n.degree >= medianDegree || isHov || isSel || connectedNodes.has(n.id);
                const labelOpacity = !highlightId
                  ? (n.degree >= medianDegree ? 0.75 : 0)
                  : (isLit ? (isHov || isSel ? 1 : 0.7) : 0.08);

                return (
                  <g key={n.id}
                    onMouseEnter={() => setHovered(n.id)}
                    onMouseLeave={() => setHovered(null)}
                    onClick={() => setSelected(prev => prev === n.id ? null : n.id)}
                    style={{ cursor: 'pointer' }}>

                    {/* Selected outer glow */}
                    {isSel && (
                      <>
                        <circle cx={n.x} cy={n.y} r={r + 9} fill={color} opacity={0.08} />
                        <circle cx={n.x} cy={n.y} r={r + 5} fill="none" stroke={color} strokeWidth={1.5} opacity={0.35} />
                      </>
                    )}

                    {/* Hover pulse ring */}
                    {isHov && !isSel && (
                      <circle cx={n.x} cy={n.y} r={r + 4} fill="none" stroke={color} strokeWidth={1} opacity={0.4} />
                    )}

                    {/* Main node */}
                    <circle cx={n.x} cy={n.y} r={r}
                      fill={color}
                      opacity={isLit ? (isSel ? 1 : isHov ? 0.95 : 0.85) : 0.18}
                    />

                    {/* White inner ring for definition */}
                    <circle cx={n.x} cy={n.y} r={r}
                      fill="none" stroke="white" strokeWidth={1.2}
                      opacity={isLit ? 0.6 : 0.15}
                    />

                    {/* Label */}
                    {showLabel && (
                      <>
                        {/* label background pill */}
                        <rect
                          x={n.x - n.ticker.length * 3.2 - 3}
                          y={n.y - r - 15}
                          width={n.ticker.length * 6.4 + 6}
                          height={11}
                          rx={3}
                          fill="white"
                          opacity={labelOpacity * 0.85}
                        />
                        <text
                          x={n.x} y={n.y - r - 6}
                          textAnchor="middle"
                          fontSize={isHov || isSel ? 8.5 : 7.5}
                          fontWeight={isSel || isHov ? 700 : 600}
                          fill={isSel ? color : '#374151'}
                          fontFamily="Inter, system-ui"
                          opacity={labelOpacity}
                        >
                          {n.ticker}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
            </svg>

            {/* No-selection hint overlay */}
            {!selected && !hovered && simDone && nodes.length > 0 && (
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-none">
                <span className="flex items-center gap-1.5 text-[10px] text-text-muted bg-white/80 backdrop-blur-sm px-3 py-1.5 rounded-full border border-border-light shadow-sm">
                  <MousePointer2 size={10} /> Click any node to inspect
                </span>
              </div>
            )}
          </div>
        </div>

        {/* ── Right Panel ── */}
        <div className="space-y-3">

          {/* Node info */}
          <AnimatePresence mode="wait">
            {selectedNode ? (
              <motion.div key={`node-${selectedNode.ticker}`}
                variants={fadeSlideUp} initial="hidden" animate="visible"
                exit={{ opacity: 0, y: -6, transition: { duration: 0.12 } }}>
                <Card>
                  {/* Sector accent bar */}
                  <div className="h-1 -mx-5 -mt-5 mb-4 rounded-t-xl"
                    style={{ background: SECTOR_COLORS[selectedNode.sector] ?? '#9CA3AF' }} />

                  <div className="flex items-center gap-2 mb-3">
                    <span className="font-display font-bold text-base text-secondary">{selectedNode.ticker}</span>
                    <span className={`ml-auto text-xs font-semibold px-2 py-0.5 rounded-full ${
                      selectedNode.dailyReturn >= 0 ? 'bg-profit-light text-profit' : 'bg-loss-light text-loss'
                    }`}>
                      {selectedNode.dailyReturn > 0 ? '+' : ''}{safeNum(selectedNode.dailyReturn).toFixed(2)}%
                    </span>
                  </div>

                  {/* 2×2 stats grid */}
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    {[
                      { label: 'Sector',  val: selectedNode.sector },
                      { label: 'Degree',  val: String(selectedNode.degree) },
                      { label: 'Weight',  val: `${selectedNode.weight}%` },
                      { label: 'Neighbors', val: String(connectedNodes.size) },
                    ].map(row => (
                      <div key={row.label} className="bg-bg-card rounded-lg px-2.5 py-2">
                        <p className="text-[10px] text-text-muted mb-0.5">{row.label}</p>
                        <p className="text-xs font-semibold text-text font-mono truncate">{row.val}</p>
                      </div>
                    ))}
                  </div>

                  <p className="text-[10px] font-medium text-text-secondary mb-1.5">Connected to</p>
                  <motion.div variants={staggerFast} initial="hidden" animate="visible"
                    className="flex flex-wrap gap-1">
                    {[...connectedNodes].slice(0, 16).map(id => {
                      const nd = nodeMap.get(id);
                      if (!nd) return null;
                      const edge = visibleEdges.find(e =>
                        (e.source === selectedNode.id && e.target === id) ||
                        (e.target === selectedNode.id && e.source === id),
                      );
                      return (
                        <motion.button key={id} variants={fadeSlideUp}
                          onClick={() => setSelected(id)}
                          className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-border-light hover:border-border transition-colors"
                          style={{ borderLeftColor: edge ? EDGE_COLORS[edge.type] : '#9CA3AF', borderLeftWidth: 2 }}>
                          {nd.ticker}
                        </motion.button>
                      );
                    })}
                  </motion.div>
                </Card>
              </motion.div>
            ) : (
              <motion.div key="hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className="rounded-xl border border-dashed border-border p-5 text-center">
                  <div className="w-9 h-9 rounded-xl bg-primary/8 flex items-center justify-center mx-auto mb-2">
                    <MousePointer2 size={16} className="text-primary" />
                  </div>
                  <p className="text-xs font-medium text-text-secondary">Click a node</p>
                  <p className="text-[10px] text-text-muted mt-0.5">to inspect connections</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Sector Legend + Graph Stats — merged */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold text-secondary uppercase tracking-wide">Sectors</h3>
              <div className="flex items-center gap-3 text-[10px] text-text-muted">
                <span>{nodes.length} nodes</span>
                <span className="text-border">·</span>
                <span>{visibleEdges.length} edges</span>
              </div>
            </div>
            <div className="space-y-1">
              {Object.entries(SECTOR_COLORS)
                .filter(([k]) => !['Unknown', 'Infra'].includes(k))
                .map(([sector, color]) => {
                  const count = nodes.filter(n => n.sector === sector).length;
                  if (count === 0) return null;
                  const pct = (count / nodes.length) * 100;
                  return (
                    <div key={sector} className="flex items-center gap-2 text-[11px]">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                      <span className="text-text-secondary flex-1 truncate">{sector}</span>
                      <div className="w-14 h-1 bg-border-light rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.7 }} />
                      </div>
                      <span className="font-mono text-text-muted w-4 text-right">{count}</span>
                    </div>
                  );
                })}
            </div>

            {/* Density + Avg Degree mini row */}
            <div className="flex gap-3 mt-3 pt-3 border-t border-border-light">
              <div className="flex-1 text-center">
                <p className="text-[10px] text-text-muted">Density</p>
                <p className="text-xs font-mono font-semibold text-text">{gnnData ? safeNum(gnnData.density).toFixed(3) : '—'}</p>
              </div>
              <div className="w-px bg-border-light" />
              <div className="flex-1 text-center">
                <p className="text-[10px] text-text-muted">Avg Degree</p>
                <p className="text-xs font-mono font-semibold text-text">{gnnData?.avg_degree ?? '—'}</p>
              </div>
              <div className="w-px bg-border-light" />
              <div className="flex-1 text-center">
                <p className="text-[10px] text-text-muted">Supply</p>
                <p className="text-xs font-mono font-semibold text-text">{edgeCounts.supply}</p>
              </div>
            </div>
          </Card>

          {/* Top Correlations */}
          {gnnData && gnnData.top_connections.length > 0 && (
            <Card>
              <h3 className="text-xs font-semibold text-secondary uppercase tracking-wide mb-3">Strongest Pairs</h3>
              <div className="space-y-2">
                {gnnData.top_connections.slice(0, 8).map((c, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-bold text-text w-9 shrink-0">{c.stock_a}</span>
                    <span className="text-[9px] text-text-muted shrink-0">↔</span>
                    <span className="text-[10px] font-mono font-bold text-text w-12 shrink-0">{c.stock_b}</span>
                    <div className="flex-1 h-1 bg-border-light rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all"
                        style={{ width: `${safeNum(c.correlation) * 100}%`, backgroundColor: EDGE_COLORS[c.type] ?? '#9CA3AF', opacity: 0.8 }} />
                    </div>
                    <span className="text-[10px] font-mono text-text-muted w-7 text-right shrink-0">
                      {safeNum(c.correlation).toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* GNN → Portfolio callout */}
      <div className="mt-4 rounded-xl border border-primary/20 bg-primary/[0.03] px-4 py-3">
        <div className="flex items-start gap-3">
          <Network size={15} className="text-primary shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-[11px] font-bold text-primary uppercase tracking-wide mb-1.5">GNN → RL Agent Pipeline</p>
            <div className="flex items-center gap-1 text-[10px] mb-2">
              {[
                { label: 'Stock Graph', sub: `${nodes.length} nodes`, highlight: false },
                { label: 'GNN Layer',   sub: '32-dim embeddings', highlight: true },
                { label: 'RL Agent',    sub: 'input features', highlight: false },
                { label: 'Portfolio',   sub: 'diversified', highlight: false },
              ].map((step, i, arr) => (
                <div key={step.label} className="flex items-center gap-1">
                  <div className={`rounded-lg px-2.5 py-1.5 text-center border ${
                    step.highlight
                      ? 'border-primary/30 bg-primary/8 text-primary'
                      : 'border-border-light bg-white text-text-secondary'
                  }`}>
                    <p className="font-semibold leading-none mb-0.5">{step.label}</p>
                    <p className="text-[9px] opacity-70">{step.sub}</p>
                  </div>
                  {i < arr.length - 1 && <span className="text-text-muted">→</span>}
                </div>
              ))}
            </div>
            <p className="text-[11px] text-text-secondary">
              Each stock gets a <span className="font-semibold text-text">32-dim embedding</span> encoding its graph neighborhood.
              The RL agent uses these to avoid concentrating in correlated clusters —
              heavily connected hub stocks are <span className="font-semibold text-text">Sharpe-penalized</span> when overweighted together.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
