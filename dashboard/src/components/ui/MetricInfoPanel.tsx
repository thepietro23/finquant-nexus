import { ChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export interface MetricDetail {
  what: string;
  why: string;
  how: string;
  good: string;
}

interface MetricInfoPanelProps {
  expandedMetric: string | null;
  onClose: () => void;
  details: Record<string, MetricDetail & { interpret?: string }>;
}

export default function MetricInfoPanel({ expandedMetric, onClose, details }: MetricInfoPanelProps) {
  return (
    <AnimatePresence>
      {expandedMetric && details[expandedMetric] && (
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
                <h3 className="font-display font-bold text-base text-secondary mb-3">
                  {expandedMetric} — Explained
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
                  <div className="bg-white/70 rounded-xl px-4 py-3">
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">What is it?</p>
                    <p className="text-xs text-text-secondary leading-relaxed">{details[expandedMetric].what}</p>
                  </div>
                  <div className="bg-white/70 rounded-xl px-4 py-3">
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">Why does it matter?</p>
                    <p className="text-xs text-text-secondary leading-relaxed">{details[expandedMetric].why}</p>
                  </div>
                  <div className="bg-white/70 rounded-xl px-4 py-3">
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">How is it calculated?</p>
                    <p className="text-xs text-text-secondary leading-relaxed font-mono">{details[expandedMetric].how}</p>
                  </div>
                  <div className="bg-white/70 rounded-xl px-4 py-3">
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">Good or bad?</p>
                    <p className="text-xs text-text-secondary leading-relaxed">{details[expandedMetric].good}</p>
                  </div>
                </div>
                {details[expandedMetric].interpret && (
                  <div className="bg-white/70 rounded-xl px-4 py-3">
                    <p className="text-[10px] uppercase tracking-wider text-text-muted font-medium mb-1">Your Data Analysis</p>
                    <p className="text-sm text-text font-medium leading-relaxed">{details[expandedMetric].interpret}</p>
                  </div>
                )}
              </div>
              <button
                onClick={onClose}
                className="shrink-0 p-1.5 rounded-lg hover:bg-white/50 transition-colors text-text-muted"
              >
                <ChevronUp size={18} />
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
