import { motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { fadeSlideUp } from '../../lib/animations';
import { valueBg, formatPct } from '../../lib/formatters';
import SparkLine from '../charts/SparkLine';
import { MetricCardSkeleton } from './Skeleton';

export type MetricBadgeVariant = 'profit' | 'loss' | 'warning' | 'neutral';

export interface MetricBadge {
  label: string;
  variant: MetricBadgeVariant;
}

const badgeClasses: Record<MetricBadgeVariant, string> = {
  profit:  'bg-profit-light text-profit border border-profit/20',
  loss:    'bg-loss-light text-loss border border-loss/20',
  warning: 'bg-amber-50 text-amber-600 border border-amber-200',
  neutral: 'bg-bg-card text-text-muted border border-border',
};

interface MetricCardProps {
  title: string;
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  change?: number;
  sparkData?: number[];
  icon?: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
  loading?: boolean;
  badge?: MetricBadge;
}

function useAnimatedNumber(end: number, decimals: number, duration = 1200) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);
  const currentRef = useRef(0);

  useEffect(() => {
    const startVal = currentRef.current;
    startRef.current = performance.now();
    function tick(now: number) {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = startVal + (end - startVal) * eased;
      currentRef.current = val;
      setDisplay(val);
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [end, duration]);

  return display.toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export default function MetricCard({
  title, value, decimals = 2, prefix = '', suffix = '',
  change, sparkData, icon, onClick, active, loading, badge,
}: MetricCardProps) {
  const animated = useAnimatedNumber(value, decimals);

  if (loading) return <MetricCardSkeleton />;

  return (
    <motion.div
      variants={fadeSlideUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      onClick={onClick}
      whileHover={{ y: -4, boxShadow: '0 12px 32px rgba(193,95,60,0.14)' }}
      whileTap={onClick ? { scale: 0.97, y: 0 } : {}}
      transition={{ type: 'spring', stiffness: 300, damping: 22 }}
      className={`bg-white rounded-2xl border p-5 relative overflow-hidden
        ${active ? 'border-primary border-l-[3px] shadow-[0_10px_30px_rgba(193,95,60,0.12)] bg-[#FFFBF8]' : 'border-border'}
        ${onClick ? 'cursor-pointer' : ''}`}
    >
      {/* Subtle gradient shine */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/[0.03] to-transparent pointer-events-none" />

      <div className="flex items-center justify-between mb-2 relative">
        <span className="text-sm font-medium text-text-secondary">{title}</span>
        {icon && <span className="text-text-muted">{icon}</span>}
      </div>

      <div className="flex items-end gap-3 mb-2 relative">
        <span className="text-3xl font-bold font-mono text-text">
          {prefix}{animated}{suffix}
        </span>
        {change !== undefined && (
          <motion.span
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className={`text-sm font-medium px-2 py-0.5 rounded-full ${valueBg(change)}`}
          >
            {formatPct(change)}
          </motion.span>
        )}
      </div>

      {badge && badge.label && (
        <motion.div
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, type: 'spring', stiffness: 260, damping: 20 }}
          className="mb-2"
        >
          <span className={`text-[10px] font-bold tracking-widest px-2 py-0.5 rounded-full ${badgeClasses[badge.variant]}`}>
            {badge.label}
          </span>
        </motion.div>
      )}

      {sparkData && sparkData.length > 0 && (
        <div className="h-10 relative">
          <SparkLine data={sparkData} />
        </div>
      )}
    </motion.div>
  );
}
