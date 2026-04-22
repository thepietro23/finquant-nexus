import { motion } from 'framer-motion';

interface PageHeaderProps {
  title: string;
  subtitle: string;
  icon?: React.ReactNode;
  badge?: string;
}

export default function PageHeader({ title, subtitle, icon, badge }: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 120, damping: 16 }}
      className="mb-6"
    >
      <div className="flex items-center gap-3 mb-1.5">
        {icon && (
          <motion.span
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.08, type: 'spring', stiffness: 260, damping: 18 }}
            className="w-9 h-9 rounded-xl bg-primary-subtle flex items-center justify-center text-primary shrink-0"
          >
            {icon}
          </motion.span>
        )}
        <h1 className="font-display font-bold text-2xl text-secondary">{title}</h1>
        {badge && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-primary-subtle text-primary border border-primary/20 tracking-wide">
            {badge}
          </span>
        )}
      </div>
      <p className="text-sm text-text-secondary leading-relaxed">{subtitle}</p>
    </motion.div>
  );
}
