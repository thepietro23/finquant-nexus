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
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 140, damping: 18 }}
      className="mb-5"
    >
      <div className="flex items-center gap-2.5 mb-1">
        {icon && (
          <span className="w-8 h-8 rounded-lg bg-primary-subtle flex items-center justify-center text-primary shrink-0">
            {icon}
          </span>
        )}
        <h1 className="font-display font-bold text-xl text-secondary">{title}</h1>
        {badge && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-primary-subtle text-primary border border-primary/20 tracking-wide">
            {badge}
          </span>
        )}
      </div>
      <p className="text-xs text-text-muted leading-relaxed pl-0.5">{subtitle}</p>
    </motion.div>
  );
}
