import { motion } from 'framer-motion';
import { fadeSlideUp } from '../../lib/animations';
import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  cream?: boolean;
  noPad?: boolean;
  interactive?: boolean;
}

export default function Card({ children, className, cream, noPad, interactive }: CardProps) {
  return (
    <motion.div
      variants={fadeSlideUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-40px' }}
      whileHover={interactive
        ? { y: -3, boxShadow: '0 12px 28px rgba(193,95,60,0.12), 0 4px 10px rgba(0,0,0,0.06)' }
        : { y: -2, boxShadow: '0 6px 18px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04)' }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
      className={clsx(
        'rounded-xl border border-border shadow-[0_1px_3px_rgba(0,0,0,0.05)]',
        cream ? 'bg-bg-cream' : 'bg-white',
        noPad ? '' : 'p-5',
        className,
      )}
    >
      {children}
    </motion.div>
  );
}
