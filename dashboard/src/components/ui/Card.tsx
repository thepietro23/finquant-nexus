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
      whileHover={interactive ? { y: -3, boxShadow: '0 12px 28px rgba(193,95,60,0.10), 0 4px 10px rgba(0,0,0,0.05)' } : { y: -2, boxShadow: '0 8px 20px rgba(193,95,60,0.07)' }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
      className={clsx(
        'rounded-2xl border border-border',
        cream ? 'bg-bg-cream' : 'bg-white',
        noPad ? '' : 'p-6',
        className,
      )}
    >
      {children}
    </motion.div>
  );
}
