/** Framer Motion animation presets — v3 */
import type { Variants } from 'framer-motion';

export const fadeSlideUp: Variants = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 100, damping: 15 } },
};

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.4 } },
};

export const staggerContainer: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
};

export const staggerFast: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.05 } },
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 100, damping: 15 } },
};

export const popIn: Variants = {
  hidden: { opacity: 0, scale: 0.85 },
  visible: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 260, damping: 20 } },
};

export const slideFromLeft: Variants = {
  hidden: { opacity: 0, x: -30 },
  visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 100, damping: 15 } },
};

export const slideFromRight: Variants = {
  hidden: { opacity: 0, x: 30 },
  visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 100, damping: 15 } },
};

/** Page-level transition — used in DashboardLayout */
export const pageTransition: Variants = {
  initial: { opacity: 0, y: 18 },
  enter: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 90, damping: 18 } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.16, ease: 'easeIn' } },
};

/** Use on clickable cards / rows for tactile feedback */
export const tapScale = { whileTap: { scale: 0.97 } };

/** Hover lift for interactive cards */
export const hoverLift = {
  whileHover: { y: -4, boxShadow: '0 12px 32px rgba(193,95,60,0.13)' },
  transition: { type: 'spring', stiffness: 300, damping: 22 },
};
