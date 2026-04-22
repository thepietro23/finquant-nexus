import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import Sidebar from './Sidebar';
import Header from './Header';

const pageVariants = {
  initial: { opacity: 0, y: 18 },
  enter: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 90, damping: 18, mass: 0.8 } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.18, ease: 'easeIn' as const } },
};

export default function DashboardLayout() {
  const location = useLocation();

  return (
    <div className="flex min-h-screen bg-bg">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Header />
        <main className="flex-1 p-6 overflow-y-auto">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              variants={pageVariants}
              initial="initial"
              animate="enter"
              exit="exit"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
