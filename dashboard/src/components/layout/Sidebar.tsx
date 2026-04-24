import { NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PieChart, Brain, AlertTriangle,
  Users, MessageSquare, GitGraph, Workflow, TrendingUp,
  Settings, ExternalLink, ChevronLeft, ChevronRight,
} from 'lucide-react';
import { useState } from 'react';

const navItems = [
  { to: '/', icon: PieChart, label: 'Portfolio' },
  { to: '/rl', icon: Brain, label: 'RL Agent' },
  { to: '/stress', icon: AlertTriangle, label: 'Stress Testing' },
  { to: '/fl', icon: Users, label: 'Federated' },
  { to: '/sentiment', icon: MessageSquare, label: 'Sentiment' },
  { to: '/graph', icon: GitGraph, label: 'Graph Viz' },
  { to: '/workflow', icon: Workflow, label: 'Pipeline' },
  { to: '/future', icon: TrendingUp, label: 'Future Prediction' },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 260 }}
      transition={{ type: 'spring', stiffness: 200, damping: 25 }}
      className="h-screen sticky top-0 bg-white shadow-[1px_0_0_0_#E2E4E9] flex flex-col z-30"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-border">
        <motion.div
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-hover flex items-center justify-center shrink-0 shadow-md shadow-primary/30"
        >
          <span className="text-white font-bold text-sm font-display">FQ</span>
        </motion.div>
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.18 }}
              className="overflow-hidden"
            >
              <h1 className="font-display font-bold text-base text-secondary leading-tight">FINQUANT</h1>
              <p className="text-[10px] text-text-muted font-medium tracking-wider">NEXUS v4</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Nav Items */}
      <nav className="flex-1 py-3 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            title={collapsed ? label : undefined}
            className={({ isActive }) =>
              `group relative flex items-center gap-3 mx-2 my-0.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                isActive
                  ? 'bg-primary/[0.08] text-primary'
                  : 'text-text-secondary hover:bg-[#F5F6F8] hover:text-text'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <motion.span
                  whileHover={{ scale: 1.15 }}
                  whileTap={{ scale: 0.9 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                >
                  <Icon size={20} className={`shrink-0 transition-colors ${isActive ? 'text-primary' : ''}`} />
                </motion.span>
                <AnimatePresence mode="wait">
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.14 }}
                    >
                      {label}
                    </motion.span>
                  )}
                </AnimatePresence>
                {isActive && (
                  <motion.span
                    layoutId="active-dot"
                    className="ml-auto w-1.5 h-1.5 rounded-full bg-primary shrink-0"
                    transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                  />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom */}
      <div className="border-t border-border py-3">
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          title={collapsed ? 'API Docs' : undefined}
          className="flex items-center gap-3 mx-2 px-3 py-2.5 rounded-xl text-sm text-text-secondary hover:bg-bg-card hover:text-text transition-colors"
        >
          <ExternalLink size={20} className="shrink-0" />
          {!collapsed && <span>API Docs</span>}
        </a>
        <button
          title={collapsed ? 'Settings' : undefined}
          className="w-full flex items-center gap-3 mx-2 px-3 py-2.5 rounded-xl text-sm text-text-secondary hover:bg-bg-card hover:text-text cursor-pointer transition-colors"
        >
          <Settings size={20} className="shrink-0" />
          {!collapsed && <span>Settings</span>}
        </button>
      </div>

      {/* Collapse Toggle */}
      <motion.button
        onClick={() => setCollapsed(c => !c)}
        whileHover={{ scale: 1.15 }}
        whileTap={{ scale: 0.9 }}
        className="absolute -right-3.5 top-20 w-7 h-7 bg-white border border-border rounded-full flex items-center justify-center shadow-md hover:shadow-lg hover:border-primary/30 transition-shadow"
      >
        {collapsed ? <ChevronRight size={14} className="text-text-secondary" /> : <ChevronLeft size={14} className="text-text-secondary" />}
      </motion.button>
    </motion.aside>
  );
}
