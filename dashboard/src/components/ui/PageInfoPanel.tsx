import { useState } from 'react';
import { Info, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface InfoSection {
  heading: string;
  text: string;
}

interface PageInfoPanelProps {
  title: string;
  sections: InfoSection[];
}

export default function PageInfoPanel({ title, sections }: PageInfoPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-xl border transition-all ${
          open
            ? 'bg-primary text-white border-primary'
            : 'border-border text-text-secondary hover:border-primary hover:text-primary hover:bg-primary-subtle'
        }`}
      >
        <Info size={14} />
        {open ? 'Close Info' : 'Page Info'}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 25 }}
            className="overflow-hidden mb-6"
          >
            <div className="bg-primary-subtle/50 border border-primary-light rounded-2xl p-6">
              <div className="flex items-start justify-between mb-4">
                <h3 className="font-display font-bold text-base text-secondary">{title}</h3>
                <button onClick={() => setOpen(false)} className="p-1 rounded-lg hover:bg-white/50 text-text-muted">
                  <X size={16} />
                </button>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {sections.map((s, i) => (
                  <div key={i} className="bg-white/70 rounded-xl px-4 py-3">
                    <p className="text-xs font-semibold text-primary mb-1">{s.heading}</p>
                    <p className="text-xs text-text-secondary leading-relaxed">{s.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
