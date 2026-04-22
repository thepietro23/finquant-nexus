import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle, XCircle, Info, AlertTriangle, X } from 'lucide-react';
import { toast as toastStore, type ToastItem } from '../../lib/toast';

const icons = {
  success: <CheckCircle size={17} className="text-profit shrink-0" />,
  error: <XCircle size={17} className="text-loss shrink-0" />,
  info: <Info size={17} className="text-info shrink-0" />,
  warning: <AlertTriangle size={17} className="text-warning shrink-0" />,
};

const borders = {
  success: 'border-l-profit',
  error: 'border-l-loss',
  info: 'border-l-info',
  warning: 'border-l-warning',
};

function ToastChip({ item }: { item: ToastItem }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.94, transition: { duration: 0.18 } }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
      className={`flex items-start gap-3 bg-white/95 backdrop-blur-sm border border-border border-l-4 ${borders[item.type]}
        rounded-xl px-4 py-3 shadow-lg min-w-[260px] max-w-[360px] group`}
    >
      {icons[item.type]}
      <span className="text-sm text-text flex-1 leading-snug">{item.message}</span>
      <button
        onClick={() => toastStore.dismiss(item.id)}
        className="text-text-muted hover:text-text transition-colors mt-0.5 opacity-0 group-hover:opacity-100"
      >
        <X size={14} />
      </button>
    </motion.div>
  );
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    const unsub = toastStore.subscribe(setToasts);
    return () => { unsub(); };
  }, []);

  return (
    <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2 items-end pointer-events-none">
      <AnimatePresence mode="popLayout">
        {toasts.map(t => (
          <div key={t.id} className="pointer-events-auto">
            <ToastChip item={t} />
          </div>
        ))}
      </AnimatePresence>
    </div>
  );
}
