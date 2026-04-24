import { Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Header() {
  const today = new Date().toLocaleDateString('en-IN', {
    weekday: 'short', month: 'short', day: 'numeric',
  });

  const [dark, setDark] = useState(() => document.documentElement.getAttribute('data-theme') === 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <header className="h-12 bg-white border-b border-border/60 flex items-center justify-between px-6 sticky top-0 z-20">
      {/* Left: data freshness indicator */}
      <div className="flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-profit animate-pulse" />
        <span className="text-xs text-text-muted font-medium">Live · {today}</span>
      </div>

      {/* Right */}
      <div className="flex items-center gap-1.5">
        <button
          onClick={() => setDark(d => !d)}
          className="p-2 rounded-lg hover:bg-bg-card transition-colors"
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {dark
            ? <Sun size={16} className="text-warning" />
            : <Moon size={16} className="text-text-muted" />}
        </button>

        <div
          className="w-8 h-8 rounded-full bg-primary-light flex items-center justify-center ring-1 ring-primary/20 cursor-pointer hover:ring-primary/40 transition-all ml-1"
          title="Profile"
        >
          <span className="text-primary font-bold text-xs">FQ</span>
        </div>
      </div>
    </header>
  );
}
