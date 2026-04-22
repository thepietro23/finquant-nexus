import { Search, Bell, Calendar, Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Header() {
  const today = new Date().toLocaleDateString('en-IN', {
    weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
  });

  const [dark, setDark] = useState(() => document.documentElement.getAttribute('data-theme') === 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <header className="h-16 bg-white/75 dark:bg-secondary/80 backdrop-blur-md border-b border-border/60 flex items-center justify-between px-6 sticky top-0 z-20 supports-[backdrop-filter]:bg-white/60">
      {/* Search */}
      <div className="flex items-center gap-2 bg-bg-card rounded-xl px-4 py-2 w-80 border border-border-light">
        <Search size={16} className="text-text-muted" />
        <input
          type="text"
          placeholder="Search stocks, modules..."
          className="bg-transparent text-sm outline-none w-full text-text placeholder:text-text-muted"
        />
      </div>

      {/* Right side */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <Calendar size={16} />
          <span>{today}</span>
        </div>

        <button
          onClick={() => setDark(d => !d)}
          className="p-2 rounded-xl hover:bg-bg-card transition-all duration-200 hover:scale-110 active:scale-95"
          title="Toggle dark mode"
        >
          {dark ? <Sun size={18} className="text-warning" /> : <Moon size={18} className="text-text-secondary" />}
        </button>

        <button className="relative p-2 rounded-xl hover:bg-bg-card transition-colors">
          <Bell size={18} className="text-text-secondary" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full animate-pulse" />
        </button>

        <div className="w-9 h-9 rounded-full bg-primary-light flex items-center justify-center ring-2 ring-primary/20 hover:ring-primary/40 transition-all duration-200 cursor-pointer">
          <span className="text-primary font-semibold text-sm">PP</span>
        </div>
      </div>
    </header>
  );
}
