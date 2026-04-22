interface SkeletonProps {
  className?: string;
  rounded?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  style?: React.CSSProperties;
}

export function Skeleton({ className = '', rounded = 'md', style }: SkeletonProps) {
  const r = { sm: 'rounded-sm', md: 'rounded-md', lg: 'rounded-lg', xl: 'rounded-xl', full: 'rounded-full' }[rounded];
  return <div className={`skeleton-shimmer ${r} ${className}`} style={style} />;
}

export function MetricCardSkeleton() {
  return (
    <div className="bg-white rounded-2xl border border-border p-5 space-y-3">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-28" rounded="md" />
        <Skeleton className="h-5 w-5" rounded="md" />
      </div>
      <Skeleton className="h-9 w-36" rounded="lg" />
      <Skeleton className="h-10 w-full" rounded="lg" />
    </div>
  );
}

export function CardSkeleton({ lines = 4 }: { lines?: number }) {
  return (
    <div className="bg-white rounded-2xl border border-border p-5 space-y-3">
      <Skeleton className="h-5 w-40" rounded="md" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="h-4 w-full" rounded="md" style={{ width: `${70 + (i % 3) * 10}%` } as React.CSSProperties} />
      ))}
    </div>
  );
}

export function TableRowSkeleton({ cols = 5 }: { cols?: number }) {
  return (
    <div className="flex gap-4 px-4 py-3 border-b border-border-light">
      {Array.from({ length: cols }).map((_, i) => (
        <Skeleton key={i} className="h-4 flex-1" rounded="md" />
      ))}
    </div>
  );
}
