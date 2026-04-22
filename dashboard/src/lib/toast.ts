type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
}

type Listener = (toasts: ToastItem[]) => void;

class ToastStore {
  private items: ToastItem[] = [];
  private listeners: Set<Listener> = new Set();

  subscribe(fn: Listener) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    this.listeners.forEach(fn => fn([...this.items]));
  }

  show(message: string, type: ToastType = 'info', duration = 3500) {
    const id = Math.random().toString(36).slice(2);
    this.items = [...this.items, { id, message, type }];
    this.notify();
    setTimeout(() => this.dismiss(id), duration);
    return id;
  }

  dismiss(id: string) {
    this.items = this.items.filter(t => t.id !== id);
    this.notify();
  }

  success(msg: string) { return this.show(msg, 'success'); }
  error(msg: string) { return this.show(msg, 'error'); }
  info(msg: string) { return this.show(msg, 'info'); }
  warning(msg: string) { return this.show(msg, 'warning'); }
}

export const toast = new ToastStore();
