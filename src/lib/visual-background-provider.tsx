'use client';

import { createContext, useCallback, useContext, useMemo, useState } from 'react';

type VisualBackgroundContextValue = {
  backgroundEnabled: boolean;
  toggleBackgroundEnabled: () => void;
  setBackgroundEnabled: (next: boolean) => void;
};

const STORAGE_KEY = 'cancerhawk.background-enabled';

const VisualBackgroundContext = createContext<VisualBackgroundContextValue | null>(null);

function readStoredBackgroundEnabled() {
  if (typeof window === 'undefined') return true;
  return window.localStorage.getItem(STORAGE_KEY) !== 'false';
}

export function VisualBackgroundProvider({ children }: { children: React.ReactNode }) {
  const [backgroundEnabled, setBackgroundEnabledState] = useState(readStoredBackgroundEnabled);

  const setBackgroundEnabled = useCallback((next: boolean) => {
    setBackgroundEnabledState(next);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, String(next));
    }
  }, []);

  const toggleBackgroundEnabled = useCallback(() => {
    setBackgroundEnabledState((current) => {
      const next = !current;
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(STORAGE_KEY, String(next));
      }
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ backgroundEnabled, toggleBackgroundEnabled, setBackgroundEnabled }),
    [backgroundEnabled, setBackgroundEnabled, toggleBackgroundEnabled],
  );

  return (
    <VisualBackgroundContext.Provider value={value}>
      {children}
    </VisualBackgroundContext.Provider>
  );
}

export function useVisualBackground() {
  const context = useContext(VisualBackgroundContext);
  if (!context) throw new Error('useVisualBackground must be used inside VisualBackgroundProvider');
  return context;
}
