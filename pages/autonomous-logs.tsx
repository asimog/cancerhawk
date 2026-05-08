import { GetStaticProps } from 'next';
import Link from 'next/link';
import { getBackendUrl, fetchWithTimeout } from '@/lib/blocks';
import { useCallback, useEffect, useRef, useState } from 'react';

type LogEntry = {
  job_id: string;
  created_at: string;
  research_goal: string;
  status: string;
  stage: string;
  message: string;
  at: string;
  block?: number;
  market_price?: number;
};

type ApiJob = {
  job_id: string;
  created_at: string;
  research_goal: string;
  status: string;
  result?: {
    title?: string;
    market_price?: number;
    block?: number;
    stats?: {
      total_calls?: number;
      total_tokens?: number;
      total_cost_usd?: number;
      elapsed_seconds?: number;
    };
  };
  error?: string | null;
  events?: Array<{
    at?: string;
    stage?: string;
    message?: string;
    data?: Record<string, unknown> | null;
  }>;
  config?: Record<string, unknown>;
};

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks')).getBackendUrl() },
});

function modeLabel(config?: Record<string, unknown>): string {
  if (!config) return 'free';
  const mode = config.mode;
  if (mode === 'paid') return 'paid';
  if (typeof config?.api_key === 'string' && config.api_key.trim()) return 'paid';
  return 'free';
}

export default function AutonomousLogsPage({ backendUrl }: { backendUrl: string }) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const logRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    if (!backendUrl) return;
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await fetchWithTimeout(`${backendUrl}/api/jobs?limit=100`, {
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!res.ok) return;
      const data = await res.json();
      const jobs: ApiJob[] = data.jobs || [];
      const all: LogEntry[] = [];
      for (const job of jobs) {
        const events = job.events || [];
        for (const ev of events) {
          all.push({
            job_id: job.job_id,
            created_at: job.created_at,
            research_goal: job.research_goal,
            status: job.status,
            stage: ev.stage || 'event',
            message: ev.message || '',
            at: ev.at || job.created_at,
            block: job.result?.block != null ? Number(job.result.block) : undefined,
            market_price: job.result?.market_price,
          });
        }
        if (events.length === 0) {
          all.push({
            job_id: job.job_id,
            created_at: job.created_at,
            research_goal: job.research_goal,
            status: job.status,
            stage: 'created',
            message: `Job created — status: ${job.status}` + (job.error ? ` (error: ${String(job.error).slice(0, 120)})` : ''),
            at: job.created_at,
            block: job.result?.block != null ? Number(job.result.block) : undefined,
            market_price: job.result?.market_price,
          });
        }
      }
      all.sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime());
      setEntries(all);
      setError(null);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [backendUrl]);

  useEffect(() => {
    void load();
    const interval = window.setInterval(() => void load(), 6000);
    return () => {
      window.clearInterval(interval);
      abortRef.current?.abort();
    };
  }, [load]);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [entries.length, autoScroll]);

  const stageColor: Record<string, string> = {
    start: '#4caf50',
    hermes: '#2196f3',
    api_call: '#ff9800',
    done: '#4caf50',
    error: '#f44336',
    stopped: '#9e9e9e',
    created: '#607d8b',
    completed: '#4caf50',
    failed: '#f44336',
    published: '#9c27b0',
  };

  return (
    <div className="page">
      <header className="page-header">
        <h1 className="page-title">Autonomous Logs</h1>
        <p className="page-kicker">
          All CancerHawk autonomous runs, peer reviews, block creations, and rewards.
          This page displays every event — no user interaction, just the logs.
        </p>
      </header>

      <div style={{ marginBottom: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 14 }}>
          <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.currentTarget.checked)} />
          Auto-scroll
        </label>
        <button className="button" onClick={() => void load()} type="button" style={{ fontSize: 13, padding: '4px 10px' }}>
          Refresh
        </button>
        <span style={{ fontSize: 12, color: '#9e9e9e' }}>{entries.length} events</span>
      </div>

      {loading && <p className="muted">Loading logs…</p>}
      {error && (
        <div className="backend-offline">
          <p className="muted">Backend offline: {error}</p>
          <button className="button" onClick={() => void load()} type="button">Retry</button>
        </div>
      )}

      {!loading && !error && entries.length === 0 && (
        <p className="muted">No logs yet. Autonomous runs will appear here.</p>
      )}

      {!loading && entries.length > 0 && (
        <div className="autonomous-log" ref={logRef} style={{
          maxHeight: 'calc(100vh - 260px)',
          overflowY: 'auto',
          background: '#0a0a0a',
          border: '1px solid #333',
          borderRadius: 8,
          padding: '12px 16px',
          fontFamily: "'Consolas', 'Fira Code', monospace",
          fontSize: 13,
          lineHeight: 1.6,
        }}>
          {entries.map((entry, i) => {
            const color = stageColor[entry.stage] || '#888';
            const time = new Date(entry.at).toLocaleTimeString();
            const mode: string = 'free';
            return (
              <div key={`${entry.job_id}-${i}`} style={{
                padding: '4px 0',
                borderBottom: '1px solid #1a1a1a',
                display: 'flex',
                gap: 10,
                alignItems: 'flex-start',
              }}>
                <span style={{ color: '#555', minWidth: 70, flexShrink: 0 }}>{time}</span>
                <span style={{
                  color,
                  minWidth: 72,
                  flexShrink: 0,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  fontSize: 11,
                  letterSpacing: 0.5,
                }}>{entry.stage}</span>
                <span style={{
                  color: mode === 'paid' ? '#ffd700' : '#888',
                  minWidth: 38,
                  flexShrink: 0,
                  fontSize: 11,
                  fontWeight: 600,
                }}>{mode.toUpperCase()}</span>
                <span style={{
                  color: '#e0e0e0',
                  flex: 1,
                  wordBreak: 'break-word',
                }}>
                  <Link href={`/jobs/${entry.job_id}`} style={{ color: '#64b5f6', textDecoration: 'none', marginRight: 6 }}>
                    [{entry.job_id.slice(0, 8)}]
                  </Link>
                  {entry.message}
                  {entry.block != null && (
                    <span style={{ marginLeft: 8, color: '#81c784' }}>
                      Block #{entry.block}
                    </span>
                  )}
                  {typeof entry.market_price === 'number' && (
                    <span style={{ marginLeft: 8, color: '#ffd700' }}>
                      {(entry.market_price * 100).toFixed(0)}%
                    </span>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <footer className="page-footer" style={{ marginTop: 24 }}>
        <Link href="/jobs" className="footer-link">← Job Feed</Link>
        <Link href="/" className="footer-link">← Home</Link>
      </footer>
    </div>
  );
}
