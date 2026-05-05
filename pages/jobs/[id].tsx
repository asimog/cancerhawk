import { GetStaticPaths, GetStaticProps } from 'next';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { getBackendUrl, fetchWithTimeout } from '@/lib/blocks';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Nav } from '@/components/nav';

type JobEvent = {
  at?: string;
  stage?: string;
  message?: string;
  data?: Record<string, unknown> | null;
};

type Job = {
  job_id: string;
  created_at: string;
  research_goal: string;
  status: 'pending' | 'running' | 'completed' | 'published' | 'failed';
  config?: Record<string, unknown>;
  result?: {
    title?: string;
    market_price?: number;
    block?: string | number;
    stats?: {
      total_calls?: number;
      total_tokens?: number;
      total_cost_usd?: number;
      elapsed_seconds?: number;
    };
  };
  error?: string | null;
  events?: JobEvent[];
};

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export const getStaticPaths: GetStaticPaths = async () => ({
  paths: [],
  fallback: 'blocking',
});

export const getStaticProps: GetStaticProps<{ job: Job | null; backendUrl: string }> = async (context) => {
  const backendUrl = await getBackendUrl();
  const jobId = context.params?.id as string;
  try {
    const res = await fetchWithTimeout(`${backendUrl}/api/jobs/${jobId}`, { cache: 'no-store' });
    if (!res.ok) return { props: { job: null, backendUrl } };
    const job = await res.json();
    return { props: { job, backendUrl } };
  } catch {
    return { props: { job: null, backendUrl } };
  }
};

export default function JobDetailPage({ job, backendUrl }: { job: Job | null; backendUrl: string }) {
  const router = useRouter();
  const [liveJob, setLiveJob] = useState<Job | null>(job);
  const [pollError, setPollError] = useState('');
  const workerUrl = useMemo(() => backendUrl.replace(/\/+$/, ''), [backendUrl]);
  const jobId = typeof router.query.id === 'string' ? router.query.id : job?.job_id || '';
  const logRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const statusRef = useRef(liveJob?.status);
  const timerRef = useRef<number>(0);

  useEffect(() => {
    setLiveJob(job);
  }, [job]);

  useEffect(() => {
    statusRef.current = liveJob?.status;
  }, [liveJob?.status]);

  const load = useCallback(async () => {
    if (!workerUrl || !jobId) return;
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetchWithTimeout(`${workerUrl}/api/jobs/${jobId}`, {
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const payload = (await response.json()) as Job;
      setLiveJob(payload);
      setPollError('');
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      setPollError(error instanceof Error ? error.message : String(error));
    }
  }, [jobId, workerUrl]);

  useEffect(() => {
    if (!workerUrl || !jobId) return;

    function schedule() {
      const delay = statusRef.current === 'running' || statusRef.current === 'pending' ? 1800 : 6000;
      timerRef.current = window.setTimeout(() => {
        void load();
        schedule();
      }, delay);
    }

    void load();
    schedule();

    return () => {
      window.clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [jobId, workerUrl, load]);

  // Auto-scroll log to bottom when events change
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [liveJob?.events?.length]);

  if (router.isFallback) {
    return <p className="muted">Loading job…</p>;
  }

  if (!liveJob) {
    return (
      <div className="page">
        <Nav />
        <h1>Job not found</h1>
        <p className="muted">This job does not exist or has been removed.</p>
        <Link href="/jobs" className="button">Back to feed</Link>
      </div>
    );
  }

  const statusClass = `badge badge-${liveJob.status}`;
  const events = liveJob.events || [];

  return (
    <div className="page job-detail">
      <Nav />
      <header className="job-header">
        <Link href="/jobs" className="back-link">← All jobs</Link>
        <div className="job-header-row">
          <span className={statusClass}>{liveJob.status}</span>
          <span className="job-date">{new Date(liveJob.created_at).toLocaleString()}</span>
        </div>
        <h1 className="job-goal">{liveJob.research_goal}</h1>
        <p className="job-id">Job ID: {liveJob.job_id}</p>
        {pollError && (
          <p className="job-error">
            Live refresh paused: {pollError}
            <button
              className="button"
              onClick={() => {
                setPollError('');
                void load();
              }}
              style={{ marginLeft: '0.5rem' }}
              type="button"
            >
              Retry
            </button>
          </p>
        )}
      </header>

      <section className="job-section">
        <h2>Run Log</h2>
        <div className="run-log job-live-log" aria-live="polite" ref={logRef}>
          {events.length === 0 ? (
            <div className="run-log-row"><span className="run-log-stage">created</span><span>Waiting for the first event.</span></div>
          ) : events.map((event, index) => (
            <div className="run-log-row" key={`${event.at || ''}-${event.stage || 'event'}-${index}`}>
              <span className="run-log-stage">{event.stage || 'event'}</span>
              <span>{event.message || ''}</span>
            </div>
          ))}
        </div>
      </section>

      {liveJob.config && (
        <section className="job-section">
          <h2>Configuration</h2>
          <pre className="job-json">{JSON.stringify(liveJob.config, null, 2)}</pre>
        </section>
      )}

      {(liveJob.status === 'completed' || liveJob.status === 'published') && liveJob.result && (
        <section className="job-section">
          <h2>Result</h2>
          {liveJob.result.title && <h3>{String(liveJob.result.title)}</h3>}
          {typeof liveJob.result.market_price === 'number' && (
            <p>Market price: <strong>{(liveJob.result.market_price * 100).toFixed(0)}%</strong></p>
          )}
          {liveJob.result.block && (
            <p>
              Block: <Link href={`/results/block-${liveJob.result.block}/paper.html`}>block-{String(liveJob.result.block)}</Link>
            </p>
          )}
          {liveJob.result.stats && (
            <div className="job-stats">
              <div className="stat-card">
                <div className="stat-label">Calls</div>
                <div className="stat-value">{liveJob.result.stats.total_calls?.toLocaleString() || '—'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Tokens</div>
                <div className="stat-value">{liveJob.result.stats.total_tokens?.toLocaleString() || '—'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Cost</div>
                <div className="stat-value">${typeof liveJob.result.stats.total_cost_usd === 'number' ? liveJob.result.stats.total_cost_usd.toFixed(4) : '0.00'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Elapsed</div>
                <div className="stat-value">{typeof liveJob.result.stats.elapsed_seconds === 'number' ? liveJob.result.stats.elapsed_seconds.toFixed(0) : '—'}s</div>
              </div>
            </div>
          )}
        </section>
      )}

      {liveJob.status === 'failed' && liveJob.error && (
        <section className="job-section job-error-section">
          <h2>Error</h2>
          <pre className="job-error" dangerouslySetInnerHTML={{ __html: escapeHtml(liveJob.error) }} />
        </section>
      )}

      <footer className="page-footer">
        <Link href="/jobs" className="footer-link">← Back to Feed</Link>
        <Link href="/" className="footer-link">← Back to Home</Link>
      </footer>
    </div>
  );
}
