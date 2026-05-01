import { GetStaticPaths, GetStaticProps } from 'next';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { getBackendUrl } from '@/lib/blocks';
import { useEffect, useMemo, useState } from 'react';
import { Nav } from '@/components/nav';

type JobEvent = {
  at?: string;
  stage?: string;
  message?: string;
  data?: Record<string, any> | null;
};

type Job = {
  job_id: string;
  created_at: string;
  research_goal: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  config?: Record<string, any>;
  result?: Record<string, any>;
  error?: string | null;
  events?: JobEvent[];
};

export const getStaticPaths: GetStaticPaths = async () => ({
  paths: [],
  fallback: 'blocking',
});

export const getStaticProps: GetStaticProps<{ job: Job | null; backendUrl: string }> = async (context) => {
  const backendUrl = await getBackendUrl();
  const jobId = context.params?.id as string;
  try {
    const res = await fetch(`${backendUrl}/api/jobs/${jobId}`, { cache: 'no-store' });
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

  useEffect(() => {
    setLiveJob(job);
  }, [job]);

  useEffect(() => {
    if (!workerUrl || !jobId) return;
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(`${workerUrl}/api/jobs/${jobId}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Backend returned ${response.status}`);
        const payload = (await response.json()) as Job;
        if (!cancelled) {
          setLiveJob(payload);
          setPollError('');
        }
      } catch (error) {
        if (!cancelled) setPollError(error instanceof Error ? error.message : String(error));
      }
    }

    void load();
    const timer = window.setInterval(() => {
      if (!cancelled) void load();
    }, liveJob?.status === 'running' || liveJob?.status === 'pending' ? 1800 : 6000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId, liveJob?.status, workerUrl]);

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
        {pollError && <p className="job-error">Live refresh paused: {pollError}</p>}
      </header>

      <section className="job-section">
        <h2>Run Log</h2>
        <div className="run-log job-live-log" aria-live="polite">
          {events.length === 0 ? (
            <div className="run-log-row"><span className="run-log-stage">created</span><span>Waiting for the first event.</span></div>
          ) : events.map((event, index) => (
            <div className="run-log-row" key={`${event.stage || 'event'}-${event.at || index}`}>
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

      {liveJob.status === 'completed' && liveJob.result && (
        <section className="job-section">
          <h2>Result</h2>
          {liveJob.result.title && <h3>{liveJob.result.title}</h3>}
          {liveJob.result.market_price != null && (
            <p>Market price: <strong>{(liveJob.result.market_price * 100).toFixed(0)}%</strong></p>
          )}
          {liveJob.result.block && (
            <p>
              Block: <Link href={`/results/block-${liveJob.result.block}/paper.html`}>block-{liveJob.result.block}</Link>
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
                <div className="stat-value">${liveJob.result.stats.total_cost_usd?.toFixed(4) || '0.00'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Elapsed</div>
                <div className="stat-value">{liveJob.result.stats.elapsed_seconds?.toFixed(0) || '—'}s</div>
              </div>
            </div>
          )}
        </section>
      )}

      {liveJob.status === 'failed' && liveJob.error && (
        <section className="job-section job-error-section">
          <h2>Error</h2>
          <pre className="job-error">{liveJob.error}</pre>
        </section>
      )}

      <footer className="page-footer">
        <Link href="/jobs" className="footer-link">← Back to Feed</Link>
        <Link href="/" className="footer-link">← Back to Home</Link>
      </footer>
    </div>
  );
}
