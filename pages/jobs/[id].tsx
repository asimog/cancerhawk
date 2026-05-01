import { GetStaticPaths, GetStaticProps } from 'next';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { getBackendUrl } from '@/lib/blocks';

import { useState } from 'react';

type Job = {
  job_id: string;
  created_at: string;
  research_goal: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  config?: Record<string, any>;
  result?: Record<string, any>;
  error?: string | null;
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

  if (router.isFallback) {
    return <p className="muted">Loading job…</p>;
  }

  if (!job) {
    return (
      <div className="page">
        <h1>Job not found</h1>
        <p className="muted">This job does not exist or has been removed.</p>
        <Link href="/jobs" className="button">Back to feed</Link>
      </div>
    );
  }

  const statusClass = `badge badge-${job.status}`;

  return (
    <div className="page job-detail">
      <header className="job-header">
        <Link href="/jobs" className="back-link">← All jobs</Link>
        <div className="job-header-row">
          <span className={statusClass}>{job.status}</span>
          <span className="job-date">{new Date(job.created_at).toLocaleString()}</span>
        </div>
        <h1 className="job-goal">{job.research_goal}</h1>
        <p className="job-id">Job ID: {job.job_id}</p>
      </header>

      {job.config && (
        <section className="job-section">
          <h2>Configuration</h2>
          <pre className="job-json">{JSON.stringify(job.config, null, 2)}</pre>
        </section>
      )}

      {job.status === 'completed' && job.result && (
        <section className="job-section">
          <h2>Result</h2>
          {job.result.title && <h3>{job.result.title}</h3>}
          {job.result.market_price != null && (
            <p>Market price: <strong>{(job.result.market_price * 100).toFixed(0)}%</strong></p>
          )}
          {job.result.block && (
            <p>
              Block: <Link href={`/results/block-${job.result.block}/paper.html`}>block-{job.result.block}</Link>
            </p>
          )}
          {job.result.stats && (
            <div className="job-stats">
              <div className="stat-card">
                <div className="stat-label">Calls</div>
                <div className="stat-value">{job.result.stats.total_calls?.toLocaleString() || '—'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Tokens</div>
                <div className="stat-value">{job.result.stats.total_tokens?.toLocaleString() || '—'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Cost</div>
                <div className="stat-value">${job.result.stats.total_cost_usd?.toFixed(4) || '0.00'}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Elapsed</div>
                <div className="stat-value">{job.result.stats.elapsed_seconds?.toFixed(0) || '—'}s</div>
              </div>
            </div>
          )}
        </section>
      )}

      {job.status === 'failed' && job.error && (
        <section className="job-section job-error-section">
          <h2>Error</h2>
          <pre className="job-error">{job.error}</pre>
        </section>
      )}

      <footer className="page-footer">
        <Link href="/jobs" className="footer-link">← Back to Feed</Link>
        <Link href="/" className="footer-link">← Back to Home</Link>
      </footer>
    </div>
  );
}
