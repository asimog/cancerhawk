import Link from 'next/link';
import { GetStaticProps } from 'next';
import { getBackendUrl } from '@/lib/blocks';

import { useState, useEffect } from 'react';

type Job = {
  job_id: string;
  created_at: string;
  research_goal: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  config?: Record<string, any>;
  result?: Record<string, any>;
  error?: string | null;
};

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks')).getBackendUrl() },
});

export default function JobsPage({ backendUrl }: { backendUrl: string }) {
  // Client-side fetch — the job list changes frequently
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${backendUrl}/api/jobs`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`Backend returned ${res.status}`);
        const data = await res.json();
        setJobs(data.jobs || []);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    if (backendUrl) void load();
  }, [backendUrl]);

  const statusBadge: Record<string, string> = {
    pending: 'badge-pending',
    running: 'badge-running',
    completed: 'badge-completed',
    failed: 'badge-failed',
  };

  return (
    <div className="page">
      <header className="page-header">
        <h1 className="page-title">Job Feed</h1>
        <p className="page-kicker">Every research run creates a job card — click to inspect.</p>
      </header>

      {loading && <p className="muted">Loading jobs…</p>}
      {!loading && error && (
        <div className="backend-offline">
          <p className="muted">Backend is offline: {error}</p>
          <p className="muted">Job cards will appear here when the Hermes worker is running.</p>
        </div>
      )}
      {!loading && !error && jobs.length === 0 && (
        <p className="muted">No jobs yet. Run a research block to create one.</p>
      )}

      {!loading && !error && jobs.length > 0 && (
        <div className="job-feed">
          {jobs.map((job) => (
            <Link
              key={job.job_id}
              href={`/jobs/${job.job_id}`}
              className="job-card"
            >
              <div className="job-card-top">
                <span className={`badge ${statusBadge[job.status] || 'badge-pending'}`}>
                  {job.status}
                </span>
                <span className="job-date">
                  {new Date(job.created_at).toLocaleString()}
                </span>
              </div>
              <h2 className="job-goal">{job.research_goal}</h2>
              {job.result?.title && (
                <p className="job-result-title">{job.result.title}</p>
              )}
              {job.error && (
                <p className="job-error">{job.error.slice(0, 120)}</p>
              )}
            </Link>
          ))}
        </div>
      )}

      <footer className="page-footer">
        <Link href="/" className="footer-link">← Back to Home</Link>
      </footer>
    </div>
  );
}
