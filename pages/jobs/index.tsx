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
  props: { backendUrl: await getBackendUrl() },
});

export default function JobsPage({ backendUrl }: { backendUrl: string }) {
  // Client-side fetch — the job list changes frequently
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${backendUrl}/api/jobs`, { cache: 'no-store' });
        if (res.ok) {
          const data = await res.json();
          setJobs(data.jobs || []);
        }
      } catch {
        // backend may be offline
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
      {!loading && jobs.length === 0 && (
        <p className="muted">No jobs yet. Run a research block to create one.</p>
      )}

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
    </div>
  );
}
