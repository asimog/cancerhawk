import { GetStaticProps } from 'next';
import Link from 'next/link';
import { getBackendUrl, fetchWithTimeout } from '@/lib/blocks';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Nav } from '@/components/nav';

type JobEvent = {
  at?: string;
  stage?: string;
  message?: string;
  data?: {
    call?: {
      role?: string;
      model?: string;
      prompt_tokens?: number;
      completion_tokens?: number;
      total_tokens?: number;
      cost_usd?: number;
      ok?: boolean;
      error?: string | null;
      prompt?: string;
      response?: string;
    };
    scores?: Record<string, number>;
    award_points?: number;
    rank?: number;
    winner_job_id?: string;
    leaderboard?: Array<{ rank: number; job_id: string; market_price: number; candidate_index?: number }>;
    [key: string]: unknown;
  } | null;
};

type ApiJob = {
  job_id: string;
  created_at: string;
  updated_at?: string;
  research_goal: string;
  status: string;
  result?: {
    title?: string;
    market_price?: number;
    block?: number | string | null;
    publication_outcome?: string;
    publication_batch_id?: string;
    candidate_index?: number;
    stats?: {
      total_calls?: number;
      total_tokens?: number;
      total_cost_usd?: number;
      elapsed_seconds?: number;
    };
  } | null;
  error?: string | null;
  events?: JobEvent[];
  config?: {
    models?: Record<string, string>;
    mode?: string;
    enable_paysh?: boolean;
    publication_batch_id?: string;
    candidate_index?: number;
    batch_size?: number;
    n_submitters?: number;
    publication_strategy?: string;
    [key: string]: unknown;
  };
};

type ExplorerBlock = {
  block: number;
  title: string;
  research_goal: string;
  market_price: number;
};

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks')).getBackendUrl() },
});

function modeLabel(job: ApiJob): string {
  if (job.config?.enable_paysh) return 'pay.sh';
  const model = job.config?.models?.submitter || '';
  if (model && model !== 'openrouter/free') return 'paid';
  return 'legacy';
}

function shortId(id: string) {
  return id.slice(0, 8);
}

function eventTime(value?: string) {
  if (!value) return '';
  return new Date(value).toLocaleTimeString();
}

function isValidatorEvent(event: JobEvent) {
  const stage = event.stage || '';
  const role = event.data?.call?.role || '';
  return (
    stage === 'validate' ||
    stage === 'review' ||
    stage === 'review_start' ||
    stage === 'review_complete' ||
    stage === 'block_validator' ||
    role.includes('validator') ||
    role.includes('peer_review') ||
    role.includes('archetype')
  );
}

function batchId(job: ApiJob) {
  return job.config?.publication_batch_id || job.result?.publication_batch_id || 'manual';
}

function marketPct(value?: number) {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : '--';
}

export default function AutonomousLogsPage({ backendUrl }: { backendUrl: string }) {
  const [jobs, setJobs] = useState<ApiJob[]>([]);
  const [blocks, setBlocks] = useState<ExplorerBlock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showValidatorsOnly, setShowValidatorsOnly] = useState(false);
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
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      const nextJobs: ApiJob[] = data.jobs || [];
      setJobs(nextJobs);

      const discoveredBlocks = Array.from(new Set(
        nextJobs
          .map((job) => Number(job.result?.block))
          .filter((block) => Number.isInteger(block) && block > 0),
      )).sort((a, b) => b - a);
      const fallbackBlocks = discoveredBlocks.length ? discoveredBlocks : Array.from({ length: 10 }, (_, index) => index + 1).reverse();
      const blockPayloads = await Promise.all(
        fallbackBlocks.slice(0, 12).map(async (block) => {
          try {
            const blockRes = await fetchWithTimeout(`${backendUrl}/api/blocks/${block}`, { cache: 'no-store', timeout: 5000 });
            if (!blockRes.ok) return null;
            const payload = await blockRes.json();
            return {
              block,
              title: payload.meta?.title || payload.meta?.paper_title || `Block ${block}`,
              research_goal: payload.meta?.research_goal || '',
              market_price: Number(payload.meta?.market_price || payload.analysis?.market_price || 0),
            } as ExplorerBlock;
          } catch {
            return null;
          }
        }),
      );
      setBlocks(blockPayloads.filter((block): block is ExplorerBlock => Boolean(block)));
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

  const batches = useMemo(() => {
    const grouped = new Map<string, ApiJob[]>();
    for (const job of jobs) {
      const key = batchId(job);
      grouped.set(key, [...(grouped.get(key) || []), job]);
    }
    return Array.from(grouped.entries()).map(([id, group]) => ({
      id,
      jobs: group.sort((a, b) => Number(a.config?.candidate_index || 99) - Number(b.config?.candidate_index || 99)),
    }));
  }, [jobs]);

  const latestBlock = blocks[0];
  const paidJobs = jobs.filter((job) => modeLabel(job) !== 'legacy').length;
  const validatorEvents = jobs.reduce((count, job) => count + (job.events || []).filter(isValidatorEvent).length, 0);

  return (
    <div className="page autonomous-explorer-page">
      <Nav />
      <header className="page-header explorer-header">
        <div>
          <h1 className="page-title">Autonomous Block Explorer</h1>
          <p className="page-kicker">Railway worker logs, validator traces, candidate awards, and publication picks</p>
        </div>
        <button className="button" onClick={() => void load()} type="button">Refresh</button>
      </header>

      <section className="explorer-stats">
        <div className="explorer-stat">
          <span>Latest Block</span>
          <strong>{latestBlock ? `#${latestBlock.block}` : '--'}</strong>
        </div>
        <div className="explorer-stat">
          <span>Tracked Jobs</span>
          <strong>{jobs.length}</strong>
        </div>
        <div className="explorer-stat">
          <span>Paid / Pay.sh Jobs</span>
          <strong>{paidJobs}</strong>
        </div>
        <div className="explorer-stat">
          <span>Validator Events</span>
          <strong>{validatorEvents}</strong>
        </div>
      </section>

      {loading && <p className="muted">Loading autonomous logs...</p>}
      {error && (
        <div className="backend-offline">
          <p className="muted">Backend offline: {error}</p>
          <button className="button" onClick={() => void load()} type="button">Retry</button>
        </div>
      )}

      {!loading && !error && (
        <>
          <section className="job-section">
            <div className="explorer-section-head">
              <h2>Published Blocks</h2>
              <span>{blocks.length} indexed</span>
            </div>
            <div className="block-explorer-grid">
              {blocks.map((block) => (
                <Link href={block.block === latestBlock?.block ? '/current-block' : `${backendUrl}/results/block-${block.block}/paper.html`} className="block-explorer-card" key={block.block}>
                  <span>Block {block.block}</span>
                  <strong>{block.title}</strong>
                  <p>{block.research_goal}</p>
                  <em>{marketPct(block.market_price)}</em>
                </Link>
              ))}
            </div>
          </section>

          <section className="job-section">
            <div className="explorer-section-head">
              <h2>Candidate Batches</h2>
              <label className="explorer-toggle">
                <input type="checkbox" checked={showValidatorsOnly} onChange={(event) => setShowValidatorsOnly(event.currentTarget.checked)} />
                Validator logs only
              </label>
            </div>

            <div className="batch-stack">
              {batches.map((batch) => (
                <article className="batch-panel" key={batch.id}>
                  <div className="batch-head">
                    <div>
                      <span>Batch</span>
                      <strong>{batch.id}</strong>
                    </div>
                    <em>{batch.jobs.length} jobs</em>
                  </div>

                  <div className="candidate-grid">
                    {batch.jobs.map((job) => {
                      const events = (job.events || []).filter((event) => !showValidatorsOnly || isValidatorEvent(event));
                      const model = job.config?.models?.submitter || 'unknown';
                      return (
                        <div className="candidate-card" key={job.job_id}>
                          <div className="candidate-topline">
                            <span className={`badge badge-${job.status}`}>{job.status}</span>
                            <span>{modeLabel(job).toUpperCase()}</span>
                          </div>
                          <h3>{job.research_goal}</h3>
                          <p className="candidate-meta">
                            <Link href={`/jobs/${job.job_id}`}>{shortId(job.job_id)}</Link>
                            {' '}· candidate {job.config?.candidate_index || job.result?.candidate_index || '--'}
                            {' '}· {model}
                          </p>
                          <div className="candidate-score-row">
                            <span>Market {marketPct(job.result?.market_price)}</span>
                            <span>{job.result?.block ? `Block ${job.result.block}` : job.result?.publication_outcome || 'running'}</span>
                          </div>

                          <div className="job-log-slice">
                            {events.length === 0 ? (
                              <p className="muted">No matching events yet.</p>
                            ) : events.slice(-18).map((event, index) => {
                              const call = event.data?.call;
                              return (
                                <details className={`event-row event-${event.stage || 'event'}`} key={`${event.at || ''}-${index}`} open={index >= events.slice(-18).length - 3}>
                                  <summary>
                                    <span>{eventTime(event.at)}</span>
                                    <strong>{event.stage || call?.role || 'event'}</strong>
                                    <em>{event.message || call?.role || ''}</em>
                                  </summary>
                                  {call && (
                                    <div className="event-call">
                                      <p>{call.role} · {call.model} · {call.total_tokens?.toLocaleString() || 0} tokens · ${Number(call.cost_usd || 0).toFixed(4)}</p>
                                      {call.error && <p className="job-error">{call.error}</p>}
                                      {isValidatorEvent(event) && (
                                        <pre>{JSON.stringify({ prompt: call.prompt, response: call.response }, null, 2)}</pre>
                                      )}
                                    </div>
                                  )}
                                  {event.data?.scores && <pre>{JSON.stringify(event.data.scores, null, 2)}</pre>}
                                  {event.data?.leaderboard && <pre>{JSON.stringify(event.data.leaderboard, null, 2)}</pre>}
                                  {typeof event.data?.award_points === 'number' && (
                                    <p className="award-line">Award: {event.data.award_points} points</p>
                                  )}
                                </details>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      )}

      <footer className="page-footer">
        <Link href="/jobs" className="footer-link">Job Feed</Link>
        <Link href="/" className="footer-link">Home</Link>
      </footer>
    </div>
  );
}
