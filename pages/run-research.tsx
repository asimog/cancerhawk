import type { GetStaticProps } from 'next';
import { useRouter } from 'next/router';
import { useEffect, useMemo, useState } from 'react';
import { Nav } from '@/components/nav';

type ModelsPayload = {
  models: string[];
  defaults: Record<string, string>;
};

const roles = ['submitter', 'validator', 'compiler', 'archetype', 'topic_deriver'] as const;

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks')).getBackendUrl() },
});

export default function RunResearchPage({ backendUrl }: { backendUrl: string }) {
  const router = useRouter();
  const [apiKey, setApiKey] = useState('');
  const [goal, setGoal] = useState('');
  const [submitterCount, setSubmitterCount] = useState(3);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>({});
  const [status, setStatus] = useState('Checking Hermes worker...');
  const [workerReady, setWorkerReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [createdJobId, setCreatedJobId] = useState('');
  const workerUrl = useMemo(() => backendUrl.replace(/\/+$/, ''), [backendUrl]);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      setWorkerReady(false);
      if (!workerUrl) {
        setStatus('Hermes worker URL is not configured.');
        return;
      }
      try {
        const health = await fetch(`${workerUrl}/api/health`, { cache: 'no-store' });
        if (!health.ok) throw new Error(`health ${health.status}`);
        const modelResponse = await fetch(`${workerUrl}/api/models`, { cache: 'no-store' });
        if (!modelResponse.ok) throw new Error(`models ${modelResponse.status}`);
        const payload = (await modelResponse.json()) as ModelsPayload;
        if (cancelled) return;
        setModels(payload.models || []);
        setSelectedModels(payload.defaults || {});
        setWorkerReady(true);
        setStatus('Hermes worker ready.');
      } catch {
        if (!cancelled) {
          setWorkerReady(false);
          setStatus('Hermes worker is not reachable yet. Check the Railway deployment URL in Vercel env.');
        }
      }
    }
    void boot();
    return () => {
      cancelled = true;
    };
  }, [workerUrl]);

  function updateModel(role: string, value: string) {
    setSelectedModels((current) => ({ ...current, [role]: value }));
  }

  async function startRun() {
    if (!workerReady || !goal.trim() || isRunning) return;
    setIsRunning(true);
    setCreatedJobId('');
    setStatus('Creating job page...');

    try {
      const response = await fetch(`${workerUrl}/api/jobs/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: apiKey.trim(),
          research_goal: goal.trim(),
          n_submitters: submitterCount,
          auto_publish: true,
          git_push: true,
          ...selectedModels,
        }),
      });
      const payload = await response.json().catch(() => ({})) as {
        detail?: string;
        job_id?: string;
        job?: { job_id?: string };
      };
      if (!response.ok) {
        throw new Error(payload.detail || `Backend returned ${response.status}`);
      }
      const jobId = String(payload.job_id || payload.job?.job_id || '');
      if (!jobId) throw new Error('Backend did not return a job id');
      setCreatedJobId(jobId);
      setStatus('Job page created. Opening live job card...');
      await router.push(`/jobs/${jobId}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Run failed to start.');
      setIsRunning(false);
    }
  }

  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Hermes worker</p>
      <h1 className="page-title">Run Research</h1>

      <div className="run-grid">
        <section className="panel run-form">
          <p className="run-status">{status}</p>
          <label>
            OpenRouter API key
            <input autoComplete="off" onChange={(event) => setApiKey(event.target.value)} placeholder="sk-or-v1-..." type="password" value={apiKey} />
          </label>
          <label>
            Research goal
            <textarea onChange={(event) => setGoal(event.target.value)} placeholder="A focused oncology research question for the next CancerHawk block." value={goal} />
          </label>
          <div className="run-models">
            {roles.map((role) => (
              <label key={role}>
                {role.replace('_', ' ')}
                <select onChange={(event) => updateModel(role, event.target.value)} value={selectedModels[role] || models[0] || ''}>
                  {models.map((model) => <option key={model} value={model}>{model}</option>)}
                </select>
              </label>
            ))}
          </div>
          <label>
            Submitter count
            <input max={8} min={1} onChange={(event) => setSubmitterCount(Number(event.target.value) || 3)} type="number" value={submitterCount} />
          </label>
          <div className="run-actions">
            <button className="button" disabled={!workerReady || !apiKey.trim() || !goal.trim() || isRunning} onClick={startRun} type="button">
              {isRunning ? 'Creating job...' : 'Run CancerHawk'}
            </button>
            {createdJobId && <a className="button" href={`/jobs/${createdJobId}`}>Open job</a>}
          </div>
        </section>

        <section className="panel">
          <div className="run-job-preview">
            <span className="badge badge-pending">job card</span>
            <h2>Runs open in their own live job page.</h2>
            <p className="muted">
              Start a research block here. CancerHawk creates the job card first, then the full paper, peer review,
              simulations, token stats, and publish log stream into that page.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
