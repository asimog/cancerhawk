import type { GetStaticProps } from 'next';
import { useRouter } from 'next/router';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Nav } from '@/components/nav';
import { fetchWithTimeout, validateWalletAddress, GIVEWELL_WALLET, GIVEWELL_URL } from '@/lib/blocks';

type ModelsPayload = {
  models: string[];
  defaults: Record<string, string>;
};

const roles = ['submitter', 'validator', 'compiler', 'archetype', 'topic_deriver'] as const;

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks.server')).getBackendUrl() },
})

export default function RunResearchPage({ backendUrl }: { backendUrl: string }) {
  const router = useRouter();
  const [apiKey, setApiKey] = useState('');
  const [goal, setGoal] = useState('');
  const [enablePaysh, setEnablePaysh] = useState(true);
  const [submitterCount, setSubmitterCount] = useState(3);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>({});
  const [status, setStatus] = useState('Checking Hermes worker...');
  const [workerReady, setWorkerReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [createdJobId, setCreatedJobId] = useState('');
  const [walletAddress, setWalletAddress] = useState('');
  const [walletError, setWalletError] = useState('');
  const workerUrl = useMemo(() => backendUrl.replace(/\/+$/, ''), [backendUrl]);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      setWorkerReady(false);
      if (!workerUrl) {
        setStatus('Hermes worker URL is not configured.');
        return;
      }
      try {
        const health = await fetchWithTimeout(`${workerUrl}/api/health`, { cache: 'no-store' });
        if (!health.ok) throw new Error(`health ${health.status}`);
        const modelResponse = await fetchWithTimeout(`${workerUrl}/api/models`, { cache: 'no-store' });
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

  const visibleModels = useMemo(() =>
    models.filter((m) => m !== 'openrouter/free'),
    [models]
  );

  function onWalletChange(value: string) {
    setWalletAddress(value);
    const result = validateWalletAddress(value);
    setWalletError(result.error || '');
  }

  async function startRun() {
    if (!workerReady || !goal.trim() || isRunning) return;

    const walletValidation = validateWalletAddress(walletAddress);
    if (!walletValidation.valid) {
      setWalletError(walletValidation.error || 'Invalid wallet address.');
      return;
    }

    const wallet = walletValidation.wallet;
    const useGiveWell = walletValidation.isDefault && !walletAddress.trim();

    const confirmed = window.confirm(
      `This will use paid worker models and may incur OpenRouter/pay.sh costs. Wallet: ${wallet.slice(0, 6)}...${wallet.slice(-4)}. Continue?`
    );
    if (!confirmed) return;

    setIsRunning(true);
    setCreatedJobId('');
    setStatus('Creating job page...');

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const idempotencyKey = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random()}`;

    try {
      const response = await fetchWithTimeout(`${workerUrl}/api/jobs/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          api_key: apiKey.trim(),
          research_goal: goal.trim(),
          n_submitters: Math.min(8, Math.max(1, Number(submitterCount) || 3)),
          auto_publish: true,
          git_push: true,
          enable_paysh: enablePaysh,
          idempotency_key: idempotencyKey,
          wallet_address: wallet,
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
      if (error instanceof Error && error.name === 'AbortError') {
        setStatus('Run cancelled.');
      } else {
        setStatus(error instanceof Error ? error.message : 'Run failed to start.');
      }
      setIsRunning(false);
    } finally {
      abortRef.current = null;
    }
  }

  const canSubmit = workerReady && goal.trim() && !isRunning && !walletError
    && visibleModels.length > 0;

  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Hermes worker</p>
      <h1 className="page-title">Run Research</h1>

      <div className="run-grid">
        <section className="panel run-form">
          <p className="run-status">{status}</p>

          <div className="mode-toggle" style={{ display: 'flex', gap: 0, marginBottom: 16 }}>
            <div className="button" style={{ background: '#1b5e20', border: '1px solid #333', flex: 1 }}>
              Paid Worker Mode · DeepSeek V4 Flash
            </div>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={enablePaysh}
              onChange={(e) => setEnablePaysh(e.currentTarget.checked)}
            />
            <span style={{ fontSize: 14 }}>
              Enable pay.sh enrichment{' '}
              <span style={{ color: '#888', fontSize: 12 }}>
                (calls Perplexity + web search for richer research — paid from CancerHawk wallet)
              </span>
            </span>
          </label>

          <label>
            OpenRouter API key (optional when Railway has server key)
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
                <select onChange={(event) => updateModel(role, event.target.value)} value={selectedModels[role] || (visibleModels[0] || '')}>
                  {visibleModels.map((model) => <option key={model} value={model}>{model}</option>)}
                </select>
              </label>
            ))}
          </div>
          <label>
            Submitter count
            <input max={8} min={1} onChange={(event) => setSubmitterCount(Math.min(8, Math.max(1, Number(event.target.value) || 3)))} type="number" value={submitterCount} />
          </label>
          <label>
            Wallet address (optional — Solana or Base)
            <input
              autoComplete="off"
              onChange={(event) => onWalletChange(event.target.value)}
              placeholder={GIVEWELL_WALLET}
              type="text"
              value={walletAddress}
            />
            {!walletAddress.trim() && (
              <span style={{ fontSize: 12, color: '#888' }}>
                Default: GiveWell charity.{' '}
                <a href={GIVEWELL_URL} target="_blank" rel="noreferrer" style={{ color: '#64b5f6' }}>
                  GiveWell Crypto ↗
                </a>
              </span>
            )}
            {walletError && <span style={{ color: '#ff6b6b', fontSize: '0.85rem' }}>{walletError}</span>}
          </label>
          <div className="run-actions">
            <button className="button" disabled={!canSubmit} onClick={startRun} type="button">
              {isRunning ? 'Creating job...' : 'Run CancerHawk (Paid)'}
            </button>
            {createdJobId && <a className="button" href={`/jobs/${createdJobId}`}>Open job</a>}
          </div>
        </section>

        <section className="panel">
          <div className="run-job-preview">
            <span className="badge badge-running">paid mode</span>
            <h2>Paid workers, pay.sh enrichment.</h2>
            <p className="muted">
              CancerHawk uses paid OpenRouter models and pay.sh enrichment for agent and subagent work. Provide a key to BYOK, or rely on the Railway server key when configured.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
