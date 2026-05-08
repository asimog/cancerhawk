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
  const [mode, setMode] = useState<'free' | 'paid'>('free');
  const [enablePaysh, setEnablePaysh] = useState(false);
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
    mode === 'paid' ? models : models.filter((m) => m === 'openrouter/free'),
    [models, mode]
  );

  function onWalletChange(value: string) {
    setWalletAddress(value);
    const result = validateWalletAddress(value);
    setWalletError(result.error || '');
  }

  async function startRun() {
    if (!workerReady || !goal.trim() || isRunning) return;

    if (mode === 'paid' && !apiKey.trim()) {
      setStatus('API key required in paid mode.');
      return;
    }

    const walletValidation = validateWalletAddress(walletAddress);
    if (!walletValidation.valid) {
      setWalletError(walletValidation.error || 'Invalid wallet address.');
      return;
    }

    const wallet = walletValidation.solana;
    const useGiveWell = walletValidation.isDefault && !walletAddress.trim();

    const confirmed = window.confirm(
      mode === 'paid'
        ? `This will use your OpenRouter API key and may incur costs. Wallet: ${wallet.slice(0, 6)}...${wallet.slice(-4)}. Continue?`
        : `This will run using free OpenRouter models at no cost. Wallet: ${wallet.slice(0, 6)}...${wallet.slice(-4)}. Continue?`
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
          api_key: mode === 'paid' ? apiKey.trim() : '',
          research_goal: goal.trim(),
          n_submitters: Math.min(8, Math.max(1, Number(submitterCount) || 3)),
          auto_publish: true,
          git_push: true,
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
    && (mode === 'free' || apiKey.trim());

  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Hermes worker</p>
      <h1 className="page-title">Run Research</h1>

      <div className="run-grid">
        <section className="panel run-form">
          <p className="run-status">{status}</p>

          <div className="mode-toggle" style={{ display: 'flex', gap: 0, marginBottom: 16 }}>
            <button
              className={`button ${mode === 'free' ? '' : 'button-outline'}`}
              onClick={() => { setMode('free'); setApiKey(''); }}
              type="button"
              style={{
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0,
                background: mode === 'free' ? '#1b5e20' : 'transparent',
                border: '1px solid #333',
                flex: 1,
              }}
            >
              Free Mode
            </button>
            <button
              className={`button ${mode === 'paid' ? '' : 'button-outline'}`}
              onClick={() => setMode('paid')}
              type="button"
              style={{
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
                background: mode === 'paid' ? '#1b5e20' : 'transparent',
                border: '1px solid #333',
                flex: 1,
              }}
            >
              Paid Mode
            </button>
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
            OpenRouter API key{mode === 'free' ? ' (optional)' : ''}
            <input autoComplete="off" onChange={(event) => setApiKey(event.target.value)} placeholder={mode === 'free' ? 'Leave empty for free tier' : 'sk-or-v1-...'} type="password" value={apiKey} />
          </label>
          {mode === 'free' && !apiKey.trim() && (
            <p style={{ fontSize: 12, color: '#888', marginTop: -8, marginBottom: 12 }}>
              Using CancerHawk server key — free models only.
            </p>
          )}
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
            Wallet address (optional — Solana)
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
              {isRunning ? 'Creating job...' : `Run CancerHawk (${mode === 'paid' ? 'Paid' : 'Free'})`}
            </button>
            {createdJobId && <a className="button" href={`/jobs/${createdJobId}`}>Open job</a>}
          </div>
        </section>

        <section className="panel">
          <div className="run-job-preview">
            <span className={`badge ${mode === 'paid' ? 'badge-running' : 'badge-pending'}`}>
              {mode === 'paid' ? 'paid mode' : 'free mode'}
            </span>
            <h2>{mode === 'paid' ? 'Your key, your model.' : 'Free models, zero cost.'}</h2>
            <p className="muted">
              {mode === 'paid'
                ? 'Provide your OpenRouter API key and select any model. Costs are charged to your account.'
                : 'No API key needed — CancerHawk uses its server key with free models only. Safe to explore and share.'}
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
