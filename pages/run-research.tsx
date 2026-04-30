import type { GetStaticProps } from 'next';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Nav } from '@/components/nav';

type ModelsPayload = {
  models: string[];
  defaults: Record<string, string>;
};

type RunEvent = {
  stage?: string;
  message?: string;
  data?: Record<string, any>;
};

const roles = ['submitter', 'validator', 'compiler', 'archetype', 'topic_deriver'] as const;

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks')).getBackendUrl() },
});

function wsUrlFromHttp(url: string) {
  const parsed = new URL(url);
  parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
  parsed.pathname = '/ws/run';
  parsed.search = '';
  parsed.hash = '';
  return parsed.toString();
}

export default function RunResearchPage({ backendUrl }: { backendUrl: string }) {
  const [apiKey, setApiKey] = useState('');
  const [goal, setGoal] = useState('');
  const [submitterCount, setSubmitterCount] = useState(3);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>({});
  const [status, setStatus] = useState('Checking Railway worker...');
  const [workerReady, setWorkerReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [resultUrl, setResultUrl] = useState('');
  const socketRef = useRef<WebSocket | null>(null);
  const workerUrl = useMemo(() => backendUrl.replace(/\/+$/, ''), [backendUrl]);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      setWorkerReady(false);
      if (!workerUrl) {
        setStatus('Railway worker URL is not configured.');
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
        setStatus('Railway worker ready.');
      } catch {
        if (!cancelled) {
          setWorkerReady(false);
          setStatus('Railway worker is not reachable yet. Check the Railway deployment URL in Vercel env.');
        }
      }
    }
    void boot();
    return () => {
      cancelled = true;
      socketRef.current?.close();
    };
  }, [workerUrl]);

  function updateModel(role: string, value: string) {
    setSelectedModels((current) => ({ ...current, [role]: value }));
  }

  function startRun() {
    if (!workerReady || !apiKey.trim() || !goal.trim() || isRunning) return;
    socketRef.current?.close();
    setEvents([]);
    setResultUrl('');
    setIsRunning(true);
    setStatus('Starting CancerHawk run on Railway...');

    const ws = new WebSocket(wsUrlFromHttp(workerUrl));
    socketRef.current = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({
        api_key: apiKey.trim(),
        research_goal: goal.trim(),
        n_submitters: submitterCount,
        auto_publish: true,
        git_push: true,
        ...selectedModels,
      }));
    };
    ws.onmessage = (message) => {
      let event: RunEvent;
      try {
        event = JSON.parse(message.data) as RunEvent;
      } catch {
        event = { stage: 'log', message: String(message.data) };
      }
      setEvents((current) => [...current.slice(-180), event]);
      if (event.stage === 'done') {
        const url = String(event.data?.result_url || '');
        setResultUrl(url.startsWith('http') ? url : `/${url.replace(/^\/+/, '')}`);
        setStatus('Research block complete.');
        setIsRunning(false);
      }
      if (event.stage === 'error') {
        setStatus(event.message || 'Run failed.');
        setIsRunning(false);
      }
    };
    ws.onerror = () => {
      setStatus('WebSocket connection to Railway failed.');
      setIsRunning(false);
    };
    ws.onclose = () => {
      setIsRunning(false);
    };
  }

  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Railway worker</p>
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
              {isRunning ? 'Running...' : 'Run CancerHawk'}
            </button>
            {resultUrl && <a className="button" href={resultUrl}>Open block</a>}
          </div>
        </section>

        <section className="panel">
          <div className="run-log" aria-live="polite">
            {events.length === 0 ? (
              <div className="run-log-row"><span className="run-log-stage">ready</span><span>Waiting for a run.</span></div>
            ) : events.map((event, index) => (
              <div className="run-log-row" key={`${event.stage || 'event'}-${index}`}>
                <span className="run-log-stage">{event.stage || 'event'}</span>
                <span>{event.message || ''}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
