import type { GetStaticProps } from 'next';
import { Nav } from '@/components/nav';

export const getStaticProps: GetStaticProps<{ backendUrl: string }> = async () => ({
  props: { backendUrl: (await import('@/lib/blocks')).getBackendUrl() },
});

export default function RunResearchPage({ backendUrl }: { backendUrl: string }) {
  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Railway backend</p>
      <h1 className="page-title">Run Research</h1>
      <section className="panel">
        <p>This Vercel app displays the research front-end. The CancerHawk backend should run on Railway and expose the FastAPI app currently served locally.</p>
        <p>Configure <code>NEXT_PUBLIC_BACKEND_URL</code> in Vercel after the Railway service is live. Current target: <code>{backendUrl}</code></p>
        <a className="button" href={backendUrl} target="_blank" rel="noreferrer">Open backend</a>
      </section>
      <iframe className="backend-frame" src={backendUrl} title="CancerHawk backend" />
    </div>
  );
}
