import type { GetStaticProps } from 'next';
import { Nav } from '@/components/nav';
import type { BlockBundle } from '@/lib/blocks.types';
import { Markdown } from '@/lib/markdown';

export const getStaticProps: GetStaticProps<{ block: BlockBundle | null }> = async () => ({
  props: { block: (await import('@/lib/blocks.server')).getCurrentBlock() },
});

export default function CurrentBlockPage({ block }: { block: BlockBundle | null }) {
  if (!block) {
    return <div className="page"><Nav /><h1 className="page-title">No block yet</h1></div>;
  }
  const simulations = block.analysis.simulations || [];
  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Current Block · {Math.round(block.meta.market_price * 100)}% synthesis price</p>
      <h1 className="page-title">{block.meta.title}</h1>
      <article className="paper panel">
        <Markdown content={block.paper} skipTitle />
        {simulations.length > 0 && (
          <section>
            <h2>Research Simulations</h2>
            <p>These simulations are part of the paper itself. They are not a separate route or standalone result.</p>
            <div className="sim-grid">
              {simulations.map((simulation) => (
                <div className="panel" key={simulation.id}>
                  <h3>{simulation.title}</h3>
                  <p>{simulation.description}</p>
                  <p>{simulation.rationale}</p>
                  <ul>{(simulation.expected_metrics || []).map((metric) => <li key={metric}>{metric}</li>)}</ul>
                </div>
              ))}
            </div>
          </section>
        )}
      </article>
    </div>
  );
}
