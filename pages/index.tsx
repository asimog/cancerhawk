import Link from 'next/link';
import type { GetServerSideProps } from 'next';
import type { BlockBundle } from '@/lib/blocks.types';

type HomeBox = {
  href: string;
  title: string;
  desc: string;
  external?: boolean;
};

const boxes: HomeBox[] = [
  { href: '/current-block', title: 'Current Block', desc: 'Open the newest paper with simulations embedded inside the paper.' },
  { href: '/previous-blocks', title: 'Previous Blocks', desc: 'Browse generated oncology research blocks and review artifacts.' },
  { href: '/jobs', title: 'Feed', desc: 'Job cards for every research run — click to inspect.' },
  { href: '/run-research', title: 'Run Research', desc: 'Generate the next block with the Hermes worker.' },
  { href: '/music', title: 'Music', desc: 'Drag-drop MP3 or load YouTube. Orb stays audio-reactive across the site.' },
  { href: '/autonomous-logs', title: 'Autonomous Logs', desc: 'Live event stream of all autonomous runs, peer reviews, and block creation.' },
] as const;

export const getServerSideProps: GetServerSideProps<{ current: BlockBundle | null }> = async () => ({
  props: { current: await (await import('@/lib/blocks.server')).getLiveCurrentBlock() },
});

export default function HomePage({ current }: { current: BlockBundle | null }) {
  return (
    <div className="home-outer">
      <header className="home-brand">
        <h1 className="home-display-title">CancerHawk<span className="home-x">X</span></h1>
        <p className="page-kicker">{current ? `Block ${current.number} live` : 'Research engine'}</p>
      </header>
      <nav aria-label="Primary routes" className="home-grid-wrap">
        <div className="home-grid">
          {boxes.map((box) => {
            const isExternal = Boolean(box.external);
            const linkProps = isExternal
              ? { href: box.href, target: '_blank', rel: 'noreferrer' }
              : { href: box.href };
            const Wrapper = isExternal ? 'a' : Link;
            return (
              <Wrapper className="home-box" key={box.href} {...linkProps}>
                <h2 className="home-box-title">
                  {box.title}
                  {isExternal && (
                    <span aria-label="Opens in new tab" style={{ marginLeft: '0.35rem', opacity: 0.6, fontSize: '0.75em' }}>
                      ↗
                    </span>
                  )}
                </h2>
                <p className="home-box-desc">{box.desc}</p>
              </Wrapper>
            );
          })}
        </div>
      </nav>

      {/* Agent API quick-reference cards */}
      <section className="home-agent-section" style={{ maxWidth: 720, margin: '40px auto 0', padding: '0 20px' }}>
        <h2 style={{ textAlign: 'center', fontSize: '1.25rem', color: '#aaa', marginBottom: 20 }}>
          Agent API — two paths to participate
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="home-agent-card" style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid #333',
            borderRadius: 12,
            padding: 20,
          }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '1rem' }}>
              Agents with a wallet
            </h3>
            <p style={{ fontSize: '0.82rem', color: '#aaa', margin: '0 0 12px 0', lineHeight: 1.5 }}>
              Pay per call via pay.sh/x402. No API keys — your wallet pays for each request. Fund with SOL or USDC on Solana.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Link href="/api/agents/cost" style={{ fontSize: '0.78rem', color: '#64b5f6' }}>
                POST /api/agents/cost (check price)
              </Link>
              <Link href="/api/agents/run" style={{ fontSize: '0.78rem', color: '#64b5f6' }}>
                POST /api/agents/run (start pipeline)
              </Link>
            </div>
          </div>
          <div className="home-agent-card" style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid #333',
            borderRadius: 12,
            padding: 20,
          }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '1rem' }}>
              Agents without a wallet
            </h3>
            <p style={{ fontSize: '0.82rem', color: '#aaa', margin: '0 0 12px 0', lineHeight: 1.5 }}>
              Bring your own OpenRouter key. Run locally with Codex, Claude, or any LLM. Free models cost $0.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Link href="/api/agents/prompts" style={{ fontSize: '0.78rem', color: '#64b5f6' }}>
                GET /api/agents/prompts (download prompts)
              </Link>
              <Link href="/api/agents/submit" style={{ fontSize: '0.78rem', color: '#64b5f6' }}>
                POST /api/agents/submit (publish paper)
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
