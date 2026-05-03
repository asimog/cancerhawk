import Link from 'next/link';
import type { GetStaticProps } from 'next';
import type { BlockBundle } from '@/lib/blocks';

const boxes = [
  { href: '/current-block', title: 'Current Block', desc: 'Open the newest paper with simulations embedded inside the paper.' },
  { href: '/previous-blocks', title: 'Previous Blocks', desc: 'Browse generated oncology research blocks and review artifacts.' },
  { href: '/jobs', title: 'Feed', desc: 'Job cards for every research run — click to inspect.' },
  { href: 'https://hypermyths.com', title: 'HyperMythX', desc: 'Explore the HyperMyth reality-expansion engine.', external: true },
  { href: '/run-research', title: 'Run Research', desc: 'Generate the next block with the Hermes worker.' },
  { href: '/music', title: 'Music', desc: 'Keep the global audio-reactive orb alive across the whole app.' },
] as const;

export const getStaticProps: GetStaticProps<{ current: BlockBundle | null }> = async () => ({
  props: { current: (await import('@/lib/blocks')).getCurrentBlock() },
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
            const isExternal = (box as any).external;
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
    </div>
  );
}
