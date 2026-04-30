import Link from 'next/link';
import type { GetStaticProps } from 'next';
import type { BlockBundle } from '@/lib/blocks';

const boxes = [
  { href: '/current-block', title: 'Current Block', desc: 'Open the newest paper with simulations embedded inside the paper.' },
  { href: '/previous-blocks', title: 'Previous Blocks', desc: 'Browse generated oncology research blocks and review artifacts.' },
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
        <h1 className="home-display-title">CancerHawk<span className="home-blink-x">X</span></h1>
        <p className="page-kicker">{current ? `Block ${current.number} live` : 'Research engine'}</p>
      </header>
      <nav aria-label="Primary routes" className="home-grid-wrap">
        <div className="home-grid">
          {boxes.map((box) => (
            <Link className="home-box" href={box.href} key={box.href}>
              <h2 className="home-box-title">{box.title}</h2>
              <p className="home-box-desc">{box.desc}</p>
            </Link>
          ))}
        </div>
      </nav>
    </div>
  );
}
