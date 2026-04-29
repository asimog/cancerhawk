import Link from 'next/link';
import type { GetStaticProps } from 'next';
import { Nav } from '@/components/nav';
import type { BlockBundle } from '@/lib/blocks';

type ArchiveBlock = {
  number: number;
  title: string;
  marketPrice: number;
  summary: string;
};

function excerpt(markdown: string, maxLength = 280) {
  const text = markdown
    .split('\n')
    .filter((line) => !line.startsWith('#') && !line.startsWith('|') && line.trim())
    .join(' ')
    .replace(/\*\*/g, '')
    .trim();
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}...` : text;
}

export const getStaticProps: GetStaticProps<{ blocks: ArchiveBlock[] }> = async () => {
  const { getBlocks } = await import('@/lib/blocks');
  return {
    props: {
      blocks: getBlocks().map((block: BlockBundle) => ({
        number: block.number,
        title: block.meta.title,
        marketPrice: block.meta.market_price,
        summary: excerpt(block.paper),
      })),
    },
  };
};

export default function PreviousBlocksPage({ blocks }: { blocks: ArchiveBlock[] }) {
  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Archive</p>
      <h1 className="page-title">Previous Blocks</h1>
      <div className="block-grid">
        {blocks.map((block) => (
          <Link className="panel" href={block.number === blocks[0]?.number ? '/current-block' : `/results/block-${block.number}/paper.html`} key={block.number}>
            <p className="page-kicker">Block {block.number} · {Math.round(block.marketPrice * 100)}%</p>
            <h2>{block.title}</h2>
            <p>{block.summary}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
