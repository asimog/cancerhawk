import Link from 'next/link';
import type { GetStaticProps } from 'next';
import { useMemo, useState } from 'react';
import { Nav } from '@/components/nav';
import type { BlockBundle } from '@/lib/blocks.types';

type ArchiveBlock = {
  number: number;
  title: string;
  marketPrice: number;
  summary: string;
};

const BLOCKS_PER_PAGE = 9;

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
  const { getBlocks } = await import('@/lib/blocks.server');
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
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(blocks.length / BLOCKS_PER_PAGE));
  const safePage = Math.min(page, totalPages);
  const visibleBlocks = useMemo(
    () => blocks.slice((safePage - 1) * BLOCKS_PER_PAGE, safePage * BLOCKS_PER_PAGE),
    [blocks, safePage],
  );

  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Archive</p>
      <h1 className="page-title">Previous Blocks</h1>
      <div className="archive-grid">
        {visibleBlocks.map((block) => (
          <Link className="panel" href={block.number === blocks[0]?.number ? '/current-block' : `/results/block-${block.number}/paper.html`} key={block.number}>
            <p className="page-kicker">Block {block.number} · {Math.round(block.marketPrice * 100)}%</p>
            <h2>{block.title}</h2>
            <p>{block.summary}</p>
          </Link>
        ))}
      </div>
      {totalPages > 1 && (
        <nav aria-label="Previous blocks pages" className="archive-pagination">
          <button className="button" disabled={safePage === 1} onClick={() => setPage((current) => Math.max(1, current - 1))} type="button">
            Previous
          </button>
          <span className="archive-page-count">Page {safePage} of {totalPages}</span>
          <button className="button" disabled={safePage === totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))} type="button">
            Next
          </button>
        </nav>
      )}
    </div>
  );
}
