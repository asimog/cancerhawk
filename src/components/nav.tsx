import Link from 'next/link';

export function Nav() {
  return (
    <nav aria-label="CancerHawk routes" className="site-nav">
      <Link href="/">Home</Link>
      <Link href="/current-block">Current Block</Link>
      <Link href="/previous-blocks">Previous Blocks</Link>
      <Link href="/run-research">Run Research</Link>
      <Link href="/music">Music</Link>
    </nav>
  );
}
