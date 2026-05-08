import Link from 'next/link';

export function Footer() {
  return (
    <footer className="site-footer" style={{
      borderTop: '1px solid #222',
      marginTop: 48,
      padding: '20px 24px',
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      gap: 24,
      fontSize: '0.78rem',
      color: '#777',
    }}>
      <Link href="/llms.txt" style={{ color: '#64b5f6' }}>llms.txt</Link>
      <Link href="/api/agents/prompts" style={{ color: '#64b5f6' }}>API Reference</Link>
      <a href="https://github.com/asimog/cancerhawk" target="_blank" rel="noreferrer" style={{ color: '#64b5f6' }}>
        GitHub
      </a>
      <Link href="/jobs" style={{ color: '#64b5f6' }}>Jobs</Link>
      <Link href="/autonomous-logs" style={{ color: '#64b5f6' }}>Logs</Link>
      <span style={{ color: '#999' }}>
        CA: 0xbad0d2c1ad7c6293c879b4439b17d8665845dba3
      </span>
      <a href="https://dexscreener.com/base/0x783985ebb197c1a40bbf352b4299df21e270b72d6edaba6d458fd55ca65edfc4" target="_blank" rel="noreferrer" style={{ color: '#64b5f6' }}>
        $CancerHawk ↗
      </a>
    </footer>
  );
}
