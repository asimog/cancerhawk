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
      <Link href="/api/agents/prompts" style={{ color: '#64b5f6' }}>Agent API</Link>
      <a href="https://github.com/asimog/cancerhawk" target="_blank" rel="noreferrer" style={{ color: '#64b5f6' }}>
        GitHub
      </a>
      <Link href="/jobs" style={{ color: '#64b5f6' }}>Job Feed</Link>
      <Link href="/autonomous-logs" style={{ color: '#64b5f6' }}>Autonomous Logs</Link>
    </footer>
  );
}
