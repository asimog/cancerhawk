import type { BlockBundle } from './blocks.types';

export function getBackendUrl(): string {
  return (
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    process.env.CANCERHAWK_BACKEND_URL ||
    ''
  ).trim().replace(/\/+$/, '');
}

export function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number } = {},
): Promise<Response> {
  const { timeout = 8000, ...rest } = options;
  const controller = new AbortController();
  if (rest.signal) {
    if (rest.signal.aborted) {
      controller.abort();
    } else {
      rest.signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
  }
  const id = setTimeout(() => controller.abort(), timeout);
  return fetch(url, { ...rest, signal: controller.signal }).finally(() => clearTimeout(id));
}

const SOLANA_BASE58 = /^[A-HJ-NP-Za-km-z1-9]{32,44}$/;
const EVM_ADDRESS = /^0x[a-fA-F0-9]{40}$/;

export function validateWalletAddress(value: string): { valid: boolean; chain: 'solana' | 'base' | null; error?: string } {
  const trimmed = value.trim();
  if (!trimmed) return { valid: true, chain: null };
  if (SOLANA_BASE58.test(trimmed)) return { valid: true, chain: 'solana' };
  if (EVM_ADDRESS.test(trimmed)) return { valid: true, chain: 'base' };
  return { valid: false, chain: null, error: 'Enter a valid Solana or Base (0x…) address.' };
}

export function excerpt(markdown: string, maxLength = 280) {
  const text = markdown
    .split('\n')
    .filter((line) => !line.startsWith('#') && !line.startsWith('|') && line.trim())
    .join(' ')
    .replace(/\*\*/g, '')
    .trim();
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}...` : text;
}

// Re-export types for client-side usage
export type { BlockBundle };
