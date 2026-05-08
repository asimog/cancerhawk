import type { BlockBundle } from './blocks.types';

const DEFAULT_BACKEND_URL = '';

function cleanBackendUrl(value: string) {
  return value.trim().replace(/^["']|["']$/g, '')
    .replace(/\\n/g, '')
    .replace(/\s+/g, '')
    .replace(/\/+$/, '');
}

export function getBackendUrl(): string {
  return cleanBackendUrl(
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    process.env.CANCERHAWK_BACKEND_URL ||
    DEFAULT_BACKEND_URL,
  );
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

const SOLANA_BASE58 = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
const BASE_EVM_HEX = /^0x[a-fA-F0-9]{40}$/;

export const GIVEWELL_WALLET = "4Z2DBVoQCJZ42cCTDMNvYDUqRjA1C3vV7B155Mc6jGah";
export const GIVEWELL_URL = "https://www.givewell.org/about/donate/cryptocurrency";

type WalletValidation = {
  valid: boolean;
  wallet: string;
  solana: string;
  chain: 'solana' | 'base' | 'default' | '';
  isDefault: boolean;
  error?: string;
};

export function validateWalletAddress(value: string): WalletValidation {
  const trimmed = value.trim();
  if (!trimmed) {
    return { valid: true, wallet: GIVEWELL_WALLET, solana: GIVEWELL_WALLET, chain: 'default', isDefault: true };
  }
  if (SOLANA_BASE58.test(trimmed)) {
    return { valid: true, wallet: trimmed, solana: trimmed, chain: 'solana', isDefault: trimmed === GIVEWELL_WALLET };
  }
  if (BASE_EVM_HEX.test(trimmed)) {
    return { valid: true, wallet: trimmed, solana: trimmed, chain: 'base', isDefault: false };
  }
  return {
    valid: false,
    wallet: "",
    solana: "",
    chain: '',
    isDefault: false,
    error: 'Enter a valid Solana base58 address or Base/EVM 0x address.',
  };
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
