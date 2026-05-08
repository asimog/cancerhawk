import * as fs from 'fs';
import * as path from 'path';
import type { BlockBundle, BlockMeta, Analysis } from './blocks.types';

const RESULTS_DIR = path.join(process.cwd(), 'results');
const DEFAULT_BACKEND_URL = 'https://cancerhawk-production.up.railway.app';

function cleanBackendUrl(value: string) {
  return value
    .trim()
    .replace(/^["']|["']$/g, '')
    .replace(/\\n/g, '')
    .replace(/\s+/g, '')
    .replace(/\/+$/, '');
}

function blockDirs() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  return fs.readdirSync(RESULTS_DIR)
    .map((name) => {
      const match = /^block-(\d+)$/.exec(name);
      return match ? { name, number: Number(match[1]) } : null;
    })
    .filter((entry): entry is { name: string; number: number } => Boolean(entry))
    .sort((a, b) => b.number - a.number);
}

function readJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, 'utf8')) as T;
}

export function getBlocks(): BlockBundle[] {
  return blockDirs().map(({ name, number }) => {
    const dir = path.join(RESULTS_DIR, name);
    return {
      number,
      meta: readJson<BlockMeta>(path.join(dir, 'block.json')),
      analysis: readJson<Analysis>(path.join(dir, 'analysis.json')),
      paper: fs.readFileSync(path.join(dir, 'paper.md'), 'utf8'),
    };
  });
}

export function getCurrentBlock(): BlockBundle | null {
  return getBlocks()[0] ?? null;
}

export function getBackendUrl(): string {
  return cleanBackendUrl(
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    process.env.CANCERHAWK_BACKEND_URL ||
    DEFAULT_BACKEND_URL,
  );
}

async function fetchBackendJson<T>(path: string): Promise<T | null> {
  const backendUrl = getBackendUrl();
  if (!backendUrl) return null;

  try {
    const response = await fetch(`${backendUrl}${path}`, { cache: 'no-store' });
    if (!response.ok) return null;
    const payload = await response.json();
    return (typeof payload === 'string' ? JSON.parse(payload) : payload) as T;
  } catch {
    return null;
  }
}

function blockNumberFromJob(job: Record<string, any>): number | null {
  const resultBlock = Number(job.result?.block);
  if (Number.isInteger(resultBlock) && resultBlock > 0) return resultBlock;

  const events = Array.isArray(job.events) ? job.events : [];
  for (const event of [...events].reverse()) {
    const message = typeof event?.message === 'string' ? event.message : '';
    const match = /^Block\s+(\d+)\s+published:/i.exec(message);
    if (match) return Number(match[1]);
  }

  return null;
}

function remoteBlockToBundle(payload: Record<string, any>): BlockBundle | null {
  const number = Number(payload.block ?? payload.meta?.block);
  if (!Number.isInteger(number) || number < 1) return null;

  const meta = {
    ...(payload.meta || {}),
    block: number,
    market_price: Number(
      payload.meta?.market_price ??
      payload.analysis?.market_price ??
      0,
    ),
  } as BlockMeta;

  const analysis = {
    ...(payload.analysis || {}),
    peer_reviews: payload.peer_reviews ?? payload.analysis?.peer_reviews ?? [],
    simulations: payload.simulations ?? payload.analysis?.simulations ?? [],
    derived_topics: payload.analysis?.topics ?? payload.analysis?.derived_topics ?? [],
    market_price: Number(payload.analysis?.market_price ?? meta.market_price ?? 0),
  } as Analysis;

  return {
    number,
    meta,
    analysis,
    paper: String(payload.paper_md || payload.paper || ''),
  };
}

async function getRemoteBlocks(): Promise<BlockBundle[]> {
  const jobsResponse = await fetchBackendJson<{ jobs?: Array<Record<string, any>> }>('/api/jobs?limit=100');
  const jobs = Array.isArray(jobsResponse?.jobs) ? jobsResponse.jobs : [];
  let blockNumbers = Array.from(new Set(
    jobs
      .filter((job) => job?.status === 'completed' || job?.status === 'published')
      .map(blockNumberFromJob)
      .filter((number): number is number => typeof number === 'number'),
  )).sort((a, b) => b - a);

  if (!blockNumbers.length) {
    blockNumbers = Array.from({ length: 50 }, (_, index) => index + 1);
  }

  const blocks = await Promise.all(
    blockNumbers.map(async (number) => {
      const payload = await fetchBackendJson<Record<string, any>>(`/api/blocks/${number}`);
      return payload ? remoteBlockToBundle(payload) : null;
    }),
  );

  return blocks.filter((block): block is BlockBundle => Boolean(block));
}

export async function getLiveBlocks(): Promise<BlockBundle[]> {
  const remoteBlocks = await getRemoteBlocks();
  return remoteBlocks.length ? remoteBlocks : getBlocks();
}

export async function getLiveCurrentBlock(): Promise<BlockBundle | null> {
  return (await getLiveBlocks())[0] ?? null;
}
