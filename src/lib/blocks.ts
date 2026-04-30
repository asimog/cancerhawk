import fs from 'node:fs';
import path from 'node:path';

export type BlockMeta = {
  block: number;
  title: string;
  research_goal: string;
  timestamp: string;
  market_price: number;
  section_count: number;
  has_peer_review: boolean;
  has_simulations: boolean;
};

export type Simulation = {
  id: string;
  title: string;
  description: string;
  rationale: string;
  expected_metrics: string[];
};

export type Analysis = {
  market_price: number;
  headline_catalysts?: string[];
  peer_reviews?: Array<Record<string, unknown>>;
  simulations?: Simulation[];
  derived_topics?: Array<Record<string, unknown>>;
  archetypes?: Array<Record<string, unknown>>;
};

export type BlockBundle = {
  number: number;
  meta: BlockMeta;
  analysis: Analysis;
  paper: string;
};

const RESULTS_DIR = path.join(process.cwd(), 'results');

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

export function getCurrentBlock() {
  return getBlocks()[0] ?? null;
}

export function getBackendUrl() {
  return (
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    process.env.CANCERHAWK_BACKEND_URL ||
    ''
  ).trim().replace(/\/+$/, '');
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
