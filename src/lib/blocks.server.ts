import * as fs from 'fs';
import * as path from 'path';
import type { BlockBundle, BlockMeta, Analysis } from './blocks.types';

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

export function getCurrentBlock(): BlockBundle | null {
  return getBlocks()[0] ?? null;
}

export function getBackendUrl(): string {
  return (
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    process.env.CANCERHAWK_BACKEND_URL ||
    ''
  ).trim().replace(/\/+$/, '');
}
