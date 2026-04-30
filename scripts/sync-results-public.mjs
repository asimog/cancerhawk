import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = join(root, 'results');
const target = join(root, 'public', 'results');

mkdirSync(dirname(target), { recursive: true });
rmSync(target, { recursive: true, force: true });

if (existsSync(source)) {
  cpSync(source, target, { recursive: true });
  console.log(`Synced ${source} -> ${target}`);
} else {
  mkdirSync(target, { recursive: true });
  console.log(`Created empty ${target}; no results directory found.`);
}
