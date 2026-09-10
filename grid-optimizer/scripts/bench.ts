import type { SolveUpdate } from '../src/solver/engine';
import { runSolver, type SolverBackend } from '../src/solver/client';
import type { Stats } from '../src/types';
import { buildBenchCase } from './benchCase';
import process from 'node:process';

const SEED = Number(process.env.SEED ?? 1);
const N = Number(process.env.N ?? 200);
const MS = Number(process.env.MS ?? 3000);
const ITERS = Number(process.env.ITERS ?? 0);
const TARGETS = process.env.TARGETS === '1';
const IMPL = (process.env.IMPL ?? 'inline') as SolverBackend;
const WORKERS = process.env.WORKERS ? Number(process.env.WORKERS) : undefined;

const { machine, board, inv } = buildBenchCase(SEED, N, TARGETS);

let latest = null as SolveUpdate | null;
const checkpoints: Record<string, { totals: Stats; tiers: number[] } | null> = {};
for (const ms of [250, 1000, 3000, 10000]) {
    if (ITERS === 0 && ms <= MS) setTimeout(() => { checkpoints[ms] = latest ? { totals: latest.totals, tiers: latest.tiers } : null; }, ms);
}

const t0 = performance.now();
const solver = runSolver(
    { machine, initialBoard: board, searchPoolInventory: inv, fullInventory: inv, seed: SEED, maxIterations: ITERS || undefined },
    (u) => { latest = u; },
    IMPL,
    WORKERS
);
if (ITERS === 0) setTimeout(() => solver.stop(), MS);
const { iterations } = await solver.done;
const dt = performance.now() - t0;

const fnv = (s: string) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16); };
const cells = latest ? latest.board.flat().map(c => c && c !== 'Locked' ? c.id : (c ?? '.')).join(',') : '';
console.log(JSON.stringify({
    seed: SEED, N, targets: TARGETS, impl: IMPL, workers: WORKERS, ms: Math.round(dt), iterations,
    itersPerSec: Math.round(iterations / (dt / 1000)),
    fingerprint: fnv(cells), totals: latest?.totals, checkpoints: ITERS ? undefined : checkpoints,
}));
