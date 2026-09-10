import { runSolver, type SolverBackend } from '../src/solver/client';
import type { SolveUpdate } from '../src/solver/engine';
import type { Stats } from '../src/types';
import { buildBenchCase } from './benchCase';

// The same benchmark as bench.ts, driven by the URL so the GPU backend can be measured where WebGPU exists
const query = new URLSearchParams(location.search);
const SEED = Number(query.get('seed') ?? 1);
const N = Number(query.get('n') ?? 200);
const MS = Number(query.get('ms') ?? 3000);
const TARGETS = Number(query.get('targets') ?? 0);
const IMPL = (query.get('impl') ?? 'gpu') as SolverBackend;
const PARALLELISM = query.get('threads') ? Number(query.get('threads')) : undefined;

const out = document.getElementById('out') as HTMLPreElement;
const log = (line: string) => { out.textContent += line + '\n'; };

const run = async () => {
    const { machine, board, inv } = buildBenchCase(SEED, N, TARGETS);
    let latest = null as SolveUpdate | null;
    const checkpoints: Record<string, Stats | null> = {};
    for (const ms of [250, 1000, 3000, 10000]) {
        if (ms <= MS) setTimeout(() => { checkpoints[ms] = latest?.totals ?? null; }, ms);
    }

    const t0 = performance.now();
    const solver = runSolver(
        { machine, initialBoard: board, searchPoolInventory: inv, fullInventory: inv, seed: SEED },
        u => { latest = u; log(`${Math.round(performance.now() - t0)} ms  tiers=${JSON.stringify(u.tiers)}  totals=${JSON.stringify(u.totals)}`); },
        IMPL,
        PARALLELISM
    );
    setTimeout(() => solver.stop(), MS);
    const { iterations } = await solver.done;
    const dt = performance.now() - t0;
    log(JSON.stringify({
        seed: SEED, N, targets: TARGETS, impl: IMPL, threads: PARALLELISM, ms: Math.round(dt), iterations,
        itersPerSec: Math.round(iterations / (dt / 1000)), totals: latest?.totals, checkpoints
    }, null, 2));
};

run().catch(error => log(`failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}`));
