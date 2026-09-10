import tgpu, { type TgpuRoot } from 'typegpu';
import type { SolverHandle, UpdateHandler } from '../handle';
import { compareTiers, TIER_VECTOR_LENGTH } from '../objective';
import { buildUpdate } from '../report';
import { randomSeed } from '../rng';
import { prepareSolve, type SolveRequest } from '../setup';
import { createSearchKernel, type SearchKernel } from './kernel';
import { DEFAULT_THREADS, NO_RECORD, WORKGROUP_SIZE } from './layout';
import { fitsGpu } from './upload';

// Dispatches are sized to take about this long, so the page stays responsive while the GPU is kept busy
const TARGET_DISPATCH_MS = 30;
const MAX_ITERS_PER_DISPATCH = 512;
const SLOW_DISPATCH_MS = 250;
// Every so many dispatches the threads in the worst quarter by record adopt the champion
const MIGRATE_EVERY = 16;
const MIGRATE_FRACTION = 4;

let rootPromise: Promise<TgpuRoot> | null = null;
const acquireRoot = () => {
    rootPromise ??= tgpu.init().catch(error => {
        rootPromise = null;
        throw error;
    });
    return rootPromise;
};

const tuneIters = (iters: number, dt: number) => {
    if (dt > SLOW_DISPATCH_MS) return Math.max(1, iters >> 2);
    const factor = Math.min(2, Math.max(0.5, TARGET_DISPATCH_MS / Math.max(dt, 0.1)));
    return Math.min(MAX_ITERS_PER_DISPATCH, Math.max(1, Math.round(iters * factor)));
};

// The thread whose record beats every other, or -1 when none has one
const bestThread = (scores: number[], threads: number, tierLength: number) => {
    let best = -1;
    for (let t = 0; t < threads; t++) {
        const at = t * TIER_VECTOR_LENGTH;
        if (scores[at] === NO_RECORD) continue;
        if (best === -1 || tiersBeat(scores, at, best * TIER_VECTOR_LENGTH, tierLength)) best = t;
    }
    return best;
};

const tiersBeat = (scores: number[], a: number, b: number, tierLength: number) => {
    for (let i = 0; i < tierLength; i++) {
        if (scores[a + i] !== scores[b + i]) return scores[a + i] > scores[b + i];
    }
    return false;
};

// The record tiers a thread must be below to be in the worst MIGRATE_FRACTION-th of the threads that have a record
const migrationThreshold = (scores: number[], threads: number, tierLength: number) => {
    const ranked: number[] = [];
    for (let t = 0; t < threads; t++) if (scores[t * TIER_VECTOR_LENGTH] !== NO_RECORD) ranked.push(t);
    ranked.sort((a, b) => tiersBeat(scores, a * TIER_VECTOR_LENGTH, b * TIER_VECTOR_LENGTH, tierLength) ? 1 : tiersBeat(scores, b * TIER_VECTOR_LENGTH, a * TIER_VECTOR_LENGTH, tierLength) ? -1 : 0);
    const at = ranked[Math.trunc(ranked.length / MIGRATE_FRACTION)] * TIER_VECTOR_LENGTH;
    return scores.slice(at, at + TIER_VECTOR_LENGTH);
};

const validated = async <T>(device: GPUDevice, work: () => Promise<T>) => {
    device.pushErrorScope('validation');
    const result = await work();
    const error = await device.popErrorScope();
    if (error) throw new Error(`WebGPU validation failed: ${error.message}`);
    return result;
};

/* Thousands of independent trajectories of the same solve, one per GPU thread
 * The host only ever reads the per-thread scores; a record board is copied out by a second tiny pass when one of them beats the best so far
 */
export const runGpuPopulation = (request: SolveRequest, onUpdate: UpdateHandler, threads = DEFAULT_THREADS): SolverHandle => {
    let running = true;
    const maxIterations = request.maxIterations ?? Infinity;

    const solve = async () => {
        const root = await acquireRoot();
        const setup = prepareSolve(request);
        if (!fitsGpu(setup)) throw new Error('Solve does not fit the GPU tables');
        const seed = request.seed ?? randomSeed();
        const kernel: SearchKernel = createSearchKernel(root, setup, seed, threads);
        const params = { ...kernel.tables.params, threadCount: threads, itersPerDispatch: 1 };

        const search = root.createComputePipeline({ compute: kernel.searchStep });
        const extract = root.createComputePipeline({ compute: kernel.extractChampion });
        const migrate = root.createComputePipeline({ compute: kernel.migrate });
        const workgroups = Math.ceil(threads / WORKGROUP_SIZE);

        let lost = false;
        root.device.lost.then(() => { lost = true; });
        const step = async () => {
            kernel.params.write(params);
            search.dispatchWorkgroups(workgroups);
            await root.device.queue.onSubmittedWorkDone();
            if (lost) throw new Error('GPU device lost');
        };

        let bestTiers: Int32Array | null = null;
        let iterations = 0;
        let dispatches = 0;
        let first = true;
        while (running && iterations < maxIterations) {
            const t0 = performance.now();
            if (first) {
                await validated(root.device, step);
                first = false;
            } else {
                await step();
            }
            const dt = performance.now() - t0;
            iterations += threads * params.itersPerDispatch;
            params.itersPerDispatch = tuneIters(params.itersPerDispatch, dt);

            const scores = await kernel.scores.read();
            const winner = bestThread(scores, threads, setup.tierLength);
            if (winner === -1) continue;
            params.championIdx = winner;

            if (++dispatches % MIGRATE_EVERY === 0) {
                const [b0, b1, b2, b3] = migrationThreshold(scores, threads, setup.tierLength);
                Object.assign(params, { migrateBelow0: b0, migrateBelow1: b1, migrateBelow2: b2, migrateBelow3: b3 });
                kernel.params.write(params);
                migrate.dispatchWorkgroups(workgroups);
            }

            const tiers = Int32Array.from(scores.slice(winner * TIER_VECTOR_LENGTH, (winner + 1) * TIER_VECTOR_LENGTH));
            if (bestTiers !== null && compareTiers(tiers, bestTiers, setup.tierLength) <= 0) continue;
            bestTiers = tiers;

            kernel.params.write(params);
            extract.dispatchWorkgroups(1);
            const board = Int32Array.from(await kernel.champion.read());
            onUpdate(buildUpdate(request, setup, board, tiers));
        }
        return { iterations };
    };

    return { stop: () => { running = false; }, done: solve() };
};
