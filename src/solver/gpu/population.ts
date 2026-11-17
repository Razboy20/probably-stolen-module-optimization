import tgpu, { type TgpuRoot } from 'typegpu';
import type { SolverHandle, UpdateHandler } from '../handle';
import { toIndexBoard } from '../indexBoard';
import { compareTiers, TIER_VECTOR_LENGTH } from '../objective';
import { buildUpdate, type SolveUpdate } from '../report';
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

/* The device or the kernel on it cannot work here, which a retry would only run into again
 * Anything else (a solve too large for the tables, a failing update handler, a lost device) says nothing about the next solve
 */
export class GpuUnusableError extends Error {}

const unusable = async <T>(what: string, work: () => T | Promise<T>) => {
    try {
        return await work();
    } catch (error) {
        throw new GpuUnusableError(`${what}: ${error instanceof Error ? error.message : String(error)}`, { cause: error });
    }
};

// A lost device is dropped, so the next solve starts on a fresh one instead of failing against the dead one
let rootPromise: Promise<TgpuRoot> | null = null;
const acquireRoot = () => {
    rootPromise ??= unusable('WebGPU unavailable', () => tgpu.init()).then(root => {
        void root.device.lost.then(() => { rootPromise = null; });
        return root;
    }, error => {
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

// A thread with no record, or failing that the one whose record every other beats
const worstThread = (scores: number[], threads: number, tierLength: number) => {
    let worst = 0;
    for (let t = 0; t < threads; t++) {
        const at = t * TIER_VECTOR_LENGTH;
        if (scores[at] === NO_RECORD) return t;
        if (tiersBeat(scores, worst * TIER_VECTOR_LENGTH, at, tierLength)) worst = t;
    }
    return worst;
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
    if (error) throw new GpuUnusableError(`WebGPU validation failed: ${error.message}`);
    return result;
};

/* Thousands of independent trajectories of the same solve, one per GPU thread
 * The host only ever reads the per-thread scores; a record board is copied out by a second tiny pass when one of them beats the best so far
 */
export const runGpuPopulation = (request: SolveRequest, onUpdate: UpdateHandler, threads = DEFAULT_THREADS): SolverHandle => {
    let running = true;
    let offered: SolveUpdate | null = null;
    const maxIterations = request.maxIterations ?? Infinity;

    const solve = async () => {
        const root = await acquireRoot();
        const setup = prepareSolve(request);
        if (!fitsGpu(setup)) throw new Error(`Solve does not fit the GPU tables: ${setup.machines.length} machines, ${setup.tables.count} modules`);
        const seed = request.seed ?? randomSeed();
        const { kernel, search, extract, migrate } = await unusable('GPU kernel failed to build', () => {
            const kernel: SearchKernel = createSearchKernel(root, setup, seed, threads);
            return {
                kernel,
                search: root.createComputePipeline({ compute: kernel.searchStep }),
                extract: root.createComputePipeline({ compute: kernel.extractChampion }),
                migrate: root.createComputePipeline({ compute: kernel.migrate })
            };
        });
        const params = { ...kernel.tables.params, threadCount: threads, itersPerDispatch: 1 };
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

        // An offered record that beats the leader's is written over the worst thread, as its record and the board it climbs from
        // That thread is the champion from then on, so a migration right away hands the record to the worst quarter as well
        const plantOffered = (scores: number[], leaderTiers: Int32Array) => {
            const record = offered;
            offered = null;
            if (record === null) return -1;
            const tiers = new Int32Array(TIER_VECTOR_LENGTH);
            tiers.set(record.tiers);
            if (compareTiers(tiers, leaderTiers, setup.tierLength) <= 0) return -1;
            bestTiers = tiers;

            const thread = worstThread(scores, threads, setup.tierLength);
            const set = record.boards.flatMap(({ board }) => Array.from(toIndexBoard(board, setup.tables.indexOf)));
            kernel.state.patch({
                [thread]: { cur: set, best: set, bestTiers: Array.from(tiers), hasRecord: 1, hasEpoch: 0, stagnation: 0, stagnations: 0, curPieces: { 0: -1 } }
            });
            return thread;
        };

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
            const leader = bestThread(scores, threads, setup.tierLength);
            if (leader === -1) continue;
            const leaderTiers = Int32Array.from(scores.slice(leader * TIER_VECTOR_LENGTH, (leader + 1) * TIER_VECTOR_LENGTH));
            const planted = plantOffered(scores, leaderTiers);
            params.championIdx = planted === -1 ? leader : planted;

            if (planted !== -1 || ++dispatches % MIGRATE_EVERY === 0) {
                const [b0, b1, b2, b3] = migrationThreshold(scores, threads, setup.tierLength);
                Object.assign(params, { migrateBelow0: b0, migrateBelow1: b1, migrateBelow2: b2, migrateBelow3: b3 });
                kernel.params.write(params);
                migrate.dispatchWorkgroups(workgroups);
            }
            if (planted !== -1) continue;

            if (bestTiers !== null && compareTiers(leaderTiers, bestTiers, setup.tierLength) <= 0) continue;
            bestTiers = leaderTiers;

            kernel.params.write(params);
            extract.dispatchWorkgroups(Math.ceil(kernel.setCells / WORKGROUP_SIZE));
            const board = Int32Array.from(await kernel.champion.read());
            onUpdate(buildUpdate(request, setup, board, leaderTiers));
        }
        return { iterations };
    };

    return { stop: () => { running = false; }, adopt: update => { offered = update; }, done: solve() };
};
