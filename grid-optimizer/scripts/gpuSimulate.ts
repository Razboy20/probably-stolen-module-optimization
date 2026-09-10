import tgpu from 'typegpu';
import process from 'node:process';
import { boardTotals, type BoardTotals } from '../src/solver/boardTotals';
import { createSearchKernel } from '../src/solver/gpu/kernel';
import { NO_RECORD } from '../src/solver/gpu/layout';
import { fitsGpu } from '../src/solver/gpu/upload';
import { compareTiers, objectiveTiers, TIER_VECTOR_LENGTH } from '../src/solver/objective';
import { prepareSolve } from '../src/solver/setup';
import { buildBenchCase } from './benchCase';

/* Runs the GPU kernel as JavaScript under TypeGPU's simulate mode, one thread for ITERS iterations, and checks its record against the host scorer
 * It also resolves the kernel to WGSL, which is where typing mistakes surface without a GPU
 */
const SEED = Number(process.env.SEED ?? 1);
const N = Number(process.env.N ?? 200);
const ITERS = Number(process.env.ITERS ?? 300);
const TARGETS = Number(process.env.TARGETS ?? 0);
const THREAD = Number(process.env.THREAD ?? 0);

(globalThis as Record<string, unknown>).GPUBufferUsage = { COPY_DST: 8, COPY_SRC: 4, STORAGE: 128, UNIFORM: 64, MAP_READ: 1 };
const root = tgpu.initFromDevice({ device: { features: new Set(), limits: {}, queue: {} } as unknown as GPUDevice });

const { machine, board, inv } = buildBenchCase(SEED, N, TARGETS);
const setup = prepareSolve({ machine, initialBoard: board, searchPoolInventory: inv, fullInventory: inv });
if (!fitsGpu(setup)) throw new Error('case does not fit the GPU tables');

const kernel = createSearchKernel(root, setup, SEED, THREAD + 1, ITERS);
const wgsl = tgpu.resolve([kernel.searchStep, kernel.extractChampion, kernel.migrate], { names: 'strict' });
if (process.env.WGSL_OUT) await Bun.write(process.env.WGSL_OUT, wgsl);

const t0 = performance.now();
const result = tgpu['~unstable'].simulate(() => kernel.runThread(THREAD));
const dt = performance.now() - t0;

const states = result.buffers.get(kernel.state.buffer) as { hasRecord: number; bestTiers: number[]; best: number[]; cur: number[]; curP: number; curQ: number; curE: number; curPieces: number }[];
const scores = result.buffers.get(kernel.scores.buffer) as number[];
const st = states[THREAD];

const hostTiers = (cells: number[]) => {
    const totals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    boardTotals(setup.tables, Int32Array.from(cells), totals);
    const tiers = new Int32Array(TIER_VECTOR_LENGTH);
    objectiveTiers(totals, setup.plan, setup.params, tiers);
    return { totals, tiers };
};

const best = hostTiers(st.best);
const cur = hostTiers(st.cur);
const kernelBest = Int32Array.from(st.bestTiers);
const reportedScore = scores.slice(THREAD * TIER_VECTOR_LENGTH, THREAD * TIER_VECTOR_LENGTH + TIER_VECTOR_LENGTH);

const checks = {
    hasRecord: st.hasRecord === 1,
    bestTiersMatchHost: compareTiers(kernelBest, best.tiers, setup.tierLength) === 0,
    curTotalsMatchHost: st.curP === cur.totals.p && st.curQ === cur.totals.q && st.curE === cur.totals.e && st.curPieces === cur.totals.pieces,
    scoresMatchBest: st.hasRecord === 1 ? reportedScore.every((v, i) => v === kernelBest[i]) : reportedScore[0] === NO_RECORD,
    // Fixed pieces must still be on the board after every iteration
    fixedKept: setup.initialIndexBoard.every(item => item < 0 || (setup.tables.flags[item] & 16) === 0 || st.best.includes(item)) && st.best[0] !== undefined
};
console.log(JSON.stringify({
    seed: SEED, targets: TARGETS, iters: ITERS, ms: Math.round(dt), wgslChars: wgsl.length,
    bestTotals: best.totals, kernelBest: Array.from(kernelBest.subarray(0, setup.tierLength)), hostBest: Array.from(best.tiers.subarray(0, setup.tierLength)),
    checks
}));
if (!Object.values(checks).every(Boolean)) process.exit(1);
