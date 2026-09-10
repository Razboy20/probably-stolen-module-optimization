import tgpu from 'typegpu';
import process from 'node:process';
import { boardTotals, type BoardTotals } from '../src/solver/boardTotals';
import { BOARD_CELLS } from '../src/solver/geometry';
import { createSearchKernel } from '../src/solver/gpu/kernel';
import { NO_RECORD } from '../src/solver/gpu/layout';
import { fitsGpu } from '../src/solver/gpu/upload';
import { addMachineTiers, compareTiers, objectiveTiers, TIER_VECTOR_LENGTH } from '../src/solver/objective';
import { boardOfSet, prepareSolve } from '../src/solver/setup';
import { FLAG_FIXED } from '../src/solver/tables';
import { buildBenchCase } from './benchCase';

/* Runs the GPU kernel as JavaScript under TypeGPU's simulate mode, one thread for ITERS iterations, and checks its record against the host scorer
 * It also resolves the kernel to WGSL, which is where typing mistakes surface without a GPU
 */
const SEED = Number(process.env.SEED ?? 1);
const N = Number(process.env.N ?? 200);
const ITERS = Number(process.env.ITERS ?? 300);
const TARGETS = Number(process.env.TARGETS ?? 0);
const THREAD = Number(process.env.THREAD ?? 0);
const MACHINES = Number(process.env.MACHINES ?? 1);

(globalThis as Record<string, unknown>).GPUBufferUsage = { COPY_DST: 8, COPY_SRC: 4, STORAGE: 128, UNIFORM: 64, MAP_READ: 1 };
const root = tgpu.initFromDevice({ device: { features: new Set(), limits: {}, queue: {} } as unknown as GPUDevice });

const { machines, inv } = buildBenchCase(SEED, N, TARGETS, MACHINES);
const setup = prepareSolve({ machines: machines.map(({ machine, board }) => ({ machine, initialBoard: board })), searchPoolInventory: inv, fullInventory: inv });
if (!fitsGpu(setup)) throw new Error('case does not fit the GPU tables');

const kernel = createSearchKernel(root, setup, SEED, THREAD + 1, ITERS);
const wgsl = tgpu.resolve([kernel.searchStep, kernel.extractChampion, kernel.migrate], { names: 'strict' });
if (process.env.WGSL_OUT) await Bun.write(process.env.WGSL_OUT, wgsl);

const t0 = performance.now();
const result = tgpu['~unstable'].simulate(() => kernel.runThread(THREAD));
const dt = performance.now() - t0;

interface State { hasRecord: number; bestTiers: number[]; best: number[]; cur: number[]; curP: number[]; curQ: number[]; curE: number[]; curPieces: number[] }
const states = result.buffers.get(kernel.state.buffer) as State[];
const scores = result.buffers.get(kernel.scores.buffer) as number[];
const st = states[THREAD];

// The host's totals of every board of a set and the combined tier vector the engine would judge it on
const hostScore = (cells: number[]) => {
    const set = Int32Array.from(cells);
    const combined = new Int32Array(TIER_VECTOR_LENGTH);
    const totals = setup.machines.map((machine, k) => {
        const boardTotal: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
        boardTotals(setup.tables, boardOfSet(set, k), boardTotal);
        const tiers = new Int32Array(TIER_VECTOR_LENGTH);
        objectiveTiers(boardTotal, machine.plan, machine.params, tiers);
        addMachineTiers(combined, tiers, machine.plan.tierCount, setup.densityIndex);
        return boardTotal;
    });
    return { totals, combined };
};

const best = hostScore(st.best);
const cur = hostScore(st.cur);
const kernelBest = Int32Array.from(st.bestTiers);
const reportedScore = scores.slice(THREAD * TIER_VECTOR_LENGTH, THREAD * TIER_VECTOR_LENGTH + TIER_VECTOR_LENGTH);

const itemsOfBoard = (cells: number[], k: number) => new Set(cells.slice(k * BOARD_CELLS, (k + 1) * BOARD_CELLS).filter(item => item >= 0));
const fixedKept = (cells: number[]) => setup.machines.every((_, k) => {
    const items = itemsOfBoard(cells, k);
    return Array.from(boardOfSet(setup.initialSet, k)).every(item => item < 0 || (setup.tables.flags[item] & FLAG_FIXED) === 0 || items.has(item));
});
const noDuplicates = (cells: number[]) => {
    const seen = new Set<number>();
    for (let k = 0; k < MACHINES; k++) for (const item of itemsOfBoard(cells, k)) {
        if (seen.has(item)) return false;
        seen.add(item);
    }
    return true;
};

const checks = {
    hasRecord: st.hasRecord === 1,
    bestTiersMatchHost: compareTiers(kernelBest, best.combined, setup.tierLength) === 0,
    curTotalsMatchHost: cur.totals.every((t, k) => st.curP[k] === t.p && st.curQ[k] === t.q && st.curE[k] === t.e && st.curPieces[k] === t.pieces),
    scoresMatchBest: st.hasRecord === 1 ? reportedScore.every((v, i) => v === kernelBest[i]) : reportedScore[0] === NO_RECORD,
    // Fixed pieces must still be on their board after every iteration, and no module may stand on two boards
    fixedKept: fixedKept(st.best) && fixedKept(st.cur),
    noDuplicates: noDuplicates(st.best) && noDuplicates(st.cur)
};
console.log(JSON.stringify({
    seed: SEED, targets: TARGETS, machines: MACHINES, iters: ITERS, ms: Math.round(dt), wgslChars: wgsl.length,
    bestTotals: best.totals, kernelBest: Array.from(kernelBest.subarray(0, setup.tierLength)), hostBest: Array.from(best.combined.subarray(0, setup.tierLength)),
    checks
}));
if (!Object.values(checks).every(Boolean)) process.exit(1);
