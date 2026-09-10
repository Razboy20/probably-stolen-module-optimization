import { boardTotals, type BoardTotals } from '../src/solver/boardTotals';
import {
    BOARD_CELLS, cellMaskHi, cellMaskLo, MAX_PIECE_CELLS, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_MASK_HI, PLACE_MASK_LO, PLACE_META,
    PLACE_VALID, placeEntry
} from '../src/solver/geometry';
import { runSolver } from '../src/solver/client';
import type { SolveUpdate } from '../src/solver/engine';
import { EMPTY, toIndexBoard } from '../src/solver/indexBoard';
import { objectiveTiers, TIER_VECTOR_LENGTH } from '../src/solver/objective';
import { type Rng, rngBelow, rngNext, seedRng } from '../src/solver/rng';
import { prepareSolve } from '../src/solver/setup';
import { FLAG_FIXED } from '../src/solver/tables';
import { buildBenchCase } from './benchCase';
import process from 'node:process';

/* A reference search that shares nothing with the solver but the objective: simulated annealing over random insert, remove and relocate moves,
 * with placements chosen uniformly rather than by the greedy scorer
 * It is far slower per improvement than the solver and is not meant to replace it; it says whether boards better than the solver's plateau exist at all
 */
const SEED = Number(process.env.SEED ?? 1);
const N = Number(process.env.N ?? 200);
const MS = Number(process.env.MS ?? 10000);
const TARGETS = Number(process.env.TARGETS ?? 0);
const T_START = Number(process.env.T0 ?? 40);
const T_END = Number(process.env.T1 ?? 0.5);
// Milliseconds of the solver to run first, so the annealing starts from the solver's plateau and shows whether anything better lies near it
const FROM_SOLVER_MS = Number(process.env.FROM_SOLVER_MS ?? 0);

const { machine, board, inv } = buildBenchCase(SEED, N, TARGETS);
const request = { machine, initialBoard: board, searchPoolInventory: inv, fullInventory: inv, seed: SEED };
const { tables, draw, plan, params, tierLength, initialIndexBoard } = prepareSolve(request);
const rng: Rng = seedRng(SEED, 99);

const startBoard = new Int32Array(initialIndexBoard);
if (FROM_SOLVER_MS > 0) {
    let latest = null as SolveUpdate | null;
    const solver = runSolver(request, u => { latest = u; }, 'inline');
    setTimeout(() => solver.stop(), FROM_SOLVER_MS);
    await solver.done;
    if (latest) startBoard.set(toIndexBoard(latest.board, tables.indexOf));
}

const current = new Int32Array(startBoard);
const best = new Int32Array(startBoard);
const totals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
const tiers = new Int32Array(TIER_VECTOR_LENGTH);

// One scalar from the tier vector, ranks a thousand apart so the order is the lexicographic one for the values these tiers take
const energy = (b: Int32Array) => {
    boardTotals(tables, b, totals);
    objectiveTiers(totals, plan, params, tiers);
    let e = 0;
    for (let i = 0; i < tierLength; i++) e = e * 1000 + tiers[i];
    return e;
};

const onBoard = new Uint8Array(tables.count);
const placed: number[] = [];
const refreshPlaced = (b: Int32Array) => {
    onBoard.fill(0);
    placed.length = 0;
    for (let i = 0; i < BOARD_CELLS; i++) {
        const item = b[i];
        if (item < 0 || onBoard[item] !== 0 || (tables.flags[item] & FLAG_FIXED) !== 0) continue;
        onBoard[item] = 1;
        placed.push(item);
    }
};

const removeItem = (b: Int32Array, item: number) => {
    for (let i = 0; i < BOARD_CELLS; i++) if (b[i] === item) b[i] = EMPTY;
};

// A uniformly random placement that fits, or false after a bounded number of tries
const insertRandom = (b: Int32Array, item: number) => {
    let lo = 0, hi = 0;
    for (let i = 0; i < BOARD_CELLS; i++) {
        if (b[i] !== EMPTY) { lo |= cellMaskLo(i); hi |= cellMaskHi(i); }
    }
    const orientStart = tables.orientStart[item];
    const orientCount = tables.orientCount[item];
    for (let tries = 0; tries < 64; tries++) {
        const entry = placeEntry(orientStart + rngBelow(rng, orientCount), rngBelow(rng, BOARD_CELLS));
        if ((PLACE_META[entry] & PLACE_VALID) === 0) continue;
        if (((PLACE_MASK_LO[entry] & lo) | (PLACE_MASK_HI[entry] & hi)) !== 0) continue;
        for (let c = 0; c < PLACE_CELL_COUNT[entry]; c++) b[PLACE_CELLS[entry * MAX_PIECE_CELLS + c]] = item;
        return true;
    }
    return false;
};

const proposal = new Int32Array(BOARD_CELLS);
const propose = () => {
    proposal.set(current);
    refreshPlaced(current);
    const move = rngBelow(rng, 3);
    if (move === 0 && placed.length > 0) {
        removeItem(proposal, placed[rngBelow(rng, placed.length)]);
        return true;
    }
    if (move === 1 && placed.length > 0) {
        const item = placed[rngBelow(rng, placed.length)];
        removeItem(proposal, item);
        return insertRandom(proposal, item);
    }
    for (let tries = 0; tries < 16; tries++) {
        const item = draw.drawList[rngBelow(rng, draw.drawList.length)];
        if (onBoard[item] !== 0) continue;
        return insertRandom(proposal, item);
    }
    return false;
};

let currentE = energy(current);
let bestE = currentE;
const checkpoints: Record<string, number[]> = {};
const t0 = performance.now();
let steps = 0;
let accepted = 0;
const marks = [250, 1000, 3000, 10000, 30000, 60000].filter(ms => ms <= MS);
let nextMark = 0;

for (;;) {
    const elapsed = performance.now() - t0;
    if (elapsed >= MS) break;
    if (nextMark < marks.length && elapsed >= marks[nextMark]) {
        energy(best);
        checkpoints[marks[nextMark]] = Array.from(tiers.subarray(0, tierLength));
        nextMark++;
    }
    steps++;
    if (!propose()) continue;
    const e = energy(proposal);
    const temperature = T_START * Math.pow(T_END / T_START, elapsed / MS);
    // Energies are in tier-0 units scaled by 1000 per lower tier, so the temperature is expressed in tier-0 points
    const delta = (e - currentE) / Math.pow(1000, tierLength - 1);
    if (delta >= 0 || (rngNext(rng) >>> 8) / 0x1000000 < Math.exp(delta / temperature)) {
        current.set(proposal);
        currentE = e;
        accepted++;
        if (e > bestE) { bestE = e; best.set(proposal); }
    }
}

energy(best);
console.log(JSON.stringify({
    seed: SEED, N, targets: TARGETS, ms: MS, steps, accepted, stepsPerSec: Math.round(steps / (MS / 1000)),
    best: { totals: { ...totals }, tiers: Array.from(tiers.subarray(0, tierLength)) }, checkpoints
}));
