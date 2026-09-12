import type { Board } from '../src/solver/board';
import type { SolveRequest, SolveUpdate } from '../src/solver/engine';
import { runSolver, type SolverBackend } from '../src/solver/client';
import { addMachineTiers, TIER_VECTOR_LENGTH } from '../src/solver/objective';
import type { InventoryItem, Stats } from '../src/types';
import { type BenchMachine, buildBenchCase } from './benchCase';
import process from 'node:process';

/* MS is the budget per machine, so MACHINES machines solved jointly get MACHINES times as long as one solved alone
 * INDEPENDENT=1 solves them one after another instead, each treating the boards before it as locked, which is what Run All used to do
 * PRESOLVE=1 solves the first machine alone for MS before the joint solve, which then gets the rest of the budget,
 * so the joint solve starts from a board that was maximized for itself and has to move modules off it
 * Either way the record is the sum of the machines' tier vectors, so the modes can be compared
 */
const SEED = Number(process.env.SEED ?? 1);
const N = Number(process.env.N ?? 200);
const MS = Number(process.env.MS ?? 3000);
const ITERS = Number(process.env.ITERS ?? 0);
const TARGETS = Number(process.env.TARGETS ?? 0);
const IMPL = (process.env.IMPL ?? 'inline') as SolverBackend;
const WORKERS = process.env.WORKERS ? Number(process.env.WORKERS) : undefined;
const MACHINES = Number(process.env.MACHINES ?? 1);
const INDEPENDENT = process.env.INDEPENDENT === '1';
const PRESOLVE = process.env.PRESOLVE === '1';
const TOTAL_MS = MS * MACHINES;
const JOINT_MS = PRESOLVE ? TOTAL_MS - MS : TOTAL_MS;

const { machines, inv } = buildBenchCase(SEED, N, TARGETS, MACHINES);

const combinedTiers = (updates: SolveUpdate[]) => {
    const sum = new Int32Array(TIER_VECTOR_LENGTH);
    const densityIndex = Math.max(...updates.map(u => u.tiers.length - 1));
    for (const u of updates) addMachineTiers(sum, Int32Array.from(u.tiers), u.tiers.length - 1, densityIndex);
    return Array.from(sum.subarray(0, densityIndex + 1));
};

const lockUsed = (inventory: InventoryItem[], boards: Board[]) => {
    const used = new Set<string>();
    for (const board of boards) for (const row of board) for (const cell of row) if (cell && cell !== 'Locked') used.add(cell.id);
    return inventory.map(item => used.has(item.id) ? { ...item, isLocked: true } : item);
};

const solve = async (request: SolveRequest, ms: number, onUpdate: (u: SolveUpdate) => void) => {
    const solver = runSolver(request, onUpdate, IMPL, WORKERS);
    if (ITERS === 0) setTimeout(() => solver.stop(), ms);
    return (await solver.done).iterations;
};

const checkpoints: Record<string, { totals: Stats[]; tiers: number[] } | null> = {};
const finals: SolveUpdate[] = [];
let iterations = 0;
const t0 = performance.now();

const request = (set: BenchMachine[], searchPoolInventory: InventoryItem[]): SolveRequest => ({
    machines: set.map(({ machine, board }) => ({ machine, initialBoard: board })),
    searchPoolInventory, fullInventory: inv, seed: SEED, maxIterations: ITERS || undefined
});

if (INDEPENDENT) {
    for (const bench of machines) {
        let latest = null as SolveUpdate | null;
        const searchPoolInventory = lockUsed(inv, finals.flatMap(u => u.boards.map(b => b.board)));
        iterations += await solve(request([bench], searchPoolInventory), MS, u => { latest = u; });
        if (latest) finals.push(latest);
    }
} else {
    let latest = null as SolveUpdate | null;
    if (PRESOLVE) {
        iterations += await solve(request([machines[0]], inv), MS, u => { latest = u; });
        if (latest) machines[0].board = latest.boards[0].board;
    }
    for (const ms of [250, 1000, 3000, 10000]) {
        if (ITERS === 0 && ms < JOINT_MS) setTimeout(() => { checkpoints[ms] = latest ? { totals: latest.boards.map(b => b.totals), tiers: latest.tiers } : null; }, ms);
    }
    iterations += await solve(request(machines, inv), JOINT_MS, u => { latest = u; });
    if (latest) finals.push(latest);
}
const dt = performance.now() - t0;
const boards = finals.flatMap(u => u.boards);
if (ITERS === 0) checkpoints[JOINT_MS] = finals.length > 0 ? { totals: boards.map(b => b.totals), tiers: combinedTiers(finals) } : null;

const fnv = (s: string) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16); };
const cells = boards.map(b => b.board.flat().map(c => c && c !== 'Locked' ? c.id : (c ?? '.')).join(',')).join(';');
const boardIds = boards.map(b => new Set(b.board.flat().flatMap(c => c && c !== 'Locked' ? [c.id] : [])));
const duplicates = boardIds.flatMap((ids, k) => [...ids].filter(id => boardIds.some((other, j) => j > k && other.has(id))));
console.log(JSON.stringify({
    seed: SEED, N, targets: TARGETS, machines: MACHINES, independent: INDEPENDENT, impl: IMPL, workers: WORKERS, ms: Math.round(dt), iterations,
    itersPerSec: Math.round(iterations / (dt / 1000)),
    fingerprint: fnv(cells), totals: boards.map(b => b.totals), checkpoints: ITERS ? undefined : checkpoints,
    duplicates: duplicates.length > 0 ? duplicates : undefined,
}));
