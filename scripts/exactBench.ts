import process from 'node:process';
import { runSolver } from '../src/solver/client';
import type { SolveUpdate } from '../src/solver/engine';
import type { MachineConfig } from '../src/solver/objective';
import { buildBenchCase } from './benchCase';

/* MACHINES machines, each asking for exactly Quality TARGET with the other stats ignored, solved jointly from a shared pool of N modules
 * Reports each machine's Quality, how many hit the target exactly, how far short and how far over the set finished, and the pieces used
 * RSEED seeds the solver apart from the case, so the same case can be run again to size the run-to-run noise
 */
const MS = Number(process.env.MS ?? 3000);
const SEED = Number(process.env.SEED ?? 1);
const RSEED = Number(process.env.RSEED ?? SEED);
const K = Number(process.env.MACHINES ?? 3);
const T = Number(process.env.TARGET ?? 150);
const N = Number(process.env.N ?? 120);

const { machines, inv } = buildBenchCase(SEED, N, 0, K);
const configured: MachineConfig[] = machines.map(m => ({
    ...m.machine,
    targetStats: { Performance: null, Quality: T, Efficiency: null },
    maximizeStats: { Performance: false, Quality: false, Efficiency: false },
    ignoreStats: { Performance: true, Quality: false, Efficiency: true }
}));

let latest: SolveUpdate | null = null;
const solver = runSolver(
    { machines: configured.map((machine, k) => ({ machine, initialBoard: machines[k].board })), searchPoolInventory: inv, fullInventory: inv, seed: RSEED },
    u => { latest = u; },
    process.env.IMPL ?? 'inline'
);
setTimeout(() => solver.stop(), MS);
const { iterations } = await solver.done;

const boards = latest!.boards;
const q = boards.map(b => b.totals.Quality);
const pieces = boards.map(b => new Set(b.board.flat().filter(c => c && c !== 'Locked').map(c => (c as { id: string }).id)).size);
const short = q.reduce((sum, v) => sum + Math.max(0, T - v), 0);
const over = q.reduce((sum, v) => sum + Math.max(0, v - T), 0);
console.log(JSON.stringify({ seed: SEED, target: T, q, exact: q.filter(v => v === T).length, short, over, pieces, iterations, tiers: latest!.tiers }));
