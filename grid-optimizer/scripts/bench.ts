import { runOptimizationEngine, type SolveUpdate } from '../src/solver/engine';
import { initializeBoard } from '../src/solver/board';
import type { MachineConfig } from '../src/solver/objective';
import { EFFECTS_LIST, MODULE_TEMPLATES } from '../src/constants';
import type { InventoryItem, ItemEffect, Stats } from '../src/types';
import process from 'node:process';

const SEED = Number(process.env.SEED ?? 1);
const N = Number(process.env.N ?? 200);
const MS = Number(process.env.MS ?? 3000);
const ITERS = Number(process.env.ITERS ?? 0);
const TARGETS = process.env.TARGETS === '1';

let seed = SEED * 7919 + 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
Math.random = rnd;

const inv: InventoryItem[] = [];
for (let i = 0; i < N; i++) {
    const t = MODULE_TEMPLATES[Math.floor(rnd() * 27)];
    const eff = (): ItemEffect => rnd() < 0.5 ? 'None' : EFFECTS_LIST[1 + Math.floor(rnd() * 9)];
    inv.push({ id: `m${i}`, shape: t.shape, color: t.color, displayName: t.displayName, effects: [eff(), eff()], effectValues: [20, 20] });
}
const node: InventoryItem = { id: 'node', shape: 'Node1x2', color: 'White', displayName: 'Node', effects: ['None', 'None'], effectValues: [0, 0], isInfinite: true };
inv.push(node);
for (let i = 0; i < 17; i++) inv.push({ ...node, id: `node_clone_${i}` });

const machine: MachineConfig = TARGETS ? {
    id: 'mach0', tier: 3,
    targetStats: { Performance: 200, Quality: 80, Efficiency: null },
    maximizeStats: { Performance: false, Quality: true, Efficiency: true },
    ignoreStats: { Performance: false, Quality: false, Efficiency: false },
    statPriority: { Performance: 1, Quality: 2, Efficiency: 3 },
} : {
    id: 'mach0', tier: 3,
    targetStats: { Performance: null, Quality: null, Efficiency: null },
    maximizeStats: { Performance: true, Quality: true, Efficiency: true },
};

const board = initializeBoard(machine.tier);
if (TARGETS) {
    const blast: InventoryItem = { id: 'blast', shape: 'Line4', color: 'Grey', displayName: 'Furnace Module (Blast)', effects: ['None', 'None'], effectValues: [0, 0] };
    const locked: InventoryItem = { id: 'lockedsq', shape: 'Square4_Base', color: 'Red', displayName: 'Performance', effects: ['Premium', 'None'], effectValues: [20, 20], isLocked: true };
    inv.push(blast, locked);
    for (let x = 0; x < 4; x++) board[4][x] = blast;
    board[0][5] = board[0][6] = board[1][5] = board[1][6] = locked;
}

const control = { running: true };
if (ITERS === 0) setTimeout(() => { control.running = false; }, MS);

let latest = null as SolveUpdate | null;
const checkpoints: Record<string, Stats | null> = {};
for (const ms of [250, 1000, 3000, 10000]) {
    if (ITERS === 0 && ms <= MS) setTimeout(() => { checkpoints[ms] = latest?.totals ?? null; }, ms);
}

const t0 = performance.now();
const { iterations } = await runOptimizationEngine(
    { machine, initialBoard: board, searchPoolInventory: inv, fullInventory: inv, seed: SEED, maxIterations: ITERS || undefined },
    control,
    (u) => { latest = u; }
);
const dt = performance.now() - t0;

const fnv = (s: string) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16); };
const cells = latest ? latest.board.flat().map(c => c && c !== 'Locked' ? c.id : (c ?? '.')).join(',') : '';
console.log(JSON.stringify({
    seed: SEED, N, targets: TARGETS, ms: Math.round(dt), iterations,
    itersPerSec: Math.round(iterations / (dt / 1000)),
    fingerprint: fnv(cells), totals: latest?.totals, checkpoints: ITERS ? undefined : checkpoints,
}));
