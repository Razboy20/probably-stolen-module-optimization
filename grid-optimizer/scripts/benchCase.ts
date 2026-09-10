import { type Board, initializeBoard } from '../src/solver/board';
import type { MachineConfig } from '../src/solver/objective';
import { EFFECTS_LIST, MODULE_TEMPLATES } from '../src/constants';
import type { InventoryItem, ItemEffect } from '../src/types';

/* A seeded synthetic inventory and machines, the same for every backend and runtime the benchmarks compare
 * Case 0 maximizes everything; case 1 holds Performance and Quality to targets the pool meets easily and maximizes the rest;
 * case 2 asks for a Performance near the most the pool can reach (661–753 across seeds 1–4), so the shortfall shapes the whole search
 * Every machine gets the same configuration and, outside case 0, its own locked square and Blast module already on the board
 */
export const BENCH_CASES = 3;

export interface BenchMachine {
    machine: MachineConfig;
    board: Board;
}

const machineConfig = (id: string, benchCase: number): MachineConfig => benchCase === 0 ? {
    id, tier: 3,
    targetStats: { Performance: null, Quality: null, Efficiency: null },
    maximizeStats: { Performance: true, Quality: true, Efficiency: true },
} : benchCase === 1 ? {
    id, tier: 3,
    targetStats: { Performance: 200, Quality: 80, Efficiency: null },
    maximizeStats: { Performance: false, Quality: true, Efficiency: true },
    ignoreStats: { Performance: false, Quality: false, Efficiency: false },
    statPriority: { Performance: 1, Quality: 2, Efficiency: 3 },
} : {
    id, tier: 3,
    targetStats: { Performance: 600, Quality: 60, Efficiency: null },
    maximizeStats: { Performance: false, Quality: false, Efficiency: true },
    ignoreStats: { Performance: false, Quality: false, Efficiency: false },
    statPriority: { Performance: 1, Quality: 2, Efficiency: 3 },
};

export const buildBenchCase = (seedValue: number, n: number, benchCase: number, machineCount = 1) => {
    let seed = seedValue * 7919 + 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    Math.random = rnd;

    const inv: InventoryItem[] = [];
    for (let i = 0; i < n; i++) {
        const t = MODULE_TEMPLATES[Math.floor(rnd() * 27)];
        const eff = (): ItemEffect => rnd() < 0.5 ? 'None' : EFFECTS_LIST[1 + Math.floor(rnd() * 9)];
        inv.push({ id: `m${i}`, shape: t.shape, color: t.color, displayName: t.displayName, effects: [eff(), eff()], effectValues: [20, 20] });
    }
    const node: InventoryItem = { id: 'node', shape: 'Node1x2', color: 'White', displayName: 'Node', effects: ['None', 'None'], effectValues: [0, 0], isInfinite: true };
    inv.push(node);
    for (let i = 0; i < 17 * machineCount; i++) inv.push({ ...node, id: `node_clone_${i}` });

    const machines: BenchMachine[] = [];
    for (let k = 0; k < machineCount; k++) {
        const machine = machineConfig(`mach${k}`, benchCase);
        const board = initializeBoard(machine.tier);
        if (benchCase !== 0) {
            const suffix = k === 0 ? '' : String(k);
            const blast: InventoryItem = { id: `blast${suffix}`, shape: 'Line4', color: 'Grey', displayName: 'Furnace Module (Blast)', effects: ['None', 'None'], effectValues: [0, 0] };
            const locked: InventoryItem = { id: `lockedsq${suffix}`, shape: 'Square4_Base', color: 'Red', displayName: 'Performance', effects: ['Premium', 'None'], effectValues: [20, 20], isLocked: true };
            inv.push(blast, locked);
            for (let x = 0; x < 4; x++) board[4][x] = blast;
            board[0][5] = board[0][6] = board[1][5] = board[1][6] = locked;
        }
        machines.push({ machine, board });
    }
    return { machines, inv };
};
