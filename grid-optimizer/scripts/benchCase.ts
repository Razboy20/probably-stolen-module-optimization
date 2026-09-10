import { initializeBoard } from '../src/solver/board';
import type { MachineConfig } from '../src/solver/objective';
import { EFFECTS_LIST, MODULE_TEMPLATES } from '../src/constants';
import type { InventoryItem, ItemEffect } from '../src/types';

// A seeded synthetic inventory and machine, the same for every backend and runtime the benchmarks compare
export const buildBenchCase = (seedValue: number, n: number, targets: boolean) => {
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
    for (let i = 0; i < 17; i++) inv.push({ ...node, id: `node_clone_${i}` });

    const machine: MachineConfig = targets ? {
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
    if (targets) {
        const blast: InventoryItem = { id: 'blast', shape: 'Line4', color: 'Grey', displayName: 'Furnace Module (Blast)', effects: ['None', 'None'], effectValues: [0, 0] };
        const locked: InventoryItem = { id: 'lockedsq', shape: 'Square4_Base', color: 'Red', displayName: 'Performance', effects: ['Premium', 'None'], effectValues: [20, 20], isLocked: true };
        inv.push(blast, locked);
        for (let x = 0; x < 4; x++) board[4][x] = blast;
        board[0][5] = board[0][6] = board[1][5] = board[1][6] = locked;
    }
    return { machine, board, inv };
};
