import type { InventoryItem, Stats } from '../types';
import type { Board } from './board';
import { indexInventoryById } from './boardStats';
import { buildDrawRanks, type DrawLayout, layoutDraw, targetedStats } from './draw';
import { BOARD_CELLS } from './geometry';
import { toIndexBoard } from './indexBoard';
import { activeRankOrder, buildTierPlan, type MachineConfig, STAT_KEYS, statIsIgnored, tierBoost, type TierPlan } from './objective';
import { buildSearchPool } from './pool';
import { buildScoringParams, MAX_TARGET, type ScoringParams } from './scoring';
import { buildPoolTables, type PoolTables } from './tables';

export interface MachineRequest {
    machine: MachineConfig;
    initialBoard: Board;
}

/* One solve over a set of machines that share the inventory: the search moves modules within and between their boards
 * and is judged on the sum of their objectives. A single machine is the ordinary case
 * Modules already on one of these boards must not be marked locked in the search pool, or no board could ever draw them
 */
export interface SolveRequest {
    machines: MachineRequest[];
    searchPoolInventory: InventoryItem[];
    fullInventory: InventoryItem[];
    seed?: number;
    // Which stream of the seed this solve follows, so a population of solves can share one seed and still diverge
    thread?: number;
    maxIterations?: number;
}

// What is fixed before the first iteration for one machine of the set
export interface MachineSetup {
    machine: MachineConfig;
    draw: DrawLayout;
    plan: TierPlan;
    params: ScoringParams;
    targeted: number[];
    drawRanks: Int32Array[];
    needsTotals: boolean;
    openCellCount: number;
}

// Everything about one solve that is fixed before the first iteration, shared by every backend
export interface SolveSetup {
    tables: PoolTables;
    inventoryById: Map<string, InventoryItem>;
    machines: MachineSetup[];
    // The combined tier vector: every machine's tiers left-aligned, and every density term at densityIndex
    densityIndex: number;
    tierLength: number;
    // The boards of the set laid end to end, BOARD_CELLS per machine
    initialSet: Int32Array;
}

export const boardOfSet = (set: Int32Array, machine: number) => set.subarray(machine * BOARD_CELLS, (machine + 1) * BOARD_CELLS);

// A shortfall costs 10000 per point per stat, and the sum over the set has to stay within 32 bits
const targetCap = (machineCount: number) => Math.min(MAX_TARGET, Math.floor(2 ** 31 / (10000 * 3 * machineCount)));

const prepareMachine = (tables: PoolTables, searchPoolInventory: InventoryItem[], { machine, initialBoard }: MachineRequest, cap: number, rankOrder: number[]): MachineSetup => {
    // Boards may already hold modules the search itself would not pick up, so the pruned pool is only used for choosing what to place
    const searchPool = buildSearchPool(searchPoolInventory, tables.internal, machine).filter(item => tables.indexOf.has(item.id));
    const draw = layoutDraw(tables, Int32Array.from(searchPool, item => tables.indexOf.get(item.id)!));

    const plan = buildTierPlan(machine, rankOrder);

    // A stat marked ignored gets weight 0 so the placement heuristic stops steering away from it at all
    const placementWeights: Stats = { Performance: 0, Quality: 0, Efficiency: 0 };
    for (let s = 0; s < 3; s++) {
        const key = STAT_KEYS[s];
        if (statIsIgnored(machine, key)) continue;
        let w = 0;
        if (machine.maximizeStats[key]) w += 10;
        if (machine.targetStats[key] !== null) w += 15;
        placementWeights[key] = w * tierBoost(plan, s);
    }
    const params = buildScoringParams(machine, placementWeights, cap);
    const targeted = targetedStats(machine);
    const drawRanks = buildDrawRanks(tables, draw.drawList, machine, plan);

    // The placement heuristic only reads the running totals to judge distance to a target,
    // so without one the recalculation after every placement is pure waste
    const needsTotals = targeted.some(s => !statIsIgnored(machine, STAT_KEYS[s]));

    // A board is empty exactly when every cell it has is free, and which cells it has is fixed by its tier
    let openCellCount = 0;
    for (const row of initialBoard) for (const cell of row) if (cell !== 'Locked') openCellCount++;

    return { machine, draw, plan, params, targeted, drawRanks, needsTotals, openCellCount };
};

export const prepareSolve = (request: SolveRequest): SolveSetup => {
    const { machines, searchPoolInventory, fullInventory } = request;
    const tables = buildPoolTables(fullInventory, machines.map(m => m.initialBoard));
    const inventoryById = indexInventoryById(fullInventory);

    const cap = targetCap(machines.length);
    // Priorities are compared across the whole set, so a rank 2 stat on one machine outranks a rank 3 stat on another
    const rankOrder = activeRankOrder(machines.map(m => m.machine));
    const machineSetups = machines.map(m => prepareMachine(tables, searchPoolInventory, m, cap, rankOrder));
    const densityIndex = rankOrder.length;

    const initialSet = new Int32Array(BOARD_CELLS * machines.length);
    machines.forEach((m, k) => boardOfSet(initialSet, k).set(toIndexBoard(m.initialBoard, tables.indexOf)));

    return { tables, inventoryById, machines: machineSetups, densityIndex, tierLength: densityIndex + 1, initialSet };
};
