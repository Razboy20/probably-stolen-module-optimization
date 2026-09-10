import type { InventoryItem, Stats } from '../types';
import type { Board } from './board';
import { indexInventoryById } from './boardStats';
import { buildDrawRanks, type DrawLayout, layoutDraw, targetedStats } from './draw';
import { BOARD_CELLS } from './geometry';
import { type IndexBoard, LOCKED, toIndexBoard } from './indexBoard';
import { buildTierPlan, type MachineConfig, STAT_KEYS, statIsIgnored, tierBoost, type TierPlan } from './objective';
import { buildSearchPool } from './pool';
import { buildScoringParams, type ScoringParams } from './scoring';
import { buildPoolTables, type PoolTables } from './tables';

export interface SolveRequest {
    machine: MachineConfig;
    initialBoard: Board;
    searchPoolInventory: InventoryItem[];
    fullInventory: InventoryItem[];
    seed?: number;
    // Which stream of the seed this solve follows, so a population of solves can share one seed and still diverge
    thread?: number;
    maxIterations?: number;
}

// Everything about one solve that is fixed before the first iteration, shared by every backend
export interface SolveSetup {
    tables: PoolTables;
    inventoryById: Map<string, InventoryItem>;
    draw: DrawLayout;
    plan: TierPlan;
    tierLength: number;
    params: ScoringParams;
    targeted: number[];
    drawRanks: Int32Array[];
    needsTotals: boolean;
    initialIndexBoard: IndexBoard;
    openCellCount: number;
}

export const prepareSolve = (request: SolveRequest): SolveSetup => {
    const { machine, initialBoard, searchPoolInventory, fullInventory } = request;
    const tables = buildPoolTables(fullInventory, initialBoard);
    const inventoryById = indexInventoryById(fullInventory);

    // Boards may already hold modules the search itself would not pick up, so the pruned pool is only used for choosing what to place
    const searchPool = buildSearchPool(searchPoolInventory, tables.internal, machine).filter(item => tables.indexOf.has(item.id));
    const draw = layoutDraw(tables, Int32Array.from(searchPool, item => tables.indexOf.get(item.id)!));
    const { drawList } = draw;

    const plan = buildTierPlan(machine);

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
    const params = buildScoringParams(machine, placementWeights);
    const targeted = targetedStats(machine);
    const drawRanks = buildDrawRanks(tables, drawList, machine, plan);

    // The placement heuristic only reads the running totals to judge distance to a target,
    // so without one the recalculation after every placement is pure waste
    const needsTotals = targeted.some(s => !statIsIgnored(machine, STAT_KEYS[s]));

    const initialIndexBoard = toIndexBoard(initialBoard, tables.indexOf);
    // A board is empty exactly when every cell it has is free, and which cells it has is fixed by its tier
    let openCellCount = 0;
    for (let i = 0; i < BOARD_CELLS; i++) if (initialIndexBoard[i] !== LOCKED) openCellCount++;

    return {
        tables, inventoryById, draw, plan, tierLength: plan.tierCount + 1, params, targeted, drawRanks,
        needsTotals, initialIndexBoard, openCellCount
    };
};
