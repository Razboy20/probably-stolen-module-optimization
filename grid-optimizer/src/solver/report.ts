import type { Stats } from '../types';
import type { Board } from './board';
import { calculateBoardStats } from './boardStats';
import { generateCodeFromState, inventoryForCode } from './codec';
import { fromIndexBoard, type IndexBoard } from './indexBoard';
import type { SolveRequest, SolveSetup } from './setup';

export interface SolveUpdate {
    board: Board;
    totals: Stats;
    pieceStats: Map<string, Stats>;
    code: string;
    // The record's objective, so a host running several solves can tell which report is the best
    tiers: number[];
}

// What the UI is shown for a record board: the object board, its solution code and the same stats the report computes
export const buildUpdate = (request: SolveRequest, setup: SolveSetup, record: IndexBoard, tiers: Int32Array): SolveUpdate => {
    const { machine, fullInventory } = request;
    const board = fromIndexBoard(record, setup.tables.items);
    const code = generateCodeFromState(
        machine.tier, machine.maximizeStats, machine.targetStats, inventoryForCode(fullInventory, board), board
    );
    const { totals, pieceStats } = calculateBoardStats(board, fullInventory, setup.inventoryById, setup.tables.internal);
    return { board, totals, pieceStats, code, tiers: Array.from(tiers.subarray(0, setup.tierLength)) };
};
