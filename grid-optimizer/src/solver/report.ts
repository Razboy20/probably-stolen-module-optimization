import type { Stats } from '../types';
import type { Board } from './board';
import { calculateBoardStats } from './boardStats';
import { generateCodeFromState, inventoryForCode } from './codec';
import { fromIndexBoard } from './indexBoard';
import { boardOfSet, type SolveRequest, type SolveSetup } from './setup';

// What the UI is shown for one machine's record board: the object board, its solution code and the same stats the report computes
export interface BoardUpdate {
    board: Board;
    totals: Stats;
    pieceStats: Map<string, Stats>;
    code: string;
}

export interface SolveUpdate {
    // One per machine of the request, in its order
    boards: BoardUpdate[];
    // The record's combined objective, so a host running several solves can tell which report is the best
    tiers: number[];
}

const buildBoardUpdate = (request: SolveRequest, setup: SolveSetup, machineIndex: number, record: Int32Array): BoardUpdate => {
    const { machine } = request.machines[machineIndex];
    const { fullInventory } = request;
    const board = fromIndexBoard(boardOfSet(record, machineIndex), setup.tables.items);
    const code = generateCodeFromState(
        machine.tier, machine.maximizeStats, machine.targetStats, inventoryForCode(fullInventory, board), board
    );
    const { totals, pieceStats } = calculateBoardStats(board, fullInventory, setup.inventoryById, setup.tables.internal);
    return { board, totals, pieceStats, code };
};

export const buildUpdate = (request: SolveRequest, setup: SolveSetup, record: Int32Array, tiers: Int32Array): SolveUpdate => ({
    boards: request.machines.map((_, k) => buildBoardUpdate(request, setup, k, record)),
    tiers: Array.from(tiers.subarray(0, setup.tierLength))
});
