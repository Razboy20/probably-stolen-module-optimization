import type { InventoryItem } from '../types';
import type { Board } from './board';
import { BOARD_CELLS, BOARD_W } from './geometry';

// A board as the search sees it: one item index per cell, or one of the two sentinels
export type IndexBoard = Int32Array;
export const EMPTY = -1;
export const LOCKED = -2;

export const toIndexBoard = (board: Board, indexOf: Map<string, number>): IndexBoard => {
    const out = new Int32Array(BOARD_CELLS);
    for (let i = 0; i < BOARD_CELLS; i++) {
        const x = i % BOARD_W;
        const cell = board[(i - x) / BOARD_W][x];
        out[i] = cell === null ? EMPTY : cell === 'Locked' ? LOCKED : indexOf.get(cell.id)!;
    }
    return out;
};

export const fromIndexBoard = (board: IndexBoard, items: InventoryItem[]): Board =>
    Array.from({ length: 5 }, (_, y) => Array.from({ length: BOARD_W }, (_, x) => {
        const v = board[y * BOARD_W + x];
        return v === EMPTY ? null : v === LOCKED ? 'Locked' : items[v];
    }));
