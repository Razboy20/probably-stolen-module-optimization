import type { GridTier, InventoryItem } from '../types';
import { EVAPORATOR_GRID_SIZE, STANDARD_GRID_SIZE } from '../constants';

export type BoardCell = InventoryItem | 'Locked' | null;
export type Board = BoardCell[][];

export const NEIGHBOR_DX = [0, 0, -1, 1];
export const NEIGHBOR_DY = [-1, 1, 0, 0];

export const copyBoard = (board: Board): Board => board.map(row => [...row]);

// The Evaporator's grid is the top-left corner of the standard one with the rest locked off,
// so the left column and top row Side and Top Mount look for are the same cells on both
const lockOutsideEvaporatorGrid = (grid: Board) => {
    for (let y = 0; y < STANDARD_GRID_SIZE.rows; y++) {
        for (let x = 0; x < STANDARD_GRID_SIZE.cols; x++) {
            if (y >= EVAPORATOR_GRID_SIZE.rows || x >= EVAPORATOR_GRID_SIZE.cols) grid[y][x] = 'Locked';
        }
    }
};

// No tier locks the top row past the Evaporator's width, so that cell alone tells the two apart
export const isEvaporatorBoard = (board: Board) => board[0][EVAPORATOR_GRID_SIZE.cols] === 'Locked';

export function initializeBoard(currentTier: GridTier, initialIds?: (string | 'Locked' | null)[][], inventory?: InventoryItem[], isEvaporator = false): Board {
    const grid: Board = Array.from({ length: STANDARD_GRID_SIZE.rows }, () => Array.from({ length: STANDARD_GRID_SIZE.cols }, () => null));
    if (isEvaporator) {
        lockOutsideEvaporatorGrid(grid);
    } else {
        if (currentTier === 1 || currentTier === 2) {
            grid[0][0] = grid[0][6] = grid[4][0] = grid[4][6] = 'Locked';
        }
        if (currentTier === 1) {
            grid[1][3] = grid[2][2] = grid[2][3] = grid[2][4] = grid[3][3] = 'Locked';
        }
    }

    if (initialIds && inventory) {
        for (let y = 0; y < STANDARD_GRID_SIZE.rows; y++) {
            for (let x = 0; x < STANDARD_GRID_SIZE.cols; x++) {
                if (grid[y][x] === 'Locked') continue;
                const cellId = initialIds[y]?.[x];
                if (cellId && cellId !== 'Locked') {
                    const item = inventory.find(i => i.id === cellId);
                    if (item) grid[y][x] = item;
                }
            }
        }
    }
    return grid;
}
