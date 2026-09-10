import type { GridTier, InventoryItem } from '../types';

export type BoardCell = InventoryItem | 'Locked' | null;
export type Board = BoardCell[][];

export const NEIGHBOR_DX = [0, 0, -1, 1];
export const NEIGHBOR_DY = [-1, 1, 0, 0];

export const copyBoard = (board: Board): Board => board.map(row => [...row]);

export function initializeBoard(currentTier: GridTier, initialIds?: (string | 'Locked' | null)[][], inventory?: InventoryItem[]): Board {
    const grid: Board = Array.from({ length: 5 }, () => Array.from({ length: 7 }, () => null));
    if (currentTier === 1 || currentTier === 2) {
        grid[0][0] = grid[0][6] = grid[4][0] = grid[4][6] = 'Locked';
    }
    if (currentTier === 1) {
        grid[1][3] = grid[2][2] = grid[2][3] = grid[2][4] = grid[3][3] = 'Locked';
    }

    if (initialIds && inventory) {
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 7; x++) {
                if (grid[y][x] === 'Locked') continue;
                const cellId = initialIds[y][x];
                if (cellId && cellId !== 'Locked') {
                    const item = inventory.find(i => i.id === cellId);
                    if (item) grid[y][x] = item;
                }
            }
        }
    }
    return grid;
}
