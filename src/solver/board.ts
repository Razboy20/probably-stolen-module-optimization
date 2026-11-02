import type { GridTier, InventoryItem, MachineKind } from '../types';
import { BOARD_H, BOARD_W } from './geometry';

export type BoardCell = InventoryItem | 'Locked' | null;
export type Board = BoardCell[][];

// Which cells of the board a machine has: a standard machine's tier, or the Evaporator's, which has no tiers
export type GridLayout = GridTier | 'Evaporator';

export const layoutOf = (kind: MachineKind | null, tier: GridTier): GridLayout => kind === 'Evaporator' ? 'Evaporator' : tier;

// The Evaporator's grid is the top-left corner of the standard one with the rest locked off,
// so the left column and top row Side and Top Mount look for are the same cells on both
export const gridSizeOf = (layout: GridLayout) => layout === 'Evaporator' ? { rows: 4, cols: 4 } : { rows: BOARD_H, cols: BOARD_W };

const CORNER_CELLS = [[0, 0], [0, 6], [4, 0], [4, 6]];
const CENTER_CELLS = [[1, 3], [2, 2], [2, 3], [2, 4], [3, 3]];
// [y, x] of the cells each tier lacks
const TIER_LOCKED_CELLS: Record<GridTier, number[][]> = { 1: [...CORNER_CELLS, ...CENTER_CELLS], 2: CORNER_CELLS, 3: [] };

export const copyBoard = (board: Board): Board => board.map(row => [...row]);

export function initializeBoard(layout: GridLayout, initialIds?: (string | 'Locked' | null)[][], inventory?: InventoryItem[]): Board {
    const { rows, cols } = gridSizeOf(layout);
    const grid: Board = Array.from({ length: BOARD_H }, (_, y) => Array.from({ length: BOARD_W }, (_, x) => y < rows && x < cols ? null : 'Locked'));
    if (layout !== 'Evaporator') for (const [y, x] of TIER_LOCKED_CELLS[layout]) grid[y][x] = 'Locked';

    if (initialIds && inventory) {
        for (let y = 0; y < BOARD_H; y++) {
            for (let x = 0; x < BOARD_W; x++) {
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
