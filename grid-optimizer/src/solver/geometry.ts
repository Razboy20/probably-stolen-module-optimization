import type { ModuleShape } from '../types';
import { SHAPE_DEFINITIONS } from '../constants';
import { PRECOMPUTED_ORIENTATIONS } from '../utils';
import { NEIGHBOR_DX, NEIGHBOR_DY } from './board';

export const BOARD_W = 7;
export const BOARD_H = 5;
export const BOARD_CELLS = BOARD_W * BOARD_H;
export const MAX_PIECE_CELLS = 5;
// A piece of at most 5 cells has at most 12 in-bounds neighbouring cells (its perimeter)
export const MAX_PIECE_NEIGHBORS = 12;

export const SHAPE_LIST = Object.keys(SHAPE_DEFINITIONS) as ModuleShape[];
export const SHAPE_COUNT = SHAPE_LIST.length;
export const shapeIndexOf = (shape: ModuleShape) => SHAPE_LIST.indexOf(shape);

/* Every orientation of every shape laid out flat, so a piece is described by a start index and a count
 * The per-orientation data is what the placement scan needs for one (orientation, anchor) pair without walking the board around the piece:
 * the cells it covers, the in-bounds cells around it (excluding its own), and whether it touches the board edge or the left column / top row
 */
export const ORIENT_START = new Int32Array(SHAPE_COUNT);
export const ORIENT_COUNT = new Int32Array(SHAPE_COUNT);
export const ORIENT_TOTAL = (() => {
    let total = 0;
    for (let s = 0; s < SHAPE_COUNT; s++) {
        ORIENT_START[s] = total;
        ORIENT_COUNT[s] = PRECOMPUTED_ORIENTATIONS.get(SHAPE_LIST[s])!.length;
        total += ORIENT_COUNT[s];
    }
    return total;
})();

export const PLACE_VALID = 1;
export const PLACE_TOUCHES_EDGE = 2;
export const PLACE_LEFT_COL = 4;
export const PLACE_TOP_ROW = 8;

export const PLACE_ENTRIES = ORIENT_TOTAL * BOARD_CELLS;
export const placeEntry = (orientation: number, anchor: number) => orientation * BOARD_CELLS + anchor;

export const PLACE_META = new Int32Array(PLACE_ENTRIES);
export const PLACE_CELL_COUNT = new Int32Array(PLACE_ENTRIES);
export const PLACE_CELLS = new Int32Array(PLACE_ENTRIES * MAX_PIECE_CELLS);
export const PLACE_NBR_COUNT = new Int32Array(PLACE_ENTRIES);
export const PLACE_NBRS = new Int32Array(PLACE_ENTRIES * MAX_PIECE_NEIGHBORS);

for (let s = 0; s < SHAPE_COUNT; s++) {
    const orientations = PRECOMPUTED_ORIENTATIONS.get(SHAPE_LIST[s])!;
    for (let o = 0; o < orientations.length; o++) {
        const { xs, ys, count, minX, maxX, minY, maxY } = orientations[o];
        const g = ORIENT_START[s] + o;
        for (let anchor = 0; anchor < BOARD_CELLS; anchor++) {
            const ax = anchor % BOARD_W;
            const ay = (anchor - ax) / BOARD_W;
            if (ax + minX < 0 || ax + maxX >= BOARD_W || ay + minY < 0 || ay + maxY >= BOARD_H) continue;

            const entry = placeEntry(g, anchor);
            let meta = PLACE_VALID;
            if (ax + minX === 0) meta |= PLACE_LEFT_COL;
            if (ay + minY === 0) meta |= PLACE_TOP_ROW;

            const covered = new Set<number>();
            for (let i = 0; i < count; i++) {
                const px = ax + xs[i];
                const py = ay + ys[i];
                const cell = py * BOARD_W + px;
                PLACE_CELLS[entry * MAX_PIECE_CELLS + i] = cell;
                covered.add(cell);
                if (px === 0 || px === BOARD_W - 1 || py === 0 || py === BOARD_H - 1) meta |= PLACE_TOUCHES_EDGE;
            }
            PLACE_CELL_COUNT[entry] = count;

            let nbrCount = 0;
            for (let i = 0; i < count; i++) {
                const px = ax + xs[i];
                const py = ay + ys[i];
                for (let d = 0; d < 4; d++) {
                    const nx = px + NEIGHBOR_DX[d];
                    const ny = py + NEIGHBOR_DY[d];
                    if (nx < 0 || nx >= BOARD_W || ny < 0 || ny >= BOARD_H) continue;
                    const cell = ny * BOARD_W + nx;
                    if (covered.has(cell)) continue;
                    if (nbrCount === MAX_PIECE_NEIGHBORS) throw new Error(`Orientation ${g} at ${anchor} has more than ${MAX_PIECE_NEIGHBORS} neighbours`);
                    PLACE_NBRS[entry * MAX_PIECE_NEIGHBORS + nbrCount++] = cell;
                }
            }
            PLACE_NBR_COUNT[entry] = nbrCount;
            PLACE_META[entry] = meta;
        }
    }
}
