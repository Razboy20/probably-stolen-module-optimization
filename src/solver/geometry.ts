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
// The covered cells as a bitmask, cells 0-31 in the low word and 32-34 in the high word, so a fit test against the occupied cells is two ANDs
export const PLACE_MASK_LO = new Int32Array(PLACE_ENTRIES);
export const PLACE_MASK_HI = new Int32Array(PLACE_ENTRIES);

export const cellMaskLo = (cell: number) => (cell < 32 ? 1 << cell : 0);
export const cellMaskHi = (cell: number) => (cell >= 32 ? 1 << (cell - 32) : 0);

// Strides coprime to the 35 board cells, so a scan from any start visits every cell once
export const SCAN_STRIDES = Int32Array.of(1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 16, 17);

/* Whether an orientation fits anywhere on a board is a handful of shifts of the free-cell mask:
 * each cell of the piece, as an offset from the top-left corner of its bounding box, shifts the mask down so that bit c says "the cell at offset c from corner c is free",
 * and ANDing them over the piece's cells, then with the corners the piece stays in bounds from, leaves a bit per placement that fits
 */
export const ORIENT_CELL_COUNT = new Int32Array(ORIENT_TOTAL);
export const ORIENT_OFFSETS = new Int32Array(ORIENT_TOTAL * MAX_PIECE_CELLS);
export const ORIENT_CORNERS_LO = new Int32Array(ORIENT_TOTAL);
export const ORIENT_CORNERS_HI = new Int32Array(ORIENT_TOTAL);

for (let s = 0; s < SHAPE_COUNT; s++) {
    const orientations = PRECOMPUTED_ORIENTATIONS.get(SHAPE_LIST[s])!;
    for (let o = 0; o < orientations.length; o++) {
        const { xs, ys, count, minX, maxX, minY, maxY } = orientations[o];
        const g = ORIENT_START[s] + o;
        ORIENT_CELL_COUNT[g] = count;
        for (let i = 0; i < count; i++) ORIENT_OFFSETS[g * MAX_PIECE_CELLS + i] = (ys[i] - minY) * BOARD_W + (xs[i] - minX);
        for (let anchor = 0; anchor < BOARD_CELLS; anchor++) {
            const ax = anchor % BOARD_W;
            const ay = (anchor - ax) / BOARD_W;
            if (ax + minX < 0 || ax + maxX >= BOARD_W || ay + minY < 0 || ay + maxY >= BOARD_H) continue;

            const corner = (ay + minY) * BOARD_W + ax + minX;
            ORIENT_CORNERS_LO[g] |= cellMaskLo(corner);
            ORIENT_CORNERS_HI[g] |= cellMaskHi(corner);

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
                PLACE_MASK_LO[entry] |= cellMaskLo(cell);
                PLACE_MASK_HI[entry] |= cellMaskHi(cell);
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

// Whether any orientation of the shape fits in the free cells, given as the complement of the occupied mask
// An offset never exceeds 28, so the bits the high word contributes come from a double shift that is a plain zero at offset 0
export const shapeFitsFree = (shape: number, freeLo: number, freeHi: number) => {
    const hiBits = freeHi & 7;
    const orientEnd = ORIENT_START[shape] + ORIENT_COUNT[shape];
    for (let g = ORIENT_START[shape]; g < orientEnd; g++) {
        let lo = ORIENT_CORNERS_LO[g];
        let hi = ORIENT_CORNERS_HI[g];
        const base = g * MAX_PIECE_CELLS;
        for (let i = 0; i < ORIENT_CELL_COUNT[g]; i++) {
            const c = ORIENT_OFFSETS[base + i];
            lo &= (freeLo >>> c) | ((hiBits << (31 - c)) << 1);
            hi &= hiBits >>> c;
        }
        if ((lo | hi) !== 0) return true;
    }
    return false;
};
