import { type BoardTotals, boardTotals } from './boardTotals';
import {
    BOARD_CELLS, cellMaskHi, cellMaskLo, MAX_PIECE_CELLS, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_MASK_HI, PLACE_MASK_LO, PLACE_META, PLACE_VALID,
    placeEntry
} from './geometry';
import { EMPTY, type IndexBoard } from './indexBoard';
import { compareTiers, objectiveTiers, TIER_VECTOR_LENGTH } from './objective';
import type { MachineSetup } from './setup';
import { FLAG_STUCK, type PoolTables } from './tables';

/* Every arrangement of the pieces already on one board
 * The search moves a few pieces at a time and judges each step, so it settles which pieces a board should hold far better than how they should lie:
 * two layouts of the same pieces that share no piece's cells, a board and its mirror image for one, have nothing but worse boards between them
 * This lifts every piece that is free to move and lays them all back down by an exact search over the open cells in scan order:
 * each cell is covered by a piece that is still up, or left empty while there are more open cells than the pieces still need
 * Every complete layout is scored with the solver's own objective, and the best one is kept
 *
 * The search is resumable, so its host can spread it over frames, and gives up at a step limit, since a board of many small pieces has more layouts than can be visited
 */
export interface ArrangementSearch {
    // Tries up to `budget` more placements, and reports whether the search is over
    advance: (budget: number) => boolean;
    // Whether every arrangement was visited, which makes `best` the best there is
    exhausted: () => boolean;
    // How many times a better arrangement has turned up
    improvements: () => number;
    // The best arrangement found so far, which is the board as given until a better one turns up
    best: IndexBoard;
}

const NOTHING = -1;
const LEFT_EMPTY = -2;

export const arrangeBoard = (tables: PoolTables, machine: MachineSetup, given: IndexBoard, stepLimit: number): ArrangementSearch => {
    const { plan, params } = machine;
    const tierLength = plan.tierCount + 1;
    const board = Int32Array.from(given.subarray(0, BOARD_CELLS));
    const best = Int32Array.from(board);

    const totals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    const tiers = new Int32Array(TIER_VECTOR_LENGTH);
    const bestTiers = new Int32Array(TIER_VECTOR_LENGTH);
    boardTotals(tables, board, totals);
    objectiveTiers(totals, plan, params, bestTiers);
    let improvements = 0;

    // Pieces the objective cannot tell apart are one kind, and only the next unused piece of a kind is ever tried on a cell
    const kindOf = new Map<string, number[]>();
    let cellsNeeded = 0;
    let remaining = 0;
    for (let i = 0; i < BOARD_CELLS; i++) {
        const item = board[i];
        if (item < 0 || (tables.flags[item] & FLAG_STUCK) !== 0) continue;
        board[i] = EMPTY;
        cellsNeeded++;
        const key = `${tables.shape[item]}|${tables.p[item]}|${tables.q[item]}|${tables.e[item]}|${tables.flags[item]}`;
        const kind = kindOf.get(key) ?? [];
        if (!kind.includes(item)) {
            kind.push(item);
            remaining++;
        }
        kindOf.set(key, kind);
    }
    const kinds = [...kindOf.values()];
    const used = new Int32Array(kinds.length);

    let occupiedLo = 0;
    let occupiedHi = 0;
    let slack = -cellsNeeded;
    for (let i = 0; i < BOARD_CELLS; i++) {
        if (board[i] === EMPTY) {
            slack++;
        } else {
            occupiedLo |= cellMaskLo(i);
            occupiedHi |= cellMaskHi(i);
        }
    }

    // One frame per decided cell: the cell, the next kind and orientation to try on it, and what was done there (a placement, left empty, or nothing yet)
    const frameCell = new Int32Array(BOARD_CELLS + 1);
    const frameKind = new Int32Array(BOARD_CELLS + 1);
    const frameOrient = new Int32Array(BOARD_CELLS + 1);
    const frameDone = new Int32Array(BOARD_CELLS + 1);
    const frameUsedKind = new Int32Array(BOARD_CELLS + 1);

    // The first open cell from `from` on becomes frame d's decision; with none left the frame cannot be opened
    const openFrame = (d: number, from: number) => {
        let cell = from;
        while (cell < BOARD_CELLS && ((occupiedLo & cellMaskLo(cell)) | (occupiedHi & cellMaskHi(cell))) !== 0) cell++;
        if (cell === BOARD_CELLS) return false;
        frameCell[d] = cell;
        frameKind[d] = 0;
        frameOrient[d] = 0;
        frameDone[d] = NOTHING;
        return true;
    };

    const undo = (d: number) => {
        const done = frameDone[d];
        if (done === NOTHING) return;
        frameDone[d] = NOTHING;
        if (done === LEFT_EMPTY) {
            occupiedLo &= ~cellMaskLo(frameCell[d]);
            occupiedHi &= ~cellMaskHi(frameCell[d]);
            slack++;
            return;
        }
        for (let i = 0; i < PLACE_CELL_COUNT[done]; i++) board[PLACE_CELLS[done * MAX_PIECE_CELLS + i]] = EMPTY;
        occupiedLo &= ~PLACE_MASK_LO[done];
        occupiedHi &= ~PLACE_MASK_HI[done];
        used[frameUsedKind[d]]--;
        remaining++;
    };

    // Every orientation's first cell is the first it covers in scan order, so a placement anchored on the first open cell never reaches back over a decided one
    // The kinds are tried in turn, each in every orientation, and leaving the cell empty comes last
    const tryNext = (d: number) => {
        const cell = frameCell[d];
        let k = frameKind[d];
        let o = frameOrient[d];
        for (; k < kinds.length; k++, o = 0) {
            if (used[k] === kinds[k].length) continue;
            const item = kinds[k][used[k]];
            for (; o < tables.orientCount[item]; o++) {
                const entry = placeEntry(tables.orientStart[item] + o, cell);
                if ((PLACE_META[entry] & PLACE_VALID) === 0) continue;
                if (((PLACE_MASK_LO[entry] & occupiedLo) | (PLACE_MASK_HI[entry] & occupiedHi)) !== 0) continue;

                for (let i = 0; i < PLACE_CELL_COUNT[entry]; i++) board[PLACE_CELLS[entry * MAX_PIECE_CELLS + i]] = item;
                occupiedLo |= PLACE_MASK_LO[entry];
                occupiedHi |= PLACE_MASK_HI[entry];
                used[k]++;
                remaining--;
                frameKind[d] = k;
                frameOrient[d] = o + 1;
                frameDone[d] = entry;
                frameUsedKind[d] = k;
                return true;
            }
        }
        frameKind[d] = kinds.length + 1;
        if (k !== kinds.length || slack === 0) return false;
        occupiedLo |= cellMaskLo(cell);
        occupiedHi |= cellMaskHi(cell);
        slack--;
        frameDone[d] = LEFT_EMPTY;
        return true;
    };

    const scoreLayout = () => {
        boardTotals(tables, board, totals);
        objectiveTiers(totals, plan, params, tiers);
        if (compareTiers(tiers, bestTiers, tierLength) <= 0) return;
        bestTiers.set(tiers);
        best.set(board);
        improvements++;
    };

    let steps = 0;
    let depth = remaining > 0 && slack >= 0 && openFrame(0, 0) ? 0 : -1;

    const advance = (budget: number) => {
        const stopAt = Math.min(steps + budget, stepLimit);
        while (depth >= 0 && steps < stopAt) {
            undo(depth);
            if (!tryNext(depth)) {
                depth--;
                continue;
            }
            steps++;
            if (remaining === 0) scoreLayout();
            else if (openFrame(depth + 1, frameCell[depth] + 1)) depth++;
        }
        return depth < 0 || steps >= stepLimit;
    };

    return { advance, exhausted: () => depth < 0, improvements: () => improvements, best };
};
