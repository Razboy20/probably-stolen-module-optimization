import { type BoardTotals, boardTotals } from './boardTotals';
import {
    BOARD_CELLS, cellMaskHi, cellMaskLo, MAX_PIECE_CELLS, ORIENT_OFFSETS, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_MASK_HI, PLACE_MASK_LO, PLACE_META,
    PLACE_VALID, placeEntry
} from './geometry';
import { EMPTY, type IndexBoard } from './indexBoard';
import { compareTiers, objectiveTiers, TIER_VECTOR_LENGTH } from './objective';
import type { MachineSetup } from './setup';
import { FLAG_STUCK, type PoolTables } from './tables';

/* Every arrangement of the pieces already on one board
 * The search moves a few pieces at a time and judges each step, so it settles which pieces a board should hold far better than how they should lie:
 * two layouts of the same pieces that share no piece's cells, a board and its mirror image for one, have nothing but worse boards between them
 * This lifts every piece that is free to move and lays them all back down by an exact search in two layers
 * The outer one covers the open cells in scan order with outlines alone: each cell takes the outline of a piece that is still up,
 * or is left empty while there are more open cells than the pieces still need
 * The inner one takes each complete covering and hands the pieces out over the places of their outline in every way that differs
 * Fitting outlines together is where the dead ends are, and this way they are met once and not once per way of handing the pieces out;
 * a dead end is also remembered by the cells it had covered and the outlines it had left, so no other order of placing walks into it again
 * Every complete layout is scored with the solver's own objective, and the best one is kept
 *
 * The search is resumable, so its host can spread it over frames, and gives up at a step limit, since a board of many small pieces has more layouts than can be visited
 */
export interface ArrangementSearch {
    // Takes up to `budget` more steps, each an outline placed or a piece handed out, and reports whether the search is over
    advance: (budget: number) => boolean;
    // Whether every arrangement was visited, which makes `best` the best there is
    exhausted: () => boolean;
    // How many complete layouts have been scored
    layouts: () => number;
    // How many times a better arrangement has turned up
    improvements: () => number;
    // The best arrangement found so far, which is the board as given until a better one turns up
    best: IndexBoard;
}

const NOTHING = -1;
const LEFT_EMPTY = -2;
// A state number has to stay an exact integer, which leaves 18 bits above the cells for the outlines still up
const MAX_OUTLINE_STATES = 2 ** 18;
// Each remembered state costs memory, and the boards that can be searched through at all have well under this many
const MAX_DEAD_STATES = 2 ** 19;

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
    let layouts = 0;

    const lifted = new Set<number>();
    let occupiedLo = 0;
    let occupiedHi = 0;
    let slack = 0;
    for (let i = 0; i < BOARD_CELLS; i++) {
        const item = board[i];
        if (item === EMPTY) {
            slack++;
        } else if (item < 0 || (tables.flags[item] & FLAG_STUCK) !== 0) {
            occupiedLo |= cellMaskLo(i);
            occupiedHi |= cellMaskHi(i);
        } else {
            board[i] = EMPTY;
            lifted.add(item);
        }
    }
    const cleared = Int32Array.from(board);
    const pieceCount = lifted.size;

    // Pieces the objective cannot tell apart are one kind, and kinds that cover the same cells in every orientation share an outline
    const kinds: number[][] = [];
    const outlineKinds: number[][] = [];
    const outlineOrientStart: number[] = [];
    const outlineOrientCount: number[] = [];
    const outlineLeft: number[] = [];
    const kindIndex = new Map<string, number>();
    const outlineIndex = new Map<string, number>();
    for (const item of lifted) {
        const start = tables.orientStart[item];
        const count = tables.orientCount[item];
        const cells = ORIENT_OFFSETS.subarray(start * MAX_PIECE_CELLS, (start + count) * MAX_PIECE_CELLS).join();
        let outline = outlineIndex.get(cells);
        if (outline === undefined) {
            outline = outlineKinds.length;
            outlineIndex.set(cells, outline);
            outlineKinds.push([]);
            outlineOrientStart.push(start);
            outlineOrientCount.push(count);
            outlineLeft.push(0);
        }
        outlineLeft[outline]++;

        const key = `${tables.shape[item]}|${tables.p[item]}|${tables.q[item]}|${tables.e[item]}|${tables.flags[item]}`;
        let kind = kindIndex.get(key);
        if (kind === undefined) {
            kind = kinds.length;
            kindIndex.set(key, kind);
            kinds.push([]);
            outlineKinds[outline].push(kind);
        }
        kinds[kind].push(item);
    }
    const outlineCount = outlineKinds.length;
    const used = new Int32Array(kinds.length);
    let remaining = pieceCount;

    // One frame per decided cell: the cell, the next outline and orientation to try on it, and what was done there (a placement, left empty, or nothing yet)
    const frameCell = new Int32Array(BOARD_CELLS + 1);
    const frameOutline = new Int32Array(BOARD_CELLS + 1);
    const frameOrient = new Int32Array(BOARD_CELLS + 1);
    const frameDone = new Int32Array(BOARD_CELLS + 1);

    // The first open cell from `from` on becomes frame d's decision; with none left the frame cannot be opened
    const openFrame = (d: number, from: number) => {
        let cell = from;
        while (cell < BOARD_CELLS && ((occupiedLo & cellMaskLo(cell)) | (occupiedHi & cellMaskHi(cell))) !== 0) cell++;
        if (cell === BOARD_CELLS) return false;
        frameCell[d] = cell;
        frameOutline[d] = 0;
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
        occupiedLo &= ~PLACE_MASK_LO[done];
        occupiedHi &= ~PLACE_MASK_HI[done];
        outlineLeft[frameOutline[d]]++;
        remaining++;
    };

    // Every orientation's first cell is the first it covers in scan order, so a placement anchored on the first open cell never reaches back over a decided one
    // The outlines are tried in turn, each in every orientation, and leaving the cell empty comes last
    const tryNext = (d: number) => {
        const cell = frameCell[d];
        let outline = frameOutline[d];
        let o = frameOrient[d];
        for (; outline < outlineCount; outline++, o = 0) {
            if (outlineLeft[outline] === 0) continue;
            for (; o < outlineOrientCount[outline]; o++) {
                const entry = placeEntry(outlineOrientStart[outline] + o, cell);
                if ((PLACE_META[entry] & PLACE_VALID) === 0) continue;
                if (((PLACE_MASK_LO[entry] & occupiedLo) | (PLACE_MASK_HI[entry] & occupiedHi)) !== 0) continue;

                occupiedLo |= PLACE_MASK_LO[entry];
                occupiedHi |= PLACE_MASK_HI[entry];
                outlineLeft[outline]--;
                remaining--;
                frameOutline[d] = outline;
                frameOrient[d] = o + 1;
                frameDone[d] = entry;
                return true;
            }
        }
        frameOutline[d] = outlineCount + 1;
        if (outline !== outlineCount || slack === 0) return false;
        occupiedLo |= cellMaskLo(cell);
        occupiedHi |= cellMaskHi(cell);
        slack--;
        frameDone[d] = LEFT_EMPTY;
        return true;
    };

    /* What is still to come depends only on which cells are covered and which outlines are still up, and many orders of placing lead to the same such state
     * One that was searched through without a single complete covering is remembered, and never entered again
     * The state is one number: the covered cells in the low 35 bits, and above them the count of each outline still up as the digits of a mixed-radix number
     */
    const outlineWeight: number[] = [];
    let stateCount = 1;
    for (let outline = 0; outline < outlineCount; outline++) {
        outlineWeight.push(stateCount);
        stateCount *= outlineLeft[outline] + 1;
    }
    const remembers = stateCount <= MAX_OUTLINE_STATES;
    const stateKey = () => {
        let up = 0;
        for (let outline = 0; outline < outlineCount; outline++) up += outlineLeft[outline] * outlineWeight[outline];
        return (occupiedLo >>> 0) + 2 ** 32 * (occupiedHi + 8 * up);
    };
    const deadStates = new Set<number>();
    // The state each frame was opened in, and how many coverings had been completed by then
    const frameState = new Float64Array(BOARD_CELLS + 1);
    const frameCoverings = new Float64Array(BOARD_CELLS + 1);
    let coverings = 0;

    // The places of the covering in hand and the outline each one takes, with one slot per place: the next kind to try on it and the kind it holds
    const slotEntry = new Int32Array(pieceCount);
    const slotOutline = new Int32Array(pieceCount);
    const slotNext = new Int32Array(pieceCount);
    const slotKind = new Int32Array(pieceCount).fill(NOTHING);

    // Places whose outline has one kind come first, so the pieces with no choice are laid once per covering
    const openSlots = (depth: number) => {
        board.set(cleared);
        let count = 0;
        for (const choice of [false, true]) {
            for (let d = 0; d <= depth; d++) {
                if (frameDone[d] < 0 || (outlineKinds[frameOutline[d]].length > 1) !== choice) continue;
                slotEntry[count] = frameDone[d];
                slotOutline[count] = frameOutline[d];
                count++;
            }
        }
        slotNext[0] = 0;
    };

    const handOut = (slot: number) => {
        if (slotKind[slot] !== NOTHING) {
            used[slotKind[slot]]--;
            slotKind[slot] = NOTHING;
        }
        const choices = outlineKinds[slotOutline[slot]];
        for (let c = slotNext[slot]; c < choices.length; c++) {
            const kind = choices[c];
            if (used[kind] === kinds[kind].length) continue;
            const item = kinds[kind][used[kind]++];
            const entry = slotEntry[slot];
            for (let i = 0; i < PLACE_CELL_COUNT[entry]; i++) board[PLACE_CELLS[entry * MAX_PIECE_CELLS + i]] = item;
            slotKind[slot] = kind;
            slotNext[slot] = c + 1;
            return true;
        }
        return false;
    };

    const scoreLayout = () => {
        layouts++;
        boardTotals(tables, board, totals);
        objectiveTiers(totals, plan, params, tiers);
        if (compareTiers(tiers, bestTiers, tierLength) <= 0) return;
        bestTiers.set(tiers);
        best.set(board);
        improvements++;
    };

    let steps = 0;
    let depth = remaining > 0 && openFrame(0, 0) ? 0 : -1;
    // The slot being decided while a covering's pieces are handed out, and -1 while the covering itself is still being made
    let slot = -1;

    const advance = (budget: number) => {
        const stopAt = Math.min(steps + budget, stepLimit);
        while ((slot >= 0 || depth >= 0) && steps < stopAt) {
            if (slot >= 0) {
                if (!handOut(slot)) {
                    slot--;
                    continue;
                }
                steps++;
                if (slot === pieceCount - 1) scoreLayout();
                else slotNext[++slot] = 0;
                continue;
            }
            undo(depth);
            if (!tryNext(depth)) {
                if (coverings === frameCoverings[depth] && remembers && deadStates.size < MAX_DEAD_STATES) deadStates.add(frameState[depth]);
                depth--;
                continue;
            }
            steps++;
            if (remaining === 0) {
                coverings++;
                openSlots(depth);
                slot = 0;
                continue;
            }
            const state = stateKey();
            if (deadStates.has(state) || !openFrame(depth + 1, frameCell[depth] + 1)) continue;
            depth++;
            frameState[depth] = state;
            frameCoverings[depth] = coverings;
        }
        return (slot < 0 && depth < 0) || steps >= stepLimit;
    };

    return { advance, exhausted: () => slot < 0 && depth < 0, layouts: () => layouts, improvements: () => improvements, best };
};
