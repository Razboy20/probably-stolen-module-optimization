import { type BoardTotals, boardTotals } from './boardTotals';
import { DRAW_TOURNAMENT, MAX_DRAWS } from './draw';
import { evalPlacement, type PlaceScore } from './evalPlacement';
import {
    BOARD_CELLS, cellMaskHi, cellMaskLo, MAX_PIECE_CELLS, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_MASK_HI, PLACE_MASK_LO, PLACE_META, PLACE_VALID,
    placeEntry, SCAN_STRIDES, SHAPE_COUNT, shapeFitsFree
} from './geometry';
import { EMPTY } from './indexBoard';
import { addMachineTiers, compareTiers, objectiveTiers, TIER_VECTOR_LENGTH } from './objective';
import { MAX_PIECES_PER_BOARD } from './pool';
import { buildUpdate, type SolveUpdate } from './report';
import { randomSeed, type Rng, rngBelow, rngCoinFlip, seedRng } from './rng';
import { boardOfSet, prepareSolve, type SolveRequest } from './setup';
import { FLAG_FIXED } from './tables';
import { createYielder, FRAME_BUDGET_MS, now, TIMER_YIELD_INTERVAL_MS } from './yielder';

// How many perturbations in a row the search rides out without a new record before it goes back to the record board
export const RESTART_AFTER_STAGNATIONS = 8;
/* Every so many restarts the search puts one board back to its initial state instead, so it does not spend all its time around one record
 * In a set that also hands the board's modules back to the pool for the other boards to pick up, which is a redistribution no single move makes
 */
export const FRESH_START_EVERY = 4;

export const STAGNATION_LIMIT = 150;

/* One iteration in so many offers a module off another board of the set to the draw, chosen uniformly among the ones this board could use
 * If the draw picks it and the fill places it, it is lifted from its owner, so the step is judged on what this board gains against what that board loses
 * Rebuilding one board at a time never moves a module otherwise: the owner would have to drop it and this board pick it up in the same accepted step
 * Offering every foreign module at once let the best-ranked ones win almost every steal iteration, and those swaps were nearly always rejected
 */
export const STEAL_ONE_IN = 4;

/* One iteration in so many drops the board's fixed pieces at random free placements instead of their best fit, and lets acceptance judge the result
 * A fixed piece's own score says nothing about what it costs the others: an Alarm across the top row scores the same as one anywhere else, while the
 * Top Mount modules that wanted that row do not. The best fit alone leaves it wherever the user put it
 */
export const WANDER_ONE_IN = 4;

// Unbiased Fisher-Yates over the first `count` entries
// `sort(() => Math.random() - 0.5)` is not a shuffle: it leaves the ordering strongly correlated with the input, which narrows the range of layouts the solver actually explores
const shuffleInPlace = (rng: Rng, arr: Int32Array, count: number) => {
    for (let i = count - 1; i > 0; i--) {
        const j = rngBelow(rng, i + 1);
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
};

const bitIsSet = (bits: Uint32Array, i: number) => (bits[i >>> 5] & (1 << (i & 31))) !== 0;
const setBit = (bits: Uint32Array, i: number) => { bits[i >>> 5] |= 1 << (i & 31); };
const clearBit = (bits: Uint32Array, i: number) => { bits[i >>> 5] &= ~(1 << (i & 31)); };

const copyTotals = (from: BoardTotals, to: BoardTotals) => {
    to.p = from.p; to.q = from.q; to.e = from.e; to.pieces = from.pieces;
};

export interface SolveControl {
    running: boolean;
}

export type { SolveRequest } from './setup';
export type { SolveUpdate } from './report';

export const runOptimizationEngine = async (
    request: SolveRequest,
    control: SolveControl,
    onUpdate: (update: SolveUpdate) => void
): Promise<{ iterations: number }> => {
    const maxIterations = request.maxIterations ?? Infinity;
    const rng = seedRng(request.seed ?? randomSeed(), request.thread ?? 0);

    const setup = prepareSolve(request);
    const { tables, tierLength, densityIndex, initialSet, machines } = setup;
    const machineCount = machines.length;
    // An iteration rebuilds one board, so a set gets as many tries per board before the big ruin as one board alone would
    const stagnationLimit = STAGNATION_LIMIT * machineCount;

    /* Each iteration rebuilds one board of the set, so what the fill reads about its machine is rebound when the board is picked
     * A set of one machine never picks and never touches the random stream for it
     */
    let { draw, plan, params, targeted, drawRanks, needsTotals, openCellCount } = machines[0];
    let { drawList, shapeStart, drawable } = draw;
    const bindMachine = (k: number) => {
        ({ draw, plan, params, targeted, drawRanks, needsTotals, openCellCount } = machines[k]);
        ({ drawList, shapeStart, drawable } = draw);
    };

    // The boards of the set laid end to end, worked on through one view per board
    const currentSet = new Int32Array(initialSet);
    // Reused across iterations so the search does not allocate a fresh board per attempt
    const testSet = new Int32Array(initialSet);
    // The best set ever seen, which is what gets reported
    // Once the search can restart, the set it is working on is no longer guaranteed to be the best one found,
    // so the record is kept separately. A restart must never be able to lose a result that has already been shown
    const bestSet = new Int32Array(initialSet);
    const currentViews = machines.map((_, k) => boardOfSet(currentSet, k));
    const testViews = machines.map((_, k) => boardOfSet(testSet, k));
    let testBoard = testViews[0];

    // The accepted totals and tier vector of every board, so only the rebuilt board is rescored per iteration
    const currentTotals = machines.map((): BoardTotals => ({ p: 0, q: 0, e: 0, pieces: 0 }));
    const boardTiers = machines.map(() => new Int32Array(TIER_VECTOR_LENGTH));
    const fillTotals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    const fillTiers = new Int32Array(TIER_VECTOR_LENGTH);
    const rescoreCurrent = () => {
        for (let k = 0; k < machineCount; k++) {
            boardTotals(tables, currentViews[k], currentTotals[k]);
            objectiveTiers(currentTotals[k], machines[k].plan, machines[k].params, boardTiers[k]);
        }
    };
    rescoreCurrent();

    // The board a steal took a module from this iteration, rescored alongside the rebuilt board
    let robbed = -1;
    const robbedTotals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    const robbedTiers = new Int32Array(TIER_VECTOR_LENGTH);

    // The set's objective is the sum of its boards' tier vectors
    const combineTiers = (rebuilt: number, out: Int32Array) => {
        out.fill(0);
        for (let k = 0; k < machineCount; k++) {
            const tiers = k === rebuilt ? fillTiers : k === robbed ? robbedTiers : boardTiers[k];
            addMachineTiers(out, tiers, machines[k].plan.tierCount, densityIndex);
        }
    };

    const currentTiers = new Int32Array(TIER_VECTOR_LENGTH);
    // epochTiers is the best this attempt has reached, bestTiers the best ever reached
    // They are the same thing until the search restarts; (see RESTART_AFTER_STAGNATIONS)
    const epochTiers = new Int32Array(TIER_VECTOR_LENGTH);
    const bestTiers = new Int32Array(TIER_VECTOR_LENGTH);
    let hasEpoch = false;
    let hasRecord = false;

    let stagnationCounter = 0;
    let stagnationRuns = 0;
    let restarts = 0;

    // Scratch, cleared per iteration rather than reallocated
    // blocked: items sitting on the board this iteration, so the draw skips them
    const blocked = new Uint32Array((tables.count + 31) >>> 5);
    // Movable pieces on the board, in the order they are first met
    const removable = new Int32Array(MAX_PIECES_PER_BOARD);
    // Movable pieces on the other boards that this board's layout can draw, with the board each one stands on
    const stealable = new Int32Array(MAX_PIECES_PER_BOARD * machineCount);
    const stealableOwner = new Int8Array(MAX_PIECES_PER_BOARD * machineCount);
    // Every special or locked piece sitting on the rebuilt board, with the cells it is currently standing on
    // They stay put until the fill lifts them one at a time, so the board is never in a state where one is missing
    const fixedItem = new Int32Array(MAX_PIECES_PER_BOARD);
    const fixedCellCount = new Int32Array(MAX_PIECES_PER_BOARD);
    const fixedCells = new Int32Array(MAX_PIECES_PER_BOARD * MAX_PIECE_CELLS);
    // Every orientation is normalised so its first cell is the anchor, so only empty cells can anchor a placement
    // Tracking them prunes the scan as the board fills up
    const freeCells = new Int32Array(BOARD_CELLS);
    let freeCount = 0;
    // The occupied cells of the board being built, as the two-word mask the placement tables are matched against
    let occupiedLo = 0;
    let occupiedHi = 0;
    const score: PlaceScore = { ok: false, major: 0, minor: 0 };

    // Drops the cells that are no longer free, keeping the rest in scan order
    const compactFreeCells = () => {
        let write = 0;
        for (let c = 0; c < freeCount; c++) {
            if (testBoard[freeCells[c]] === EMPTY) freeCells[write++] = freeCells[c];
        }
        freeCount = write;
    };

    /* The draw offers modules of the shapes that still fit and still have a module off the board, and nothing else
     * A candidate is a uniform position across those shapes' runs of the draw list, so every offered module is as likely as any other,
     * and one that turns out to be on the board already just sits out the tournament
     */
    const shapeBlocked = new Int32Array(SHAPE_COUNT);
    const shapeOffered = (shape: number, infeasible: number) =>
        (infeasible & (1 << shape)) === 0 && shapeBlocked[shape] < shapeStart[shape + 1] - shapeStart[shape];

    // Whether a shape fits is a property of the shape and the free cells, never of the individual module, and filling a board only ever removes free cells
    // So a shape with nowhere left to go is settled here from the masks, without waiting for the draw to offer a module of it and fail
    // Only the shapes the draw could still offer are worth settling
    const infeasibleShapesNow = (known: number) => {
        let infeasible = known;
        for (let shape = 0; shape < SHAPE_COUNT; shape++) {
            if (shapeOffered(shape, infeasible) && !shapeFitsFree(shape, ~occupiedLo, ~occupiedHi)) infeasible |= 1 << shape;
        }
        return infeasible;
    };

    const drawWeight = (infeasible: number) => {
        let weight = 0;
        for (let shape = 0; shape < SHAPE_COUNT; shape++) {
            if (shapeOffered(shape, infeasible)) weight += shapeStart[shape + 1] - shapeStart[shape];
        }
        return weight;
    };

    const drawPosition = (infeasible: number, r: number) => {
        let rest = r;
        for (let shape = 0; shape < SHAPE_COUNT; shape++) {
            if (!shapeOffered(shape, infeasible)) continue;
            const run = shapeStart[shape + 1] - shapeStart[shape];
            if (rest < run) return shapeStart[shape] + rest;
            rest -= run;
        }
        return -1;
    };

    // Best of DRAW_TOURNAMENT candidates by rank; -1 when every candidate is already on the board
    const drawTournament = (infeasible: number, weight: number, ranks: Int32Array) => {
        let pick = -1;
        for (let t = 0; t < DRAW_TOURNAMENT; t++) {
            const pos = drawPosition(infeasible, rngBelow(rng, weight));
            if (bitIsSet(blocked, drawList[pos])) continue;
            if (pick === -1 || ranks[pos] > ranks[pick]) pick = pos;
        }
        return pick;
    };

    /* Commits one piece at its best-scoring placement among the free cells, and reports whether it found one
     * The pool fill and the special-module relocation both go through here on purpose:
     * a special is only allowed to move because the fill can judge where it should go, and it has to judge it on exactly the terms it judges everything else
     *
     * `incumbent` is the placement the piece is standing in already, for a piece being relocated rather than placed for the first time
     * It is scored ahead of everything else and the scan only takes a strictly better cell, so a piece with nowhere better to be simply stays
     * Without it the ties decide, and for a module whose score barely varies across the board (which is every special) that means the first cell in scan order:
     * a Line4 lands in the top-left of whatever the ruin opened up, every iteration, taking the best space on the board from the modules that would have earned something with it
     */
    const placeBestFit = (item: number, boardIsEmpty: boolean, incumbent: number) => {
        let bestEntry = -1;
        let haveScore = incumbent === -1;
        let bestMajor = 0, bestMinor = 0;

        if (incumbent !== -1) {
            evalPlacement(tables, incumbent, item, testBoard, occupiedLo, occupiedHi, boardIsEmpty, params, fillTotals.p, fillTotals.q, fillTotals.e, score);
            if (score.ok) {
                haveScore = true;
                bestMajor = score.major; bestMinor = score.minor;
                bestEntry = incumbent;
            }
        }

        const orientStart = tables.orientStart[item];
        const orientEnd = orientStart + tables.orientCount[item];
        for (let c = 0; c < freeCount; c++) {
            const anchor = freeCells[c];
            for (let g = orientStart; g < orientEnd; g++) {
                const entry = placeEntry(g, anchor);
                // Most anchors on a 7x5 board are out of bounds for a given orientation, and the table settles it without the scoring call
                if ((PLACE_META[entry] & PLACE_VALID) === 0) continue;

                evalPlacement(tables, entry, item, testBoard, occupiedLo, occupiedHi, boardIsEmpty, params, fillTotals.p, fillTotals.q, fillTotals.e, score);
                if (!score.ok) continue;
                if (!haveScore || score.major > bestMajor || (score.major === bestMajor && score.minor > bestMinor)) {
                    haveScore = true;
                    bestMajor = score.major; bestMinor = score.minor;
                    bestEntry = entry;
                }
            }
        }

        if (bestEntry === -1) return false;
        commitPlacement(bestEntry, item);
        return true;
    };

    const commitPlacement = (entry: number, item: number) => {
        const cellCount = PLACE_CELL_COUNT[entry];
        for (let i = 0; i < cellCount; i++) testBoard[PLACE_CELLS[entry * MAX_PIECE_CELLS + i]] = item;
        occupiedLo |= PLACE_MASK_LO[entry];
        occupiedHi |= PLACE_MASK_HI[entry];
        compactFreeCells();
    };

    const placementIsOpen = (entry: number) =>
        (PLACE_META[entry] & PLACE_VALID) !== 0 && ((PLACE_MASK_LO[entry] & occupiedLo) | (PLACE_MASK_HI[entry] & occupiedHi)) === 0;

    // A uniformly random open placement; the piece's own cells are free, so there is always at least one
    const placeAnywhere = (item: number) => {
        const orientStart = tables.orientStart[item];
        const orientEnd = orientStart + tables.orientCount[item];
        let open = 0;
        for (let c = 0; c < freeCount; c++) {
            for (let g = orientStart; g < orientEnd; g++) if (placementIsOpen(placeEntry(g, freeCells[c]))) open++;
        }
        let pick = rngBelow(rng, open);
        for (let c = 0; c < freeCount; c++) {
            for (let g = orientStart; g < orientEnd; g++) {
                const entry = placeEntry(g, freeCells[c]);
                if (!placementIsOpen(entry)) continue;
                if (pick === 0) { commitPlacement(entry, item); return; }
                pick--;
            }
        }
    };

    // The cells were collected in row-major order, and an orientation's cells are in that same order and anchored on its first cell,
    // so the cells the piece is standing on say which placement it is standing in
    const homeEntryOf = (fixed: number) => {
        const item = fixedItem[fixed];
        const cellCount = fixedCellCount[fixed];
        const anchor = fixedCells[fixed * MAX_PIECE_CELLS];
        const orientEnd = tables.orientStart[item] + tables.orientCount[item];
        for (let g = tables.orientStart[item]; g < orientEnd; g++) {
            const entry = placeEntry(g, anchor);
            if ((PLACE_META[entry] & PLACE_VALID) === 0 || PLACE_CELL_COUNT[entry] !== cellCount) continue;
            let matches = true;
            for (let i = 0; i < cellCount; i++) {
                if (fixedCells[fixed * MAX_PIECE_CELLS + i] !== PLACE_CELLS[entry * MAX_PIECE_CELLS + i]) { matches = false; break; }
            }
            if (matches) return entry;
        }
        return -1;
    };

    // Every module on another board of the set is out of this board's reach, and counts against its shape like a kept piece here would
    // The movable ones this board's layout can draw are noted, since a steal may offer them after all
    const blockOtherBoards = (board: number) => {
        let stealableCount = 0;
        for (let k = 0; k < machineCount; k++) {
            if (k === board) continue;
            const other = testViews[k];
            for (let i = 0; i < BOARD_CELLS; i++) {
                const item = other[i];
                if (item < 0 || bitIsSet(blocked, item)) continue;
                setBit(blocked, item);
                if (drawable[item] === 0) continue;
                shapeBlocked[tables.shape[item]]++;
                if ((tables.flags[item] & FLAG_FIXED) !== 0) continue;
                stealable[stealableCount] = item;
                stealableOwner[stealableCount++] = k;
            }
        }
        return stealableCount;
    };

    // One of the other boards' modules, chosen uniformly, joins the draw for this iteration; it moves only if the draw picks it and the fill places it
    let offered = -1;
    let offeredOwner = -1;
    const offerForSteal = (offerCount: number) => {
        const s = rngBelow(rng, offerCount);
        offered = stealable[s];
        offeredOwner = stealableOwner[s];
        clearBit(blocked, offered);
        shapeBlocked[tables.shape[offered]]--;
    };

    const takeFromOwner = () => {
        robbed = offeredOwner;
        const source = testViews[robbed];
        for (let i = 0; i < BOARD_CELLS; i++) if (source[i] === offered) source[i] = EMPTY;
    };

    // A board goes back to its initial state minus the modules the other boards have since taken from it, since they keep theirs
    const freshStart = (k: number) => {
        const view = currentViews[k];
        view.set(boardOfSet(initialSet, k));
        if (machineCount === 1) return;
        blocked.fill(0);
        for (let other = 0; other < machineCount; other++) {
            if (other === k) continue;
            for (let i = 0; i < BOARD_CELLS; i++) if (currentViews[other][i] >= 0) setBit(blocked, currentViews[other][i]);
        }
        for (let i = 0; i < BOARD_CELLS; i++) if (view[i] >= 0 && bitIsSet(blocked, view[i])) view[i] = EMPTY;
    };

    let pendingUpdate = false;
    const flushUpdate = () => {
        if (!pendingUpdate) return;
        pendingUpdate = false;
        onUpdate(buildUpdate(request, setup, bestSet, bestTiers));
    };

    const { portYield, timerYield, dispose } = createYielder();
    let lastYield = now();
    let lastTimerYield = lastYield;
    let iterations = 0;

    try {
        while (control.running && iterations < maxIterations) {
            iterations++;
            const isStagnant = stagnationCounter >= stagnationLimit;

            let board = 0;
            if (machineCount > 1) {
                board = rngBelow(rng, machineCount);
                bindMachine(board);
                testBoard = testViews[board];
            }

            testSet.set(currentSet);
            blocked.fill(0);
            shapeBlocked.fill(0);
            robbed = -1;

            const stealableCount = blockOtherBoards(board);
            offered = -1;
            if (stealableCount > 0 && !isStagnant && rngBelow(rng, STEAL_ONE_IN) === 0) offerForSteal(stealableCount);

            let removableCount = 0;
            let fixedCount = 0;
            for (let i = 0; i < BOARD_CELLS; i++) {
                const item = testBoard[i];
                if (item < 0) continue;

                /* A special is offered a better cell every iteration instead of taking a share of the ruin's removal budget
                 * It is not the ruin's kind of move: the ruin takes a piece away and lets the fill find something better to do with the space,
                 * and a special is going straight back down whatever happens. Spending a removal on one only means one fewer real piece is reconsidered that iteration
                 */
                if ((tables.flags[item] & FLAG_FIXED) !== 0) {
                    let f = 0;
                    while (f < fixedCount && fixedItem[f] !== item) f++;
                    if (f === fixedCount) {
                        fixedItem[fixedCount] = item;
                        fixedCellCount[fixedCount] = 0;
                        fixedCount++;
                    }
                    fixedCells[f * MAX_PIECE_CELLS + fixedCellCount[f]++] = i;
                    continue;
                }

                if (!bitIsSet(blocked, item)) {
                    setBit(blocked, item);
                    removable[removableCount++] = item;
                }
            }

            let removeCount = 0;
            if (removableCount > 0) {
                // A stagnant board loses half to nine tenths of its pieces, an ordinary iteration one to three
                removeCount = isStagnant
                    ? Math.max(1, Math.trunc(removableCount * (50 + rngBelow(rng, 40)) / 100))
                    : rngBelow(rng, Math.min(3, removableCount)) + 1;

                shuffleInPlace(rng, removable, removableCount);
                for (let i = 0; i < removeCount; i++) clearBit(blocked, removable[i]);

                for (let i = 0; i < BOARD_CELLS; i++) {
                    const item = testBoard[i];
                    if (item >= 0 && (tables.flags[item] & FLAG_FIXED) === 0 && !bitIsSet(blocked, item)) testBoard[i] = EMPTY;
                }
            }

            for (let i = removeCount; i < removableCount; i++) {
                const item = removable[i];
                if (drawable[item] !== 0) shapeBlocked[tables.shape[item]]++;
            }

            freeCount = 0;
            occupiedLo = 0;
            occupiedHi = 0;
            // The free cells are visited from a random cell with a random stride coprime to the board size, which is what breaks ties between equally scored placements
            // A full shuffle did the same job for a draw per cell
            const scanStart = rngBelow(rng, BOARD_CELLS);
            const scanStride = SCAN_STRIDES[rngBelow(rng, SCAN_STRIDES.length)];
            for (let k = 0; k < BOARD_CELLS; k++) {
                const i = (scanStart + k * scanStride) % BOARD_CELLS;
                if (testBoard[i] === EMPTY) {
                    freeCells[freeCount++] = i;
                } else {
                    occupiedLo |= cellMaskLo(i);
                    occupiedHi |= cellMaskHi(i);
                }
            }
            let boardIsEmpty = freeCount === openCellCount;

            if (needsTotals) boardTotals(tables, testBoard, fillTotals);
            else fillTotals.p = fillTotals.q = fillTotals.e = 0;

            /* The board's specials, offered a better cell one at a time and before anything is drawn
             *
             * One at a time is what makes this safe: at the moment a special is placed its own cells are still free, and an anchor-normalised orientation always has an
             * anchor among them, so placeBestFit can never come back empty-handed and a special can never be lost on the way. Lifting them all at once would let the
             * first one take the second one's cells and leave the second with nowhere guaranteed to go
             * Going before the draw also means they choose out of the whole ruined area rather than whatever the fill leaves over
             */
            const wander = fixedCount > 0 && rngBelow(rng, WANDER_ONE_IN) === 0;
            for (let f = 0; f < fixedCount; f++) {
                const home = homeEntryOf(f);
                for (let c = 0; c < fixedCellCount[f]; c++) {
                    const idx = fixedCells[f * MAX_PIECE_CELLS + c];
                    testBoard[idx] = EMPTY;
                    occupiedLo &= ~cellMaskLo(idx);
                    occupiedHi &= ~cellMaskHi(idx);
                    freeCells[freeCount++] = idx;
                }
                if (wander) placeAnywhere(fixedItem[f]);
                else placeBestFit(fixedItem[f], boardIsEmpty, home);
                boardIsEmpty = false;
                if (needsTotals) boardTotals(tables, testBoard, fillTotals);
            }

            let infeasibleShapes = infeasibleShapesNow(0);

            // Which of the machine's targets the accepted board already meets picks the draw table, so the fill stops being offered more of a stat it has enough of
            let metMask = 0;
            for (let i = 0; i < targeted.length; i++) {
                const s = targeted[i];
                const accepted = currentTotals[board];
                const t = s === 0 ? accepted.p : s === 1 ? accepted.q : accepted.e;
                if (t >= params.target[s]) metMask |= 1 << i;
            }
            const ranks = drawRanks[metMask];

            // Stops once nothing off the board has a shape that fits, which cannot change while the board is only losing free cells
            let weight = drawWeight(infeasibleShapes);
            for (let drawn = 0; drawn < MAX_DRAWS && weight > 0; drawn++) {
                const pos = drawTournament(infeasibleShapes, weight, ranks);
                if (pos === -1) continue;
                const item = drawList[pos];
                const shape = tables.shape[item];

                if (!placeBestFit(item, boardIsEmpty, -1)) {
                    infeasibleShapes |= 1 << shape;
                } else {
                    setBit(blocked, item);
                    shapeBlocked[shape]++;
                    boardIsEmpty = false;
                    if (needsTotals) boardTotals(tables, testBoard, fillTotals);
                    infeasibleShapes = infeasibleShapesNow(infeasibleShapes);
                    if (item === offered) takeFromOwner();
                }
                weight = drawWeight(infeasibleShapes);
            }

            boardTotals(tables, testBoard, fillTotals);
            objectiveTiers(fillTotals, plan, params, fillTiers);
            if (robbed !== -1) {
                boardTotals(tables, testViews[robbed], robbedTotals);
                objectiveTiers(robbedTotals, machines[robbed].plan, machines[robbed].params, robbedTiers);
            }
            combineTiers(board, currentTiers);

            if (!control.running) break;

            /* Judged against the best of THIS attempt, not the best ever
             * After a restart the board is deliberately worse than the record, and comparing it to the record would reject every move and leave the restart unable to climb at all
             *
             * The big ruin of a stagnant board is a perturbation, not a move, and is kept whatever it scores: judged like a move it would almost never survive on a board that
             * has been climbed for hundreds of iterations, and the search would only ever leave a local optimum by starting over. Keeping it lets the climb resume from a
             * board that still has most of a good packing, which is a different place to look than a fresh greedy build. The record is banked in bestBoard either way
             */
            const ordering = hasEpoch ? compareTiers(currentTiers, epochTiers, tierLength) : 1;
            const improved = ordering > 0;
            if (isStagnant || improved || (ordering === 0 && rngCoinFlip(rng))) {
                if (isStagnant || improved) {
                    epochTiers.set(currentTiers);
                    hasEpoch = true;
                }
                currentSet.set(testSet);
                copyTotals(fillTotals, currentTotals[board]);
                boardTiers[board].set(fillTiers);
                if (robbed !== -1) {
                    copyTotals(robbedTotals, currentTotals[robbed]);
                    boardTiers[robbed].set(robbedTiers);
                }

                // Any step up from the epoch's best is progress worth riding out; a new record is additionally the only thing worth reporting
                if (improved) {
                    stagnationCounter = 0;
                    if (!hasRecord || compareTiers(currentTiers, bestTiers, tierLength) > 0) {
                        bestTiers.set(currentTiers);
                        hasRecord = true;
                        bestSet.set(currentSet);
                        pendingUpdate = true;
                    }
                } else {
                    stagnationCounter++;
                }
            } else {
                stagnationCounter++;
            }

            if (isStagnant) {
                stagnationCounter = 0;
                // Perturbing the same neighbourhood for long without a new record means the record is probably as good as this region gets
                // The next perturbations go back to the record itself, and now and then to the initial board so the search also sees other regions
                if (++stagnationRuns >= RESTART_AFTER_STAGNATIONS) {
                    stagnationRuns = 0;
                    hasEpoch = false;
                    currentSet.set(bestSet);
                    if (++restarts % FRESH_START_EVERY === 0) freshStart(machineCount > 1 ? rngBelow(rng, machineCount) : 0);
                    rescoreCurrent();
                }
            }

            if (now() - lastYield >= FRAME_BUDGET_MS) {
                flushUpdate();
                if (now() - lastTimerYield >= TIMER_YIELD_INTERVAL_MS) {
                    await timerYield();
                    lastTimerYield = now();
                } else {
                    await portYield();
                }
                lastYield = now();
            }
        }
    } finally {
        flushUpdate();
        dispose();
    }

    return { iterations };
};
