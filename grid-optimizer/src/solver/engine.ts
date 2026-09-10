import type { InventoryItem, Stats } from '../types';
import type { Board } from './board';
import { calculateBoardStats, indexInventoryById } from './boardStats';
import { type BoardTotals, boardTotals } from './boardTotals';
import { generateCodeFromState, inventoryForCode } from './codec';
import { buildDrawRanks, targetedStats } from './draw';
import { evalPlacement, type PlaceScore } from './evalPlacement';
import {
    BOARD_CELLS, MAX_PIECE_CELLS, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_META, PLACE_VALID, placeEntry
} from './geometry';
import { EMPTY, fromIndexBoard, LOCKED, toIndexBoard } from './indexBoard';
import {
    buildTierPlan, compareTiers, type MachineConfig, objectiveTiers, STAT_KEYS, statIsIgnored, TIER_VECTOR_LENGTH, tierBoost
} from './objective';
import { buildSearchPool, MAX_PIECES_PER_BOARD } from './pool';
import { randomSeed, type Rng, rngBelow, rngCoinFlip, seedRng } from './rng';
import { buildScoringParams } from './scoring';
import { buildPoolTables, FLAG_FIXED } from './tables';
import { createYielder, FRAME_BUDGET_MS, now, TIMER_YIELD_INTERVAL_MS } from './yielder';

// A coordinated multi-machine mode (shared pool consumption, a balance penalty across machines, rebuild-all on stagnation)
// existed at commit e5c614c. The UI only ever ran one machine per engine, so it was removed
// Run All is N independent single-machine solves

// How many random candidates a draw compares before taking the best of them
// 1 is a uniform draw; higher steers the fill toward high-value modules without ever ruling any out
const DRAW_TOURNAMENT = 4;

// How many stagnations in a row the search rides out on the same board before it gives up on it and starts over from the initial board
const RESTART_AFTER_STAGNATIONS = 8;

const STAGNATION_LIMIT = 150;

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

export interface SolveRequest {
    machine: MachineConfig;
    initialBoard: Board;
    searchPoolInventory: InventoryItem[];
    fullInventory: InventoryItem[];
    seed?: number;
    // Which stream of the seed this solve follows, so a population of solves can share one seed and still diverge
    thread?: number;
    maxIterations?: number;
}

export interface SolveControl {
    running: boolean;
}

export interface SolveUpdate {
    board: Board;
    totals: Stats;
    pieceStats: Map<string, Stats>;
    code: string;
    // The record's objective, so a host running several solves can tell which report is the best
    tiers: number[];
}

export const runOptimizationEngine = async (
    request: SolveRequest,
    control: SolveControl,
    onUpdate: (update: SolveUpdate) => void
): Promise<{ iterations: number }> => {
    const { machine, initialBoard, searchPoolInventory, fullInventory } = request;
    const maxIterations = request.maxIterations ?? Infinity;
    const rng = seedRng(request.seed ?? randomSeed(), request.thread ?? 0);

    const tables = buildPoolTables(fullInventory, initialBoard);
    const inventoryById = indexInventoryById(fullInventory);

    // Boards may already hold modules the search itself would not pick up, so the pruned pool is only used for choosing what to place
    const searchPool = buildSearchPool(searchPoolInventory, tables.internal, machine).filter(item => tables.indexOf.has(item.id));
    const drawList = Int32Array.from(searchPool, item => tables.indexOf.get(item.id)!);
    const poolCount = drawList.length;
    let allShapesMask = 0;
    for (let i = 0; i < poolCount; i++) allShapesMask |= 1 << tables.shape[drawList[i]];

    // A board holds at most MAX_PIECES_PER_BOARD pieces, so shuffling the entire pool to fill one was work thrown away
    // This permutation is drawn from a Fisher-Yates that stops as soon as the board can take nothing more
    // and it persists across iterations, staying a valid permutation because a partial shuffle only ever swaps within it
    const poolOrder = new Int32Array(poolCount);
    for (let i = 0; i < poolCount; i++) poolOrder[i] = i;

    const plan = buildTierPlan(machine);
    const tierLength = plan.tierCount + 1;

    // A stat marked ignored gets weight 0 so the placement heuristic stops steering away from it at all
    const placementWeights: Stats = { Performance: 0, Quality: 0, Efficiency: 0 };
    for (let s = 0; s < 3; s++) {
        const key = STAT_KEYS[s];
        if (statIsIgnored(machine, key)) continue;
        let w = 0;
        if (machine.maximizeStats[key]) w += 10;
        if (machine.targetStats[key] !== null) w += 15;
        placementWeights[key] = w * tierBoost(plan, s);
    }
    const params = buildScoringParams(machine, placementWeights);
    const targeted = targetedStats(machine);
    const drawRanks = buildDrawRanks(tables, drawList, machine, plan);

    // The placement heuristic only reads the running totals to judge distance to a target,
    // so without one the recalculation after every placement is pure waste
    const needsTotals = targeted.some(s => !statIsIgnored(machine, STAT_KEYS[s]));

    const initialIndexBoard = toIndexBoard(initialBoard, tables.indexOf);
    const currentBoard = new Int32Array(initialIndexBoard);
    // Reused across iterations so the search does not allocate a fresh board per attempt
    const testBoard = new Int32Array(initialIndexBoard);
    // The best board ever seen, which is what gets reported
    // Once the search can restart, the board it is working on is no longer guaranteed to be the best one found,
    // so the record is kept separately. A restart must never be able to lose a result that has already been shown
    const bestBoard = new Int32Array(initialIndexBoard);

    const currentTotals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    const fillTotals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    boardTotals(tables, currentBoard, currentTotals);

    const currentTiers = new Int32Array(TIER_VECTOR_LENGTH);
    // epochTiers is the best this attempt has reached, bestTiers the best ever reached
    // They are the same thing until the search restarts; (see RESTART_AFTER_STAGNATIONS)
    const epochTiers = new Int32Array(TIER_VECTOR_LENGTH);
    const bestTiers = new Int32Array(TIER_VECTOR_LENGTH);
    let hasEpoch = false;
    let hasRecord = false;

    let stagnationCounter = 0;
    let stagnationRuns = 0;

    // Scratch, cleared per iteration rather than reallocated
    // blocked: items sitting on the board this iteration, so the draw skips them
    const blocked = new Uint32Array((tables.count + 31) >>> 5);
    // Movable pieces on the board, in the order they are first met
    const removable = new Int32Array(MAX_PIECES_PER_BOARD);
    // Every special or locked piece sitting on the rebuilt board, with the cells it is currently standing on
    // They stay put until the fill lifts them one at a time, so the board is never in a state where one is missing
    const fixedItem = new Int32Array(MAX_PIECES_PER_BOARD);
    const fixedCellCount = new Int32Array(MAX_PIECES_PER_BOARD);
    const fixedCells = new Int32Array(MAX_PIECES_PER_BOARD * MAX_PIECE_CELLS);
    // Every orientation is normalised so its first cell is the anchor, so only empty cells can anchor a placement
    // Tracking them prunes the scan as the board fills up
    const freeCells = new Int32Array(BOARD_CELLS);
    let freeCount = 0;
    const score: PlaceScore = { ok: false, major: 0, minor: 0 };

    // A board is empty exactly when every cell it has is free, and which cells it has is fixed by its tier
    let openCellCount = 0;
    for (let i = 0; i < BOARD_CELLS; i++) if (initialIndexBoard[i] !== LOCKED) openCellCount++;

    // Drops the cells that are no longer free, keeping the rest in scan order
    const compactFreeCells = () => {
        let write = 0;
        for (let c = 0; c < freeCount; c++) {
            if (testBoard[freeCells[c]] === EMPTY) freeCells[write++] = freeCells[c];
        }
        freeCount = write;
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
            evalPlacement(tables, incumbent, item, testBoard, boardIsEmpty, params, fillTotals.p, fillTotals.q, fillTotals.e, score);
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

                evalPlacement(tables, entry, item, testBoard, boardIsEmpty, params, fillTotals.p, fillTotals.q, fillTotals.e, score);
                if (!score.ok) continue;
                if (!haveScore || score.major > bestMajor || (score.major === bestMajor && score.minor > bestMinor)) {
                    haveScore = true;
                    bestMajor = score.major; bestMinor = score.minor;
                    bestEntry = entry;
                }
            }
        }

        if (bestEntry === -1) return false;

        const cellCount = PLACE_CELL_COUNT[bestEntry];
        for (let i = 0; i < cellCount; i++) testBoard[PLACE_CELLS[bestEntry * MAX_PIECE_CELLS + i]] = item;
        compactFreeCells();
        return true;
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

    let currentCode = '';
    let codeIsStale = true;
    let pendingUpdate = false;
    let reportedBoard: Board = initialBoard;
    const flushUpdate = () => {
        if (!pendingUpdate) return;
        pendingUpdate = false;

        if (codeIsStale) {
            reportedBoard = fromIndexBoard(bestBoard, tables.items);
            currentCode = generateCodeFromState(
                machine.tier, machine.maximizeStats, machine.targetStats,
                inventoryForCode(fullInventory, reportedBoard), reportedBoard
            );
            codeIsStale = false;
        }
        const { totals, pieceStats } = calculateBoardStats(reportedBoard, fullInventory, inventoryById, tables.internal);
        onUpdate({ board: reportedBoard, totals, pieceStats, code: currentCode, tiers: Array.from(bestTiers.subarray(0, tierLength)) });
    };

    const { portYield, timerYield, dispose } = createYielder();
    let lastYield = now();
    let lastTimerYield = lastYield;
    let iterations = 0;

    try {
        while (control.running && iterations < maxIterations) {
            iterations++;
            const isStagnant = stagnationCounter >= STAGNATION_LIMIT;

            testBoard.set(currentBoard);
            blocked.fill(0);

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

            if (removableCount > 0) {
                // A stagnant board loses half to nine tenths of its pieces, an ordinary iteration one to three
                const removeCount = isStagnant
                    ? Math.max(1, Math.trunc(removableCount * (50 + rngBelow(rng, 40)) / 100))
                    : rngBelow(rng, Math.min(3, removableCount)) + 1;

                shuffleInPlace(rng, removable, removableCount);
                for (let i = 0; i < removeCount; i++) clearBit(blocked, removable[i]);

                for (let i = 0; i < BOARD_CELLS; i++) {
                    const item = testBoard[i];
                    if (item >= 0 && (tables.flags[item] & FLAG_FIXED) === 0 && !bitIsSet(blocked, item)) testBoard[i] = EMPTY;
                }
            }

            freeCount = 0;
            for (let i = 0; i < BOARD_CELLS; i++) {
                if (testBoard[i] === EMPTY) freeCells[freeCount++] = i;
            }
            shuffleInPlace(rng, freeCells, freeCount);
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
            for (let f = 0; f < fixedCount; f++) {
                const home = homeEntryOf(f);
                for (let c = 0; c < fixedCellCount[f]; c++) {
                    const idx = fixedCells[f * MAX_PIECE_CELLS + c];
                    testBoard[idx] = EMPTY;
                    freeCells[freeCount++] = idx;
                }
                placeBestFit(fixedItem[f], boardIsEmpty, home);
                boardIsEmpty = false;
                if (needsTotals) boardTotals(tables, testBoard, fillTotals);
            }

            // Whether a shape fits is a property of the shape and the free cells, never of the individual module, and filling a board only ever removes free cells
            // So once one module of a shape finds nowhere to go, every later module of that shape in the same pass finds nowhere either, and can be skipped without scanning
            let infeasibleShapes = 0;

            // Which of the machine's targets the accepted board already meets picks the draw table, so the fill stops being offered more of a stat it has enough of
            let metMask = 0;
            for (let i = 0; i < targeted.length; i++) {
                const s = targeted[i];
                const t = s === 0 ? currentTotals.p : s === 1 ? currentTotals.q : currentTotals.e;
                if (t >= params.target[s]) metMask |= 1 << i;
            }
            const ranks = drawRanks[metMask];

            // Stops once every shape in the pool has been shown to fit nowhere, which cannot change while the board is only losing free cells
            let drawn = 0;
            while (drawn < poolCount && (infeasibleShapes & allShapesMask) !== allShapesMask) {
                // Best of DRAW_TOURNAMENT random candidates rather than the first one drawn
                // The permutation is still only partially shuffled, so the losers stay in the undrawn region and can be picked again later this fill
                const remaining = poolCount - drawn;
                let swapAt = drawn + rngBelow(rng, remaining);
                for (let t = 1; t < DRAW_TOURNAMENT && t < remaining; t++) {
                    const alt = drawn + rngBelow(rng, remaining);
                    if (ranks[poolOrder[alt]] > ranks[poolOrder[swapAt]]) swapAt = alt;
                }
                const pos = poolOrder[swapAt];
                poolOrder[swapAt] = poolOrder[drawn];
                poolOrder[drawn] = pos;
                drawn++;

                const item = drawList[pos];
                if (bitIsSet(blocked, item)) continue;

                const shapeBit = 1 << tables.shape[item];
                if ((infeasibleShapes & shapeBit) !== 0) continue;

                // Too few cells left for this shape is itself a permanent verdict on it
                if (freeCount < tables.size[item]) {
                    infeasibleShapes |= shapeBit;
                    continue;
                }

                if (placeBestFit(item, boardIsEmpty, -1)) {
                    boardIsEmpty = false;
                    if (needsTotals) boardTotals(tables, testBoard, fillTotals);
                } else {
                    infeasibleShapes |= shapeBit;
                }
            }

            boardTotals(tables, testBoard, fillTotals);
            objectiveTiers(fillTotals, plan, params, currentTiers);

            if (!control.running) break;

            // Judged against the best of THIS attempt, not the best ever
            // After a restart the board is deliberately worse than the record, and comparing it to the record would reject every move and leave the restart unable to climb at all
            const ordering = hasEpoch ? compareTiers(currentTiers, epochTiers, tierLength) : 1;
            const improved = ordering > 0;
            if (improved || (ordering === 0 && rngCoinFlip(rng))) {
                if (improved) {
                    epochTiers.set(currentTiers);
                    hasEpoch = true;
                }
                currentBoard.set(testBoard);
                copyTotals(fillTotals, currentTotals);

                // A new record is the only thing worth reporting, and the only thing that resets the stagnation count
                if (improved && (!hasRecord || compareTiers(currentTiers, bestTiers, tierLength) > 0)) {
                    bestTiers.set(currentTiers);
                    hasRecord = true;
                    bestBoard.set(currentBoard);
                    codeIsStale = true;
                    pendingUpdate = true;
                    stagnationCounter = 0;
                } else {
                    stagnationCounter++;
                }
            } else {
                stagnationCounter++;
            }

            if (isStagnant) {
                stagnationCounter = 0;
                // A big ruin is still judged against the epoch's best, so it only ever gets kept if it comes out ahead
                // On a board that has been hill-climbed for thousands of iterations it almost never does
                // Past a point it is very likely that this attempt is finished, and the iterations are better spent on a fresh one than on shaking the same board forever
                // The record is already banked in bestBoard, so a restart can only cost time, never a result
                if (++stagnationRuns >= RESTART_AFTER_STAGNATIONS) {
                    stagnationRuns = 0;
                    hasEpoch = false;
                    currentBoard.set(initialIndexBoard);
                    boardTotals(tables, currentBoard, currentTotals);
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
