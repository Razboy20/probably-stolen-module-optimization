import type { InventoryItem, ModuleShape, Stats } from '../types';
import { type Orientation, applyInternalEffects, PRECOMPUTED_ORIENTATIONS } from '../utils';
import { type Board, copyBoard } from './board';
import { type BoardStats, calculateBoardStats, indexInventoryById } from './boardStats';
import { generateCodeFromState, inventoryForCode } from './codec';
import {
    type MachineConfig, STAT_KEYS, DEFAULT_STAT_PRIORITY, PRIORITY_WEIGHT_STEP,
    compareTiers, priorityOf, statIsIgnored
} from './objective';
import { type PlacementContext, buildPlacementContext, evaluatePlacementDelta } from './placement';
import { buildSearchPool, isSpecialModule } from './pool';
import { createYielder, FRAME_BUDGET_MS, now, TIMER_YIELD_INTERVAL_MS } from './yielder';

// A coordinated multi-machine mode (shared pool consumption, a balance penalty across machines, rebuild-all on stagnation)
// existed at commit e5c614c. The UI only ever ran one machine per engine, so it was removed
// Run All is N independent single-machine solves

// How many random candidates a draw compares before taking the best of them
// 1 is a uniform draw; higher steers the fill toward high-value modules without ever ruling any out
const DRAW_TOURNAMENT = 4;

// How much of its draw weight a targeted stat keeps once the target is met
const TARGET_MET_DRAW_SCALE = 0.25;

// How many stagnations in a row the search rides out on the same board before it gives up on it and starts over from the initial board
const RESTART_AFTER_STAGNATIONS = 8;

const STAGNATION_LIMIT = 150;

// Unbiased Fisher-Yates
// `sort(() => Math.random() - 0.5)` is not a shuffle: it leaves the ordering strongly correlated with the input, which narrows the range of layouts the solver actually explores
const shuffleInPlace = <T,>(arr: T[]) => {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
};

export interface SolveRequest {
    machine: MachineConfig;
    initialBoard: Board;
    searchPoolInventory: InventoryItem[];
    fullInventory: InventoryItem[];
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
}

export const runOptimizationEngine = async (
    request: SolveRequest,
    control: SolveControl,
    onUpdate: (update: SolveUpdate) => void
): Promise<{ iterations: number }> => {
    const { machine, initialBoard, searchPoolInventory, fullInventory } = request;
    const maxIterations = request.maxIterations ?? Infinity;

    const precomputedInternal = new Map<string, Stats>();
    fullInventory.forEach(item => precomputedInternal.set(item.id, applyInternalEffects(item)));

    const placementContexts = new Map<string, PlacementContext>();
    fullInventory.forEach(item => placementContexts.set(item.id, buildPlacementContext(item, precomputedInternal)));

    // Boards may already hold modules the search itself would not pick up, so the pruned pool is only used for choosing what to place
    const searchPool = buildSearchPool(searchPoolInventory, precomputedInternal, machine);

    // The inner loops used to key everything off item.id, which meant hashing a string for every pool entry on every iteration
    // Everything hot is addressed by pool index instead:
    // the item -> index map is walked once per placed cell (a few dozen), and the per-candidate work becomes a typed-array read
    const poolIndexOf = new Map<string, number>();
    searchPool.forEach((item, i) => poolIndexOf.set(item.id, i));
    const ctxByIndex: PlacementContext[] = searchPool.map(item => placementContexts.get(item.id)!);
    const orientationsByIndex = searchPool.map(item => PRECOMPUTED_ORIENTATIONS.get(item.shape));

    // A board holds at most MAX_PIECES_PER_BOARD pieces, so shuffling the entire pool to fill one was work thrown away
    // This permutation is drawn from a Fisher-Yates that stops as soon as the board can take nothing more
    // and it persists across iterations, staying a valid permutation because a partial shuffle only ever swaps within it
    const poolOrder = new Int32Array(searchPool.length);
    for (let i = 0; i < poolOrder.length; i++) poolOrder[i] = i;
    const poolShapeCount = new Set(searchPool.map(item => item.shape)).size;
    const inventoryById = indexInventoryById(fullInventory);

    // One tier per distinct rank in play, most important first
    // With no priorities set this collapses to a single tier holding everything, which is the original objective
    const activeRanks = new Set<number>();
    for (const key of STAT_KEYS) {
        if (!statIsIgnored(machine, key)) activeRanks.add(priorityOf(machine, key));
    }
    if (activeRanks.size === 0) activeRanks.add(DEFAULT_STAT_PRIORITY);
    const rankOrder = [...activeRanks].sort((a, b) => a - b);
    const tierCount = rankOrder.length;
    const tierOfRank = new Map<number, number>(rankOrder.map((r, i) => [r, i]));
    const tierOf = (key: keyof Stats) => tierOfRank.get(priorityOf(machine, key))!;

    const currentTiers = new Float64Array(tierCount + 1);
    // epochTiers is the best this attempt has reached, bestTiers the best ever reached
    // They are the same thing until the search restarts; (see RESTART_AFTER_STAGNATIONS)
    const epochTiers = new Float64Array(tierCount + 1).fill(-Infinity);
    const bestTiers = new Float64Array(tierCount + 1).fill(-Infinity);
    let currentBoard = copyBoard(initialBoard);
    let currentStats: BoardStats = calculateBoardStats(currentBoard, fullInventory, inventoryById);
    let currentCode = '';
    let codeIsStale = true;

    // The best board ever seen, which is what gets reported
    // Once the search can restart, the board it is working on is no longer guaranteed to be the best one found,
    // so the record is kept separately. A restart must never be able to lose a result that has already been shown
    let bestBoard = copyBoard(currentBoard);
    let bestStats = currentStats;

    // Reused across iterations so the search does not allocate a fresh board per attempt
    const testBoard = copyBoard(currentBoard);

    let stagnationCounter = 0;
    let stagnationRuns = 0;

    // A stat marked ignored gets weight 0 so the placement heuristic stops steering away from it at all
    const getW = (key: keyof Stats) => {
        if (statIsIgnored(machine, key)) return 0;
        let w = 0;
        if (machine.maximizeStats[key]) w += 10;
        if (machine.targetStats[key] !== null) w += 15;
        return w;
    };
    const boost = (key: keyof Stats) =>
        statIsIgnored(machine, key) ? 1 : Math.pow(PRIORITY_WEIGHT_STEP, tierCount - 1 - tierOf(key));
    const dynWp = getW('Performance') * boost('Performance');
    const dynWq = getW('Quality') * boost('Quality');
    const dynWe = getW('Efficiency') * boost('Efficiency');

    /* What each pool entry is worth to this machine per cell it occupies, used to decide which modules the fill is offered
     * Board space is the scarce resource, so per-cell is the comparison that matters
     *
     * These weights are not the placement weights above, and the difference is the point
     * The score pays nothing for overshooting a target, so once a target is met, another point of that stat is worth next to nothing for a stat that is only being held to a target
     * A machine sitting comfortably above P 500 and Q 150 should be offered Efficiency modules, not more of what it already has
     * The placement heuristic keeps the full target weight in every case, and that is what stops a met target being undercut. This only decides what gets offered to it
     *
     * Which targets are met changes as the board moves, so there is one table per combination of met targets (at most eight), built once here, picked by bitmask per fill
     * Rebuilding a table whenever the totals shifted would costs more than the bias is worth
     */
    const targetedKeys = STAT_KEYS.filter(key => machine.targetStats[key] !== null);

    const buildDrawValues = (w: Stats) => {
        const values = new Float64Array(searchPool.length);
        const stated: number[] = [];
        for (let i = 0; i < searchPool.length; i++) {
            const ctx = ctxByIndex[i];
            if (ctx.isWhite || ctx.size === 0) continue;
            const s = ctx.internal;
            values[i] = (s.Performance * w.Performance + s.Quality * w.Quality
                + s.Efficiency * w.Efficiency) / ctx.size;
            stated.push(values[i]);
        }
        // Nodes carry no stats of their own: all their worth is the 20% they add to whatever ends up beside them, which only evaluatePlacementDelta can see
        // Scoring them at the median leaves them drawn about as often as an ordinary module instead of never
        if (stated.length > 0) {
            stated.sort((a, b) => a - b);
            const median = stated[stated.length >> 1];
            for (let i = 0; i < searchPool.length; i++) {
                if (ctxByIndex[i].isWhite) values[i] = median;
            }
        }
        return values;
    };

    const drawValues: Float64Array[] = [];
    for (let mask = 0; mask < (1 << targetedKeys.length); mask++) {
        const w: Stats = { Performance: 0, Quality: 0, Efficiency: 0 };
        for (const key of STAT_KEYS) {
            if (statIsIgnored(machine, key)) {
                w[key] = 0;
                continue;
            }
            let v = 0;
            if (machine.maximizeStats[key]) v += 10;
            const ti = targetedKeys.indexOf(key);
            // Bit set means the target is already met, so the push for it is cut back, but never to nothing
            // The draw decides which modules the fill is even offered,
            // so a stat whose modules stop being drawn cannot be rebuilt when a ruin knocks it below its target, and every repair after that is rejected
            if (ti !== -1) v += 15 * ((mask & (1 << ti)) !== 0 ? TARGET_MET_DRAW_SCALE : 1);
            w[key] = v * Math.pow(PRIORITY_WEIGHT_STEP, tierCount - 1 - tierOf(key));
        }
        drawValues.push(buildDrawValues(w));
    }

    // Scratch buffers, cleared per iteration rather than reallocated
    // placedMark: pool entries sitting on the board this iteration
    // Generation-stamped rather than cleared, so a new iteration invalidates every stale entry by bumping a counter instead of walking the array
    const placedMark = new Int32Array(searchPool.length);
    let markGen = 0;
    const piecesOnTarget: InventoryItem[] = [];
    const seenOnTarget = new Set<string>();
    const removedIds = new Set<string>();
    // Every special sitting on the rebuilt board, with the cells it is currently standing on
    // They stay put until the fill lifts them one at a time, so the board is never in a state where one is missing
    const specialsOnBoard: { piece: InventoryItem; cells: number[] }[] = [];
    const freeCells: number[] = [];
    // Whether a shape fits is a property of the shape and the free cells, never of the individual module, and filling a board only ever removes free cells
    // So once one module of a shape finds nowhere to go, every later module of that shape in the same pass finds nowhere either, and can be skipped without scanning
    const infeasibleShapes = new Set<ModuleShape>();
    // A board is empty exactly when every cell it has is free, and which cells it has is fixed by its tier
    let openCellCount = 0;
    for (let y = 0; y < 5; y++) for (let x = 0; x < 7; x++) if (currentBoard[y][x] !== 'Locked') openCellCount++;

    // Drops the cells that are no longer free, keeping the rest in scan order
    const compactFreeCells = (fillBoard: Board) => {
        let write = 0;
        for (let c = 0; c < freeCells.length; c++) {
            const idx = freeCells[c];
            const cx = idx % 7;
            const cy = (idx - cx) / 7;
            if (fillBoard[cy][cx] === null) freeCells[write++] = idx;
        }
        freeCells.length = write;
    };

    /* Commits one piece at its best-scoring placement among the free cells, and reports whether it found one
     * The pool fill and the special-module relocation both go through here on purpose:
     * a special is only allowed to move because the fill can judge where it should go, and it has to judge it on exactly the terms it judges everything else
     *
     * `incumbent` is where the piece is standing already, for a piece being relocated rather than placed for the first time
     * It is scored ahead of everything else and the scan only takes a strictly better cell, so a piece with nowhere better to be simply stays
     * Without it the ties decide, and for a module whose score barely varies across the board (which is every special) that means the first cell in scan order:
     * a Line4 lands in the top-left of whatever the ruin opened up, every iteration, taking the best space on the board from the modules that would have earned something with it
     */
    const placeBestFit = (
        ctx: PlacementContext,
        orientations: Orientation[],
        piece: InventoryItem,
        fillBoard: Board,
        fillBoardEmpty: boolean,
        currentP: number, currentQ: number, currentE: number,
        incumbent: { x: number; y: number; orientation: Orientation } | null = null
    ) => {
        let bestX = -1, bestY = -1;
        let bestOrientation: Orientation | null = null;
        let highestHeuristic = incumbent !== null ? -Infinity : 0.0001;

        if (incumbent !== null) {
            const incumbentScore = evaluatePlacementDelta(
                ctx, incumbent.x, incumbent.y, incumbent.orientation, fillBoard, fillBoardEmpty,
                precomputedInternal, dynWp, dynWq, dynWe, currentP, currentQ, currentE, machine
            );
            if (incumbentScore !== -Infinity) {
                highestHeuristic = incumbentScore;
                bestX = incumbent.x; bestY = incumbent.y;
                bestOrientation = incumbent.orientation;
            }
        }

        for (let c = 0; c < freeCells.length; c++) {
            const idx = freeCells[c];
            const x = idx % 7;
            const y = (idx - x) / 7;

            for (let o = 0; o < orientations.length; o++) {
                const orientation = orientations[o];
                // evaluatePlacementDelta rejects these too, but most anchors on a 7x5 board are out of bounds for a given orientation,
                // and the bounding box settles it here without the nine-argument call
                if (x + orientation.minX < 0 || x + orientation.maxX > 6 ||
                    y + orientation.minY < 0 || y + orientation.maxY > 4) continue;

                const deltaScore = evaluatePlacementDelta(
                    ctx, x, y, orientation, fillBoard, fillBoardEmpty,
                    precomputedInternal, dynWp, dynWq, dynWe, currentP, currentQ, currentE, machine
                );
                if (deltaScore > highestHeuristic && deltaScore !== -Infinity) {
                    highestHeuristic = deltaScore;
                    bestX = x; bestY = y;
                    bestOrientation = orientation;
                }
            }
        }

        if (!bestOrientation) return false;

        const { xs, ys, count } = bestOrientation;
        for (let i = 0; i < count; i++) {
            fillBoard[bestY + ys[i]][bestX + xs[i]] = piece;
        }
        compactFreeCells(fillBoard);
        return true;
    };

    let pendingUpdate = false;
    const flushUpdate = () => {
        if (!pendingUpdate) return;
        pendingUpdate = false;

        if (codeIsStale) {
            currentCode = generateCodeFromState(
                machine.tier, machine.maximizeStats, machine.targetStats,
                inventoryForCode(fullInventory, bestBoard), bestBoard
            );
            codeIsStale = false;
        }
        onUpdate({
            board: copyBoard(bestBoard),
            totals: bestStats.totals,
            pieceStats: bestStats.pieceStats,
            code: currentCode
        });
    };

    const { portYield, timerYield, dispose } = createYielder();
    let lastYield = now();
    let lastTimerYield = lastYield;
    let iterations = 0;

    try {
        while (control.running && iterations < maxIterations) {
            iterations++;
            const isStagnant = stagnationCounter >= STAGNATION_LIMIT;

            for (let y = 0; y < 5; y++) {
                const srcRow = currentBoard[y];
                const dstRow = testBoard[y];
                for (let x = 0; x < 7; x++) dstRow[x] = srcRow[x];
            }

            markGen++;

            piecesOnTarget.length = 0;
            seenOnTarget.clear();
            specialsOnBoard.length = 0;

            for (let y = 0; y < 5; y++) {
                for (let x = 0; x < 7; x++) {
                    const cell = testBoard[y][x];
                    if (!cell || cell === 'Locked') continue;

                    /* A special is offered a better cell every iteration instead of taking a share of the ruin's removal budget
                     * It is not the ruin's kind of move: the ruin takes a piece away and lets the fill find something better to do with the space,
                     * and a special is going straight back down whatever happens. Spending a removal on one only means one fewer real piece is reconsidered that iteration
                     */
                    if (isSpecialModule(cell) || cell.isLocked) {
                        let entry = specialsOnBoard.find(e => e.piece.id === cell.id);
                        if (entry === undefined) {
                            entry = { piece: cell, cells: [] };
                            specialsOnBoard.push(entry);
                        }
                        entry.cells.push(y * 7 + x);
                        continue;
                    }

                    if (!seenOnTarget.has(cell.id)) {
                        seenOnTarget.add(cell.id);
                        piecesOnTarget.push(cell);
                    }
                }
            }

            if (piecesOnTarget.length > 0) {
                const removeCount = isStagnant
                    ? Math.max(1, Math.floor(piecesOnTarget.length * (0.5 + Math.random() * 0.4)))
                    : Math.floor(Math.random() * Math.min(3, piecesOnTarget.length)) + 1;

                shuffleInPlace(piecesOnTarget);
                removedIds.clear();
                for (let i = 0; i < removeCount; i++) removedIds.add(piecesOnTarget[i].id);

                for (let y = 0; y < 5; y++) {
                    for (let x = 0; x < 7; x++) {
                        const cell = testBoard[y][x];
                        if (cell && cell !== 'Locked') {
                            if (removedIds.has(cell.id)) {
                                testBoard[y][x] = null;
                            } else {
                                const pIdx = poolIndexOf.get(cell.id);
                                if (pIdx !== undefined) placedMark[pIdx] = markGen;
                            }
                        }
                    }
                }
            }

            // Every orientation is normalised so its first cell is the anchor, so only empty cells can anchor a placement
            // Tracking them prunes the scan as the board fills up
            freeCells.length = 0;
            for (let y = 0; y < 5; y++) {
                for (let x = 0; x < 7; x++) {
                    if (testBoard[y][x] === null) freeCells.push(y * 7 + x);
                }
            }
            shuffleInPlace(freeCells);
            let fillBoardEmpty = freeCells.length === openCellCount;

            let currentTotals = calculateBoardStats(testBoard, fullInventory, inventoryById).totals;

            /* The board's specials, offered a better cell one at a time and before anything is drawn
             *
             * One at a time is what makes this safe: at the moment a special is placed its own cells are still free, and an anchor-normalised orientation always has an
             * anchor among them, so placeBestFit can never come back empty-handed and a special can never be lost on the way. Lifting them all at once would let the
             * first one take the second one's cells and leave the second with nowhere guaranteed to go
             * Going before the draw also means they choose out of the whole ruined area rather than whatever the fill leaves over
             */
            for (const special of specialsOnBoard) {
                const ctx = placementContexts.get(special.piece.id);
                const orientations = PRECOMPUTED_ORIENTATIONS.get(special.piece.shape);
                if (ctx === undefined || orientations === undefined) continue;

                // The cells were collected in row-major order, and an orientation's offsets are in that same order and anchored on its first cell,
                // so the cells the piece is standing on say which orientation it is standing in
                const anchor = special.cells[0];
                const homeX = anchor % 7;
                const homeY = (anchor - homeX) / 7;
                let home: Orientation | null = null;
                for (const orientation of orientations) {
                    if (orientation.count !== special.cells.length) continue;
                    let matches = true;
                    for (let i = 0; i < orientation.count; i++) {
                        if (special.cells[i] !== anchor + orientation.ys[i] * 7 + orientation.xs[i]) { matches = false; break; }
                    }
                    if (matches) { home = orientation; break; }
                }

                for (const idx of special.cells) {
                    const cx = idx % 7;
                    testBoard[(idx - cx) / 7][cx] = null;
                    freeCells.push(idx);
                }
                placeBestFit(
                    ctx, orientations, special.piece, testBoard, fillBoardEmpty,
                    currentTotals.Performance, currentTotals.Quality, currentTotals.Efficiency,
                    home === null ? null : { x: homeX, y: homeY, orientation: home }
                );
                fillBoardEmpty = false;
                currentTotals = calculateBoardStats(testBoard, fullInventory, inventoryById).totals;
            }

            infeasibleShapes.clear();

            // Which of the machine's targets the accepted board already meets picks the draw table, so the fill stops being offered more of a stat it has enough of
            let metMask = 0;
            for (let i = 0; i < targetedKeys.length; i++) {
                const key = targetedKeys[i];
                if (currentStats.totals[key] >= machine.targetStats[key]!) {
                    metMask |= 1 << i;
                }
            }
            const values = drawValues[metMask];

            // Stops once every shape in the pool has been shown to fit nowhere, which cannot change while the board is only losing free cells
            let drawn = 0;
            while (drawn < poolOrder.length && infeasibleShapes.size < poolShapeCount) {
                // Best of DRAW_TOURNAMENT random candidates rather than the first one drawn
                // The permutation is still only partially shuffled, so the losers stay in the undrawn region and can be picked again later this fill
                const remaining = poolOrder.length - drawn;
                let swapAt = drawn + Math.floor(Math.random() * remaining);
                for (let t = 1; t < DRAW_TOURNAMENT && t < remaining; t++) {
                    const alt = drawn + Math.floor(Math.random() * remaining);
                    if (values[poolOrder[alt]] > values[poolOrder[swapAt]]) swapAt = alt;
                }
                const pieceIdx = poolOrder[swapAt];
                poolOrder[swapAt] = poolOrder[drawn];
                poolOrder[drawn] = pieceIdx;
                drawn++;

                if (placedMark[pieceIdx] === markGen) continue;

                const piece = searchPool[pieceIdx];
                if (infeasibleShapes.has(piece.shape)) continue;

                const ctx = ctxByIndex[pieceIdx];
                // Too few cells left for this shape is itself a permanent verdict on it
                if (freeCells.length < ctx.size) {
                    infeasibleShapes.add(piece.shape);
                    continue;
                }

                const orientations = orientationsByIndex[pieceIdx];
                if (!orientations) continue;

                if (placeBestFit(ctx, orientations, piece, testBoard, fillBoardEmpty, currentTotals.Performance, currentTotals.Quality, currentTotals.Efficiency)) {
                    fillBoardEmpty = false;
                    currentTotals = calculateBoardStats(testBoard, fullInventory, inventoryById).totals;
                } else {
                    infeasibleShapes.add(piece.shape);
                }
            }

            const rebuiltStats = calculateBoardStats(testBoard, fullInventory, inventoryById);

            currentTiers.fill(0);
            const t = rebuiltStats.totals;
            for (const key of STAT_KEYS) {
                if (statIsIgnored(machine, key)) continue;

                const ti = tierOf(key);
                const target = machine.targetStats[key];
                if (target !== null && t[key] < target) currentTiers[ti] -= (target - t[key]) * 10000;
                if (machine.maximizeStats[key]) currentTiers[ti] += (t[key] * 10);
            }

            // Density reward: a general tiebreak, so it sits in the least important tier and can never take a ranked stat out of its own tier
            currentTiers[tierCount] -= rebuiltStats.placedPiecesCount * 5;

            if (!control.running) break;

            // Judged against the best of THIS attempt, not the best ever
            // After a restart the board is deliberately worse than the record, and comparing it to the record would reject every move and leave the restart unable to climb at all
            const ordering = compareTiers(currentTiers, epochTiers);
            const improved = ordering > 0;
            if (improved || (ordering === 0 && Math.random() > 0.5)) {
                if (improved) epochTiers.set(currentTiers);
                currentBoard = copyBoard(testBoard);
                currentStats = rebuiltStats;

                // A new record is the only thing worth reporting, and the only thing that resets the stagnation count
                if (improved && compareTiers(currentTiers, bestTiers) > 0) {
                    bestTiers.set(currentTiers);
                    bestBoard = copyBoard(currentBoard);
                    bestStats = currentStats;
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
                    epochTiers.fill(-Infinity);
                    currentBoard = copyBoard(initialBoard);
                    currentStats = calculateBoardStats(currentBoard, fullInventory, inventoryById);
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
