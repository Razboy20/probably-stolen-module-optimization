import { expect, test } from 'bun:test';
import { arrangeBoard } from '../src/solver/arrange';
import { ARRANGE_FINISH_MS, runArranger } from '../src/solver/arranger';
import type { Board } from '../src/solver/board';
import { calculateBoardStats } from '../src/solver/boardStats';
import { runSolver } from '../src/solver/client';
import { decodeSolution } from '../src/solver/codec';
import { runOptimizationEngine } from '../src/solver/engine';
import { BOARD_CELLS, MAX_PIECE_CELLS, ORIENT_CELL_COUNT, ORIENT_OFFSETS, ORIENT_TOTAL } from '../src/solver/geometry';
import { fromIndexBoard, type IndexBoard } from '../src/solver/indexBoard';
import type { MachineConfig } from '../src/solver/objective';
import { buildUpdate, type SolveUpdate } from '../src/solver/report';
import { prepareSolve, type SolveRequest } from '../src/solver/setup';
import { buildBenchCase } from './benchCase';

/* A Hydroponic System held to Performance 150 with Quality maximized, its ten pieces filling the board
 * They tile it 12,560 ways and two of those meet the target: this one at Quality 74 and its left-right mirror at Quality 80,
 * where the Side Mount Refinement reaches the left column. No piece keeps its cells between the two
 */
const CODE = '}A^?Vk}dEPw1IITs{ccUx/H<cTJWa+cni{yt7U[=3CKQ/z)V42khI5?%*x3N*vRe9)5Yy(fZqC@MA20yM3zG?:*ZEC<1Up/#/1FF?!+UEK{^sf0p6-gVBV+Zn>^c/y/49L%06rA.f{k0%p>?I*{$HPVqJc3lhCwt*?NQMFE*XRxb9.9$#B8SS]4?WvxHYM]AuX8BvvC=R<:fk*weGSu5u3RPd1*r/jz6RQ!=+N1[^D-:M^3J%FdeuW<g6i6*2LD9Aheyd/2M5#^53p$^>W%W0E%m!u^#+xQNWUCc<QFltYcqn&XmT)KIbWn3d3UN1Z$j$ailqlUS^braLUMt]d{A5>@MlynvQfitPb9/W11bG52:%s^hf*VLqY4e{#+eEi]-)mA(hyJT(Ocij!&{xwG>uM$QLQ#T8W#8V4Wz-W:<CU@Fq(IFyM1IrkoQ23g)oqZ}@.$P3+3gP6IYsW[!d[GCGCZeJV?H4g91#-KxSll=Uc9zG!G!=vW1<xe=eer27&c$MbNB^K?}.]3?ef)zjY@2Um-1QT1Oka{3@fB?tT!yx&A1@AOUA95z3F8/&+fz/cr%GdKVP1fzrJQ+&^WK{QCtk%6>-!4>2h<SyYmJ:';
const LOOM = 'Kotton Auto Loom Hydroponic Module';
const STEP_LIMIT = 20_000_000;

// takenOff removes that many pieces from the board, the first ones met reading it row by row, so their cells are empty
const mirrorCase = (stuckLoom = false, takenOff = 0) => {
    const decoded = decodeSolution(CODE);
    if (stuckLoom) decoded.inventory.find(item => item.displayName === LOOM)!.isStuck = true;
    const onBoard = [...new Set(decoded.board.flat().flatMap(cell => cell && cell !== 'Locked' ? [cell.id] : []))];
    const gone = new Set(onBoard.slice(0, takenOff));
    const initialBoard: Board = decoded.board.map(row => row.map(cell => cell && cell !== 'Locked' && gone.has(cell.id) ? null : cell));
    const machine: MachineConfig = { id: 'm', tier: 3, kind: 'Hydroponic System', targetStats: decoded.targetStats, maximizeStats: decoded.maximizeStats };
    const request: SolveRequest = { machines: [{ machine, initialBoard }], searchPoolInventory: decoded.inventory, fullInventory: decoded.inventory };
    return { request, setup: prepareSolve(request) };
};

const arrangeFully = (setup: ReturnType<typeof prepareSolve>) => {
    const search = arrangeBoard(setup.tables, setup.machines[0], setup.initialSet, STEP_LIMIT);
    while (!search.advance(100_000));
    return search;
};

const cellsPerItem = (board: IndexBoard) => {
    const counts = new Map<number, number>();
    for (const item of board) if (item >= 0) counts.set(item, (counts.get(item) ?? 0) + 1);
    return [...counts].sort((a, b) => a[0] - b[0]);
};

const totalsOf = (board: Board, request: SolveRequest) => calculateBoardStats(board, request.fullInventory).totals;

test('finds the mirrored layout, which no piece shares a cell with', () => {
    const { request, setup } = mirrorCase();
    const search = arrangeFully(setup);

    expect(search.exhausted()).toBe(true);
    expect(search.improvements()).toBe(1);
    expect(totalsOf(fromIndexBoard(search.best, setup.tables.items), request)).toEqual({ Performance: 155, Quality: 80, Efficiency: -426 });
    expect(cellsPerItem(search.best)).toEqual(cellsPerItem(setup.initialSet));
});

// The counts are those of a plain search that tries every piece on every cell in turn, so nothing is lost to the dead ends that are skipped
test('visits every layout of a full board', () => {
    expect(arrangeFully(mirrorCase().setup).layouts()).toBe(12_560);
});

test('visits every layout of a board with empty cells', () => {
    const { request, setup } = mirrorCase(false, 6);
    const search = arrangeFully(setup);

    expect(search.exhausted()).toBe(true);
    expect(search.layouts()).toBe(193_984);
    expect(totalsOf(fromIndexBoard(search.best, setup.tables.items), request)).toEqual({ Performance: -16, Quality: 30, Efficiency: -170 });
    expect(cellsPerItem(search.best)).toEqual(cellsPerItem(setup.initialSet));
});

test('leaves the board it was given untouched', () => {
    const { setup } = mirrorCase();
    const before = Int32Array.from(setup.initialSet);
    arrangeFully(setup);
    expect(setup.initialSet).toEqual(before);
});

test('stops at the step limit with the best arrangement so far', () => {
    const { setup } = mirrorCase();
    const search = arrangeBoard(setup.tables, setup.machines[0], setup.initialSet, 1000);
    while (!search.advance(300));
    expect(search.exhausted()).toBe(false);
    expect(search.improvements()).toBe(0);
    expect(search.best).toEqual(setup.initialSet);
});

test('never moves a stuck piece', () => {
    const { setup } = mirrorCase(true);
    const loom = setup.tables.items.findIndex(item => item.displayName === LOOM);
    const search = arrangeFully(setup);

    // With the loom held on the right the mirror is out of reach, and no other arrangement meets the target
    expect(search.exhausted()).toBe(true);
    expect(search.improvements()).toBe(0);
    for (let i = 0; i < BOARD_CELLS; i++) expect(search.best[i] === loom).toBe(setup.initialSet[i] === loom);
});

test('a placement anchored on a cell covers no cell before it in scan order', () => {
    for (let g = 0; g < ORIENT_TOTAL; g++) {
        for (let i = 1; i < ORIENT_CELL_COUNT[g]; i++) {
            expect(ORIENT_OFFSETS[g * MAX_PIECE_CELLS + i]).toBeGreaterThan(ORIENT_OFFSETS[g * MAX_PIECE_CELLS]);
        }
    }
});

const record = ({ request, setup }: ReturnType<typeof mirrorCase>) => buildUpdate(request, setup, setup.initialSet, Int32Array.of(740, -56, 0, 0));
const settle = () => new Promise(resolve => setTimeout(resolve, 1000));

test('a live arranger reports a better arrangement of a record as soon as it has one', async () => {
    const mirror = mirrorCase();
    const updates: SolveUpdate[] = [];
    const arranger = runArranger(mirror.request, update => updates.push(update), true);
    arranger.offer(record(mirror));
    await settle();

    expect(updates.length).toBe(1);
    expect(updates[0].boards[0].totals).toEqual({ Performance: 155, Quality: 80, Efficiency: -426 });
    expect(updates[0].tiers).toEqual([800, -55]);
    await arranger.finish();
    expect(updates.length).toBe(1);
});

test('a search handed a record it cannot reach takes it as its own', async () => {
    const mirror = mirrorCase();
    const mirrored = buildUpdate(mirror.request, mirror.setup, arrangeFully(mirror.setup).best, Int32Array.of(800, -55, 0, 0));
    // Every module off the board is out of the pool, as if the other machines held them, so the search has only this board's pieces to work with
    const onBoard = new Set(mirror.request.machines[0].initialBoard.flat().flatMap(cell => cell && cell !== 'Locked' ? [cell.id] : []));
    const searchPoolInventory = mirror.request.fullInventory.map(item => onBoard.has(item.id) ? item : { ...item, isLocked: true });
    const reportsOf = async (offered: SolveUpdate | null) => {
        const tiers: number[][] = [];
        const request = { ...mirror.request, searchPoolInventory, seed: 1, maxIterations: 20_000 };
        await runOptimizationEngine(request, { running: true, offered }, update => tiers.push(update.tiers));
        return tiers;
    };

    // Left to itself the search reports the board it was given and never finds the mirror; holding the mirror, it has nothing better to report
    expect(await reportsOf(null)).toEqual([[740, -56]]);
    expect(await reportsOf(mirrored)).toEqual([]);
});

test('an arranger that is not live holds the last record until it is told to finish', async () => {
    const mirror = mirrorCase();
    const updates: SolveUpdate[] = [];
    const arranger = runArranger(mirror.request, update => updates.push(update), false);
    arranger.offer(record(mirror));
    await settle();
    expect(updates.length).toBe(0);

    await arranger.finish();
    expect(updates.length).toBe(1);
    expect(updates[0].boards[0].totals.Quality).toBe(80);
});

test('a stopped solve waits for the arranger no longer than its finishing time', async () => {
    // A board whose arrangements run past the step limit, which takes seconds
    const { machines, inv } = buildBenchCase(2, 200, 0, 1);
    const request: SolveRequest = {
        machines: machines.map(({ machine, board }) => ({ machine, initialBoard: board })), searchPoolInventory: inv, fullInventory: inv, seed: 2
    };
    const solver = runSolver(request, () => {}, 'inline');
    await new Promise(resolve => setTimeout(resolve, 500));

    const stoppedAt = performance.now();
    solver.stop();
    await solver.done;
    expect(performance.now() - stoppedAt).toBeLessThan(ARRANGE_FINISH_MS + 200);
});

test('the arranger stays silent about a record it cannot improve', async () => {
    const stuck = mirrorCase(true);
    const updates: SolveUpdate[] = [];
    const arranger = runArranger(stuck.request, update => updates.push(update), true);
    arranger.offer(record(stuck));
    await arranger.finish();

    expect(updates.length).toBe(0);
});
