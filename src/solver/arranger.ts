import { arrangeBoard } from './arrange';
import { type BoardTotals, boardTotals } from './boardTotals';
import { BOARD_CELLS } from './geometry';
import type { UpdateHandler } from './handle';
import { toIndexBoard } from './indexBoard';
import { addMachineTiers, objectiveTiers, TIER_VECTOR_LENGTH } from './objective';
import { buildUpdate, type SolveUpdate } from './report';
import { boardOfSet, prepareSolve, type SolveRequest, type SolveSetup } from './setup';
import { createFramePacer, now } from './yielder';

// How many placements the arrangement of one record may try, shared out evenly over its boards
// A board of ten large pieces has ten thousand to a million layouts and is settled well inside this; one of many small pieces is not, and gets a few seconds
export const ARRANGE_STEP_LIMIT = 20_000_000;
// Placements tried between looks at the clock
const ARRANGE_SLICE_STEPS = 4096;
// How long a stopped solve waits for the arranger, shared out evenly over the boards; most of what an arrangement gains turns up in its first two million placements
export const ARRANGE_FINISH_MS = 300;

/* Runs beside a solve and rearranges the pieces of the records the search reports, on any backend
 * Only the latest record is worked on: a newer one replaces it, since the search has moved on
 * A better arrangement is reported like any other record, so the host shows whichever of the search and the arranger is ahead
 *
 * With a thread to itself the arranger is `live` and takes each record as it comes
 * Sharing the search's thread it would take half of it, most of that spent on records about to be beaten, so there it waits and arranges the last record once the search is over
 */
export interface Arranger {
    offer: (update: SolveUpdate) => void;
    // Gives the last record ARRANGE_FINISH_MS more, and resolves once the arranger has reported what it found
    finish: () => Promise<void>;
}

const setTiers = (setup: SolveSetup, set: Int32Array) => {
    const totals: BoardTotals = { p: 0, q: 0, e: 0, pieces: 0 };
    const tiers = new Int32Array(TIER_VECTOR_LENGTH);
    const sum = new Int32Array(TIER_VECTOR_LENGTH);
    setup.machines.forEach(({ plan, params }, k) => {
        boardTotals(setup.tables, boardOfSet(set, k), totals);
        objectiveTiers(totals, plan, params, tiers);
        addMachineTiers(sum, tiers, plan.tierCount, setup.densityIndex);
    });
    return sum;
};

export const runArranger = (request: SolveRequest, onUpdate: UpdateHandler, live: boolean): Arranger => {
    const setup = prepareSolve(request);
    const machineCount = setup.machines.length;
    const stepLimit = Math.floor(ARRANGE_STEP_LIMIT / machineCount);
    const pacer = createFramePacer();

    let pending: SolveUpdate | null = null;
    let finishAt = Infinity;
    let wake = () => {};
    const finishing = () => finishAt !== Infinity;
    const timeUp = (k: number) => now() >= finishAt + ARRANGE_FINISH_MS * (k + 1) / machineCount;

    // Boards are arranged one after another, each improvement reported within a frame of being found; a newer record cuts the rest short
    const arrange = async (record: SolveUpdate) => {
        const set = new Int32Array(BOARD_CELLS * machineCount);
        record.boards.forEach(({ board }, k) => boardOfSet(set, k).set(toIndexBoard(board, setup.tables.indexOf)));

        for (let k = 0; k < machineCount; k++) {
            const search = arrangeBoard(setup.tables, setup.machines[k], boardOfSet(set, k), stepLimit);
            let reported = 0;
            const report = () => {
                if (search.improvements() === reported) return;
                reported = search.improvements();
                boardOfSet(set, k).set(search.best);
                onUpdate(buildUpdate(request, setup, set, setTiers(setup, set)));
            };

            while (!search.advance(ARRANGE_SLICE_STEPS)) {
                if (!pacer.due()) continue;
                report();
                if (timeUp(k)) break;
                await pacer.pause();
                if (pending !== null) return;
            }
            report();
        }
    };

    const work = async () => {
        try {
            while (pending !== null || !finishing()) {
                if (pending === null || !(live || finishing())) {
                    await new Promise<void>(resolve => { wake = resolve; });
                    continue;
                }
                const record = pending;
                pending = null;
                await arrange(record);
            }
        } finally {
            pacer.dispose();
        }
    };
    const done = work().catch(error => console.warn('Arranger failed, keeping what the search found', error));

    return {
        offer: update => { pending = update; wake(); },
        finish: () => { finishAt = Math.min(finishAt, now()); wake(); return done; }
    };
};
