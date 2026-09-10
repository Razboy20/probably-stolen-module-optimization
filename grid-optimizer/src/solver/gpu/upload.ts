import { RESTART_AFTER_STAGNATIONS, STAGNATION_LIMIT } from '../engine';
import {
    ORIENT_CELL_COUNT, ORIENT_CORNERS_HI, ORIENT_CORNERS_LO, ORIENT_COUNT, ORIENT_OFFSETS, ORIENT_START, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_MASK_HI,
    PLACE_MASK_LO, PLACE_META, PLACE_NBR_COUNT, PLACE_NBRS, SCAN_STRIDES
} from '../geometry';
import { RNG_CTR, RNG_INC, seedRng } from '../rng';
import type { SolveSetup } from '../setup';
import {
    GEO_CELL_COUNT, GEO_CELLS, GEO_LENGTH, GEO_MASK_HI, GEO_MASK_LO, GEO_META, GEO_NBR_COUNT, GEO_NBRS, GEO_ORIENT_CELL_COUNT, GEO_ORIENT_CORNERS_HI,
    GEO_ORIENT_CORNERS_LO, GEO_ORIENT_COUNT, GEO_ORIENT_OFFSETS, GEO_ORIENT_START, GEO_SCAN_STRIDES,
    MAX_ITEMS, type ParamsValue, type PoolEntryValue,
    STAT_FIELDS, STAT_HAS_TARGET, STAT_MAXIMIZE, STAT_TARGET, STAT_TARGETED_INDEX, STAT_TIER_OF, STAT_WEIGHT, type ThreadStateValue
} from './layout';

// Everything a solve uploads once: the flat tables the kernel reads and the parameters that never change during the solve
export interface GpuTables {
    pool: PoolEntryValue[];
    geometry: Int32Array;
    aux: Int32Array;
    params: ParamsValue;
}

export const fitsGpu = (setup: SolveSetup) => setup.tables.count <= MAX_ITEMS && setup.draw.drawList.length > 0;

const buildPool = (setup: SolveSetup): PoolEntryValue[] => {
    const t = setup.tables;
    return Array.from({ length: t.count }, (_, i) => ({
        p: t.p[i], q: t.q[i], e: t.e[i], p20: t.p20[i], q20: t.q20[i], e20: t.e20[i],
        flags: t.flags[i], shape: t.shape[i], drawable: setup.draw.drawable[i],
        orientStart: t.orientStart[i], orientCount: t.orientCount[i], recvSlot: t.recvSlot[i]
    }));
};

const buildGeometry = () => {
    const geo = new Int32Array(GEO_LENGTH);
    geo.set(PLACE_META, GEO_META);
    geo.set(PLACE_CELL_COUNT, GEO_CELL_COUNT);
    geo.set(PLACE_CELLS, GEO_CELLS);
    geo.set(PLACE_NBR_COUNT, GEO_NBR_COUNT);
    geo.set(PLACE_NBRS, GEO_NBRS);
    geo.set(PLACE_MASK_LO, GEO_MASK_LO);
    geo.set(PLACE_MASK_HI, GEO_MASK_HI);
    geo.set(ORIENT_START, GEO_ORIENT_START);
    geo.set(ORIENT_COUNT, GEO_ORIENT_COUNT);
    geo.set(ORIENT_CELL_COUNT, GEO_ORIENT_CELL_COUNT);
    geo.set(ORIENT_OFFSETS, GEO_ORIENT_OFFSETS);
    geo.set(ORIENT_CORNERS_LO, GEO_ORIENT_CORNERS_LO);
    geo.set(ORIENT_CORNERS_HI, GEO_ORIENT_CORNERS_HI);
    geo.set(SCAN_STRIDES, GEO_SCAN_STRIDES);
    return geo;
};

const buildStatTable = (setup: SolveSetup) => {
    const stat = new Int32Array(STAT_FIELDS * 3);
    for (let s = 0; s < 3; s++) {
        stat[STAT_TARGET * 3 + s] = setup.params.target[s];
        stat[STAT_HAS_TARGET * 3 + s] = setup.params.hasTarget[s];
        stat[STAT_MAXIMIZE * 3 + s] = setup.params.maximize[s];
        stat[STAT_WEIGHT * 3 + s] = setup.params.w[s];
        stat[STAT_TIER_OF * 3 + s] = setup.plan.tierOf[s];
        stat[STAT_TARGETED_INDEX * 3 + s] = setup.targeted.indexOf(s);
    }
    return stat;
};

export const buildGpuTables = (setup: SolveSetup): GpuTables => {
    const { tables, draw, drawRanks, initialIndexBoard } = setup;
    const sections = [tables.recvTable, draw.drawList, draw.shapeStart, ...drawRanks, initialIndexBoard, buildStatTable(setup)];
    const aux = new Int32Array(sections.reduce((n, s) => n + s.length, 0));
    const offsets: number[] = [];
    let at = 0;
    for (const section of sections) {
        offsets.push(at);
        aux.set(section, at);
        at += section.length;
    }

    const params: ParamsValue = {
        threadCount: 0,
        itersPerDispatch: 1,
        drawCount: draw.drawList.length,
        needsTotals: setup.needsTotals ? 1 : 0,
        openCellCount: setup.openCellCount,
        tierCount: setup.plan.tierCount,
        stagnationLimit: STAGNATION_LIMIT,
        restartAfter: RESTART_AFTER_STAGNATIONS,
        recvOffset: offsets[0],
        drawListOffset: offsets[1],
        shapeStartOffset: offsets[2],
        drawRankOffset: offsets[3],
        initialBoardOffset: offsets[3 + drawRanks.length],
        statOffset: offsets[4 + drawRanks.length],
        championIdx: 0,
        migrateBelow0: 0, migrateBelow1: 0, migrateBelow2: 0, migrateBelow3: 0
    };
    return { pool: buildPool(setup), geometry: buildGeometry(), aux, params };
};

// Every thread starts on the initial board with its own stream of the shared seed; totals are computed by the kernel on its first iteration
export const buildInitialStates = (setup: SolveSetup, seed: number, threads: number): ThreadStateValue[] =>
    Array.from({ length: threads }, (_, thread) => {
        const rng = seedRng(seed, thread);
        return {
            rngCtr: rng[RNG_CTR], rngInc: rng[RNG_INC],
            stagnation: 0, stagnations: 0, restarts: 0, hasEpoch: 0, hasRecord: 0,
            curP: 0, curQ: 0, curE: 0, curPieces: -1,
            epochTiers: [0, 0, 0, 0], bestTiers: [0, 0, 0, 0],
            cur: Array.from(setup.initialIndexBoard), best: Array.from(setup.initialIndexBoard)
        };
    });
