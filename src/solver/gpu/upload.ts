import { RESTART_AFTER_STAGNATIONS, STAGNATION_LIMIT } from '../engine';
import {
    ORIENT_CELL_COUNT, ORIENT_CORNERS_HI, ORIENT_CORNERS_LO, ORIENT_COUNT, ORIENT_OFFSETS, ORIENT_START, PLACE_CELL_COUNT, PLACE_CELLS, PLACE_MASK_HI,
    PLACE_MASK_LO, PLACE_META, PLACE_NBR_COUNT, PLACE_NBRS, SCAN_STRIDES
} from '../geometry';
import { TIER_VECTOR_LENGTH } from '../objective';
import { RNG_CTR, RNG_INC, seedRng } from '../rng';
import type { MachineSetup, SolveSetup } from '../setup';
import { BOARD_CELLS } from '../geometry';
import {
    GEO_CELL_COUNT, GEO_CELLS, GEO_LENGTH, GEO_MASK_HI, GEO_MASK_LO, GEO_META, GEO_NBR_COUNT, GEO_NBRS, GEO_ORIENT_CELL_COUNT, GEO_ORIENT_CORNERS_HI,
    GEO_ORIENT_CORNERS_LO, GEO_ORIENT_COUNT, GEO_ORIENT_OFFSETS, GEO_ORIENT_START, GEO_SCAN_STRIDES,
    M_DRAW_COUNT, M_DRAW_LIST_OFFSET, M_DRAW_RANK_OFFSET, M_DRAWABLE_OFFSET, M_INITIAL_BOARD_OFFSET, M_NEEDS_TOTALS, M_OPEN_CELL_COUNT, M_SHAPE_START_OFFSET,
    M_STAT_OFFSET, M_TIER_COUNT, MACHINE_FIELDS, MAX_ITEMS, MAX_MACHINES, type ParamsValue, type PoolEntryValue,
    STAT_FIELDS, STAT_HAS_TARGET, STAT_MAXIMIZE, STAT_TARGET, STAT_TARGETED_INDEX, STAT_TIER_OF, STAT_WEIGHT, type ThreadStateValue
} from './layout';

// Everything a solve uploads once: the flat tables the kernel reads and the parameters that never change during the solve
export interface GpuTables {
    pool: PoolEntryValue[];
    geometry: Int32Array;
    aux: Int32Array;
    params: ParamsValue;
}

export const fitsGpu = (setup: SolveSetup) =>
    setup.machines.length <= MAX_MACHINES && setup.tables.count <= MAX_ITEMS && setup.machines.some(m => m.draw.drawList.length > 0);

const buildPool = (setup: SolveSetup): PoolEntryValue[] => {
    const t = setup.tables;
    return Array.from({ length: t.count }, (_, i) => ({
        p: t.p[i], q: t.q[i], e: t.e[i], p20: t.p20[i], q20: t.q20[i], e20: t.e20[i],
        flags: t.flags[i], shape: t.shape[i], pad: 0,
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

const buildStatTable = ({ params, plan, targeted }: MachineSetup) => {
    const stat = new Int32Array(STAT_FIELDS * 3);
    for (let s = 0; s < 3; s++) {
        stat[STAT_TARGET * 3 + s] = params.target[s];
        stat[STAT_HAS_TARGET * 3 + s] = params.hasTarget[s];
        stat[STAT_MAXIMIZE * 3 + s] = params.maximize[s];
        stat[STAT_WEIGHT * 3 + s] = params.w[s];
        stat[STAT_TIER_OF * 3 + s] = plan.tierOf[s];
        stat[STAT_TARGETED_INDEX * 3 + s] = targeted.indexOf(s);
    }
    return stat;
};

export const buildGpuTables = (setup: SolveSetup): GpuTables => {
    const { tables, initialSet, machines } = setup;
    const sections: Int32Array[] = [];
    let at = 0;
    const place = (section: Int32Array) => {
        sections.push(section);
        at += section.length;
        return at - section.length;
    };

    const recvOffset = place(tables.recvTable);
    const initialSetOffset = place(initialSet);
    const machineTable = new Int32Array(MACHINE_FIELDS * machines.length);
    machines.forEach((machine, k) => {
        const fields = machineTable.subarray(k * MACHINE_FIELDS, (k + 1) * MACHINE_FIELDS);
        fields[M_OPEN_CELL_COUNT] = machine.openCellCount;
        fields[M_TIER_COUNT] = machine.plan.tierCount;
        fields[M_NEEDS_TOTALS] = machine.needsTotals ? 1 : 0;
        fields[M_DRAW_COUNT] = machine.draw.drawList.length;
        fields[M_DRAW_LIST_OFFSET] = place(machine.draw.drawList);
        fields[M_SHAPE_START_OFFSET] = place(machine.draw.shapeStart);
        fields[M_DRAW_RANK_OFFSET] = at;
        machine.drawRanks.forEach(place);
        fields[M_INITIAL_BOARD_OFFSET] = initialSetOffset + k * BOARD_CELLS;
        fields[M_STAT_OFFSET] = place(buildStatTable(machine));
        fields[M_DRAWABLE_OFFSET] = place(Int32Array.from(machine.draw.drawable));
    });
    const machineOffset = place(machineTable);

    const aux = new Int32Array(at);
    let write = 0;
    for (const section of sections) {
        aux.set(section, write);
        write += section.length;
    }

    const params: ParamsValue = {
        threadCount: 0,
        itersPerDispatch: 1,
        machineCount: machines.length,
        machineOffset,
        densityIndex: setup.densityIndex,
        stagnationLimit: STAGNATION_LIMIT * machines.length,
        restartAfter: RESTART_AFTER_STAGNATIONS,
        recvOffset,
        championIdx: 0,
        migrateBelow0: 0, migrateBelow1: 0, migrateBelow2: 0, migrateBelow3: 0
    };
    return { pool: buildPool(setup), geometry: buildGeometry(), aux, params };
};

// Every thread starts on the initial set with its own stream of the shared seed; totals are computed by the kernel on its first iteration
export const buildInitialStates = (setup: SolveSetup, seed: number, threads: number): ThreadStateValue[] => {
    const machineCount = setup.machines.length;
    const perBoard = (value: number) => Array.from({ length: machineCount }, () => value);
    return Array.from({ length: threads }, (_, thread) => {
        const rng = seedRng(seed, thread);
        return {
            rngCtr: rng[RNG_CTR], rngInc: rng[RNG_INC],
            stagnation: 0, stagnations: 0, restarts: 0, hasEpoch: 0, hasRecord: 0,
            curP: perBoard(0), curQ: perBoard(0), curE: perBoard(0), curPieces: perBoard(-1),
            boardTiers: Array.from({ length: TIER_VECTOR_LENGTH * machineCount }, () => 0),
            epochTiers: [0, 0, 0, 0], bestTiers: [0, 0, 0, 0],
            cur: Array.from(setup.initialSet), best: Array.from(setup.initialSet)
        };
    });
};
