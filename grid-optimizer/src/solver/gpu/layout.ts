import { d } from 'typegpu';
import { BOARD_CELLS, MAX_PIECE_CELLS, MAX_PIECE_NEIGHBORS, ORIENT_TOTAL, PLACE_ENTRIES, SCAN_STRIDES, SHAPE_COUNT } from '../geometry';
import { TIER_VECTOR_LENGTH } from '../objective';

export const WORKGROUP_SIZE = 64;
export const DEFAULT_THREADS = 4096;

// The per-thread blocked bitset is a fixed number of words, which caps how many modules one solve may know about
export const BLOCKED_WORDS = 32;
export const MAX_ITEMS = BLOCKED_WORDS * 32;

// A thread that has no record yet reports this in the first tier, below any objective the search can produce
export const NO_RECORD = -2000000000;

// The placement tables laid end to end in one buffer
export const GEO_META = 0;
export const GEO_CELL_COUNT = GEO_META + PLACE_ENTRIES;
export const GEO_CELLS = GEO_CELL_COUNT + PLACE_ENTRIES;
export const GEO_NBR_COUNT = GEO_CELLS + PLACE_ENTRIES * MAX_PIECE_CELLS;
export const GEO_NBRS = GEO_NBR_COUNT + PLACE_ENTRIES;
export const GEO_MASK_LO = GEO_NBRS + PLACE_ENTRIES * MAX_PIECE_NEIGHBORS;
export const GEO_MASK_HI = GEO_MASK_LO + PLACE_ENTRIES;
export const GEO_ORIENT_START = GEO_MASK_HI + PLACE_ENTRIES;
export const GEO_ORIENT_COUNT = GEO_ORIENT_START + SHAPE_COUNT;
export const GEO_ORIENT_CELL_COUNT = GEO_ORIENT_COUNT + SHAPE_COUNT;
export const GEO_ORIENT_OFFSETS = GEO_ORIENT_CELL_COUNT + ORIENT_TOTAL;
export const GEO_ORIENT_CORNERS_LO = GEO_ORIENT_OFFSETS + ORIENT_TOTAL * MAX_PIECE_CELLS;
export const GEO_ORIENT_CORNERS_HI = GEO_ORIENT_CORNERS_LO + ORIENT_TOTAL;
export const GEO_SCAN_STRIDES = GEO_ORIENT_CORNERS_HI + ORIENT_TOTAL;
export const GEO_LENGTH = GEO_SCAN_STRIDES + SCAN_STRIDES.length;

// Twelve words, so the uniform array stride stays a multiple of sixteen bytes
export const PoolEntry = d.struct({
    p: d.i32, q: d.i32, e: d.i32,
    p20: d.i32, q20: d.i32, e20: d.i32,
    flags: d.i32, shape: d.i32, drawable: d.i32,
    orientStart: d.i32, orientCount: d.i32, recvSlot: d.i32
});

export const ThreadState = d.struct({
    rngCtr: d.u32,
    rngInc: d.u32,
    stagnation: d.i32,
    stagnations: d.i32,
    hasEpoch: d.i32,
    hasRecord: d.i32,
    curP: d.i32, curQ: d.i32, curE: d.i32, curPieces: d.i32,
    epochTiers: d.arrayOf(d.i32, TIER_VECTOR_LENGTH),
    bestTiers: d.arrayOf(d.i32, TIER_VECTOR_LENGTH),
    cur: d.arrayOf(d.i32, BOARD_CELLS),
    best: d.arrayOf(d.i32, BOARD_CELLS)
});

// The aux buffer holds the receiver table, the draw list, the draw ranks, the initial board and the per-stat table, each at an offset given here
export const STAT_FIELDS = 6;
export const STAT_TARGET = 0;
export const STAT_HAS_TARGET = 1;
export const STAT_MAXIMIZE = 2;
export const STAT_WEIGHT = 3;
export const STAT_TIER_OF = 4;
// Position of each stat among the targeted ones, which is its bit in the met-target mask; -1 when it has no target
export const STAT_TARGETED_INDEX = 5;

export const Params = d.struct({
    threadCount: d.i32,
    itersPerDispatch: d.i32,
    drawCount: d.i32,
    needsTotals: d.i32,
    openCellCount: d.i32,
    tierCount: d.i32,
    stagnationLimit: d.i32,
    restartAfter: d.i32,
    recvOffset: d.i32,
    drawListOffset: d.i32,
    shapeStartOffset: d.i32,
    drawRankOffset: d.i32,
    initialBoardOffset: d.i32,
    statOffset: d.i32,
    championIdx: d.i32
});

export type ParamsValue = d.Infer<typeof Params>;
export type ThreadStateValue = d.Infer<typeof ThreadState>;
export type PoolEntryValue = d.Infer<typeof PoolEntry>;
