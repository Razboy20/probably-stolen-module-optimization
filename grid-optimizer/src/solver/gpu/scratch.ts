import tgpu, { d } from 'typegpu';
import { BOARD_CELLS, MAX_PIECE_CELLS, MAX_PIECE_NEIGHBORS, SHAPE_COUNT } from '../geometry';
import { TIER_VECTOR_LENGTH } from '../objective';
import { MAX_PIECES_PER_BOARD } from '../pool';
import { BLOCKED_WORDS, MAX_MACHINES } from './layout';

// Everything one thread keeps between the functions of an iteration: WGSL has no closures, so the kernel's working state lives in private variables
const i32s = (n: number) => tgpu.privateVar(d.arrayOf(d.i32, n), d.arrayOf(d.i32, n)());
const i32v = () => tgpu.privateVar(d.i32, 0);

// The accepted board set, the board of it being rebuilt this iteration, and the one board the fill works on
export const cur = i32s(BOARD_CELLS * MAX_MACHINES);
export const test = i32s(BOARD_CELLS);
export const board = i32v();
// What the kernel reads about the machine whose board is in test, rebound when another board is scored
export const mOpenCellCount = i32v(), mTierCount = i32v(), mNeedsTotals = i32v(), mDrawCount = i32v();
export const mDrawListOffset = i32v(), mShapeStartOffset = i32v(), mDrawRankOffset = i32v(), mStatOffset = i32v(), mDrawableOffset = i32v();

// The accepted totals and tier vector of every board, so only the boards touched this iteration are rescored
export const curP = i32s(MAX_MACHINES), curQ = i32s(MAX_MACHINES), curE = i32s(MAX_MACHINES), curPieces = i32s(MAX_MACHINES);
export const boardTiers = i32s(TIER_VECTOR_LENGTH * MAX_MACHINES);
export const fillP = i32v(), fillQ = i32v(), fillE = i32v();
// The rebuilt board's totals and tiers once the fill is done, and the same for the board a steal took from
export const builtP = i32v(), builtQ = i32v(), builtE = i32v(), builtPieces = i32v();
export const builtTiers = i32s(TIER_VECTOR_LENGTH);
export const built = i32s(BOARD_CELLS);
export const robbed = i32v();
export const robbedP = i32v(), robbedQ = i32v(), robbedE = i32v(), robbedPieces = i32v();
export const robbedTiers = i32s(TIER_VECTOR_LENGTH);
// Output of objectiveTiers, for the machine bound at the time
export const scoredTiers = i32s(TIER_VECTOR_LENGTH);

// Movable pieces on the other boards this board's layout can draw, each packed as owner << 10 | item, and the one offered to the draw
export const stealable = i32s(MAX_PIECES_PER_BOARD * MAX_MACHINES);
export const offered = i32v(), offeredOwner = i32v();

// The combined tier vector of the set being judged
export const curTiers = i32s(TIER_VECTOR_LENGTH);
export const epochTiers = i32s(TIER_VECTOR_LENGTH);
export const bestTiers = i32s(TIER_VECTOR_LENGTH);
export const hasEpoch = i32v(), hasRecord = i32v();
export const stagnation = i32v(), stagnations = i32v(), restarts = i32v();

// Items on the board this iteration, so the draw skips them
export const blocked = tgpu.privateVar(d.arrayOf(d.u32, BLOCKED_WORDS), d.arrayOf(d.u32, BLOCKED_WORDS)());
export const removable = i32s(MAX_PIECES_PER_BOARD);
export const removableCount = i32v();
export const fixedItem = i32s(MAX_PIECES_PER_BOARD);
export const fixedCellCount = i32s(MAX_PIECES_PER_BOARD);
export const fixedCells = i32s(MAX_PIECES_PER_BOARD * MAX_PIECE_CELLS);
export const freeCells = i32s(BOARD_CELLS);
export const freeCount = i32v();
// The occupied cells of the board being built, matched against the placement masks
export const occLo = i32v(), occHi = i32v();
// How many drawable modules of each shape are on the board, so the draw knows when a shape has none left to offer
export const shapeBlocked = i32s(SHAPE_COUNT);
export const boardIsEmpty = i32v();

// Outputs of evalPlacement
export const scoreOk = i32v(), scoreMajor = i32v(), scoreMinor = i32v();
export const seenNeighbors = i32s(MAX_PIECE_NEIGHBORS);

// Scratch and outputs of boardTotals
export const pieceItem = i32s(MAX_PIECES_PER_BOARD);
export const pieceMinX = i32s(MAX_PIECES_PER_BOARD);
export const pieceMinY = i32s(MAX_PIECES_PER_BOARD);
export const pieceCellCount = i32s(MAX_PIECES_PER_BOARD);
export const pieceCells = i32s(MAX_PIECES_PER_BOARD * MAX_PIECE_CELLS);
export const pieceAdjNodes = i32s(MAX_PIECES_PER_BOARD);
// The slot of the piece standing on each occupied cell, so a neighbour's piece is one lookup away
export const cellSlot = i32s(BOARD_CELLS);
export const seen = i32s(MAX_PIECE_NEIGHBORS);
export const totP = i32v(), totQ = i32v(), totE = i32v(), totPieces = i32v();

const bitOf = (i: number) => {
    'use gpu';
    return d.u32(d.u32(1) << d.u32(i & 31));
};

export const bit32 = (i: number) => {
    'use gpu';
    return d.i32(d.u32(1) << d.u32(i));
};

export const bitIsSet = (i: number) => {
    'use gpu';
    return (blocked.$[i >> d.u32(5)] & bitOf(i)) !== d.u32(0);
};

export const setBit = (i: number) => {
    'use gpu';
    blocked.$[i >> d.u32(5)] = blocked.$[i >> d.u32(5)] | bitOf(i);
};

export const clearBit = (i: number) => {
    'use gpu';
    blocked.$[i >> d.u32(5)] = blocked.$[i >> d.u32(5)] & ~bitOf(i);
};

export const clearBlocked = () => {
    'use gpu';
    for (let w = 0; w < BLOCKED_WORDS; w++) blocked.$[w] = d.u32(0);
};
