import tgpu, { d, type TgpuRoot } from 'typegpu';
import { DRAW_TOURNAMENT, MAX_DRAWS } from '../draw';
import {
    BOARD_CELLS, BOARD_H, BOARD_W, MAX_PIECE_CELLS, MAX_PIECE_NEIGHBORS, PLACE_LEFT_COL, PLACE_TOP_ROW, PLACE_TOUCHES_EDGE, PLACE_VALID, SCAN_STRIDES,
    SHAPE_COUNT
} from '../geometry';
import { EMPTY } from '../indexBoard';
import { DENSITY_TIER_WEIGHT, TIER_VECTOR_LENGTH } from '../objective';
import type { SolveSetup } from '../setup';
import { FLAG_FIXED, FLAG_PURE_NEGATIVE, FLAG_RECEIVER, FLAG_SIDE_MOUNT, FLAG_TOP_MOUNT, FLAG_WHITE, NF_SHIFT, RECV_MAX_NODES } from '../tables';
import {
    GEO_CELL_COUNT, GEO_CELLS, GEO_LENGTH, GEO_MASK_HI, GEO_MASK_LO, GEO_META, GEO_NBR_COUNT, GEO_NBRS, GEO_ORIENT_CELL_COUNT, GEO_ORIENT_CORNERS_HI,
    GEO_ORIENT_CORNERS_LO, GEO_ORIENT_COUNT, GEO_ORIENT_OFFSETS, GEO_ORIENT_START, GEO_SCAN_STRIDES,
    NO_RECORD, Params, PoolEntry,
    STAT_HAS_TARGET, STAT_MAXIMIZE, STAT_TARGET, STAT_TARGETED_INDEX, STAT_TIER_OF, STAT_WEIGHT, ThreadState, WORKGROUP_SIZE
} from './layout';
import { rngBelow, rngCoinFlip, rngCtr, rngInc } from './rng';
import * as s from './scratch';
import { buildGpuTables, buildInitialStates } from './upload';

const NO_SCORE_MAJOR = -10000;
const NEGATIVE_CONTACT_PENALTY = 1000;
const RECV_STRIDE = RECV_MAX_NODES + 1;

// The search of ../engine.ts, written to run one trajectory per GPU thread
export const createSearchKernel = (root: TgpuRoot, setup: SolveSetup, seed: number, threads: number, itersPerDispatch = 1) => {
    const tables = buildGpuTables(setup);
    const pool = root.createUniform(d.arrayOf(PoolEntry, tables.pool.length), tables.pool);
    const geometry = root.createReadonly(d.arrayOf(d.i32, GEO_LENGTH), Array.from(tables.geometry));
    const aux = root.createReadonly(d.arrayOf(d.i32, tables.aux.length), Array.from(tables.aux));
    const params = root.createUniform(Params, { ...tables.params, threadCount: threads, itersPerDispatch });
    const state = root.createMutable(d.arrayOf(ThreadState, threads), buildInitialStates(setup, seed, threads));
    const scores = root.createMutable(d.arrayOf(d.i32, threads * TIER_VECTOR_LENGTH));
    const champion = root.createMutable(d.arrayOf(d.i32, BOARD_CELLS));

    const statParam = (field: number, stat: number) => {
        'use gpu';
        return aux.$[params.$.statOffset + field * 3 + stat];
    };

    const recvBonus = (slot: number, stat: number, adjNodes: number) => {
        'use gpu';
        return aux.$[params.$.recvOffset + (slot * 3 + stat) * RECV_STRIDE + adjNodes];
    };

    const scoreStat = (delta: number, current: number, stat: number) => {
        'use gpu';
        const w = statParam(STAT_WEIGHT, stat);
        if (delta === 0 || w === 0) return d.i32(0);
        const target = statParam(STAT_TARGET, stat);
        const hasTarget = statParam(STAT_HAS_TARGET, stat);
        const maximize = statParam(STAT_MAXIMIZE, stat);
        const after = current + delta;

        if (hasTarget !== 0 && maximize === 0) {
            if (current >= target) {
                if (after >= target) return d.i32(0);
                return delta * w * 100;
            }
            if (after <= target) return delta * w;
            return (target - current) * w;
        }
        if (hasTarget !== 0) {
            if (current >= target) return delta * w;
            if (after <= target) return delta * w * 10;
            return (target - current) * w * 10 + (after - target) * w;
        }
        if (maximize !== 0) return delta * w;
        return d.i32(0);
    };

    const evalPlacement = (entry: number, item: number) => {
        'use gpu';
        s.scoreOk.$ = 0;
        const meta = geometry.$[GEO_META + entry];
        if ((meta & PLACE_VALID) === 0) return;
        if (((geometry.$[GEO_MASK_LO + entry] & s.occLo.$) | (geometry.$[GEO_MASK_HI + entry] & s.occHi.$)) !== 0) return;

        const flags = pool.$[item].flags;
        const isWhite = (flags & FLAG_WHITE) !== 0;
        const isPureNegative = (flags & FLAG_PURE_NEGATIVE) !== 0;
        const nfCount = (flags >> d.u32(NF_SHIFT)) & 3;

        let isConnected = (meta & PLACE_TOUCHES_EDGE) !== 0;
        let negativeContacts = 0;
        let seenCount = 0;

        const nbrCount = geometry.$[GEO_NBR_COUNT + entry];
        for (let k = 0; k < nbrCount; k++) {
            const adj = s.test.$[geometry.$[GEO_NBRS + entry * MAX_PIECE_NEIGHBORS + k]];
            if (adj < 0) continue;
            isConnected = true;

            const adjFlags = pool.$[adj].flags;
            const adjWhite = (adjFlags & FLAG_WHITE) !== 0;
            if (isWhite) {
                if (!adjWhite && (adjFlags & FLAG_PURE_NEGATIVE) !== 0) negativeContacts = negativeContacts + 1;
            } else if (isPureNegative && adjWhite) {
                negativeContacts = negativeContacts + 1;
            }

            let isSeen = false;
            for (let q = 0; q < seenCount; q++) {
                if (s.seenNeighbors.$[q] === adj) { isSeen = true; break; }
            }
            if (!isSeen) {
                s.seenNeighbors.$[seenCount] = adj;
                seenCount = seenCount + 1;
            }
        }

        if (!isConnected && s.boardIsEmpty.$ === 0) {
            s.scoreOk.$ = 1;
            s.scoreMajor.$ = NO_SCORE_MAJOR;
            s.scoreMinor.$ = 0;
            return;
        }

        let adjNodes = 0;
        let pDelta = 0;
        let qDelta = 0;
        let eDelta = 0;
        let nfPerf = 0;
        let nfQual = 0;
        let nfEff = 0;

        for (let q = 0; q < seenCount; q++) {
            const adj = s.seenNeighbors.$[q];
            const adjWhite = (pool.$[adj].flags & FLAG_WHITE) !== 0;
            if (!isWhite && adjWhite) {
                adjNodes = adjNodes + 1;
                pDelta = pDelta + pool.$[item].p20;
                qDelta = qDelta + pool.$[item].q20;
                eDelta = eDelta + pool.$[item].e20;
            } else if (isWhite && !adjWhite) {
                pDelta = pDelta + pool.$[adj].p20;
                qDelta = qDelta + pool.$[adj].q20;
                eDelta = eDelta + pool.$[adj].e20;
            }

            if (nfCount > 0 && !adjWhite) {
                if (pool.$[adj].p < 0) nfPerf = nfPerf + pool.$[adj].p;
                if (pool.$[adj].q < 0) nfQual = nfQual + pool.$[adj].q;
                if (pool.$[adj].e < 0) nfEff = nfEff + pool.$[adj].e;
            }
        }

        let myP = pool.$[item].p;
        let myQ = pool.$[item].q;
        let myE = pool.$[item].e;

        if ((flags & FLAG_SIDE_MOUNT) !== 0 && (meta & PLACE_LEFT_COL) !== 0) {
            myP = myP + pool.$[item].p20;
            myQ = myQ + pool.$[item].q20;
            myE = myE + pool.$[item].e20;
        }
        if ((flags & FLAG_TOP_MOUNT) !== 0 && (meta & PLACE_TOP_ROW) !== 0) {
            myP = myP + pool.$[item].p20;
            myQ = myQ + pool.$[item].q20;
            myE = myE + pool.$[item].e20;
        }
        if ((flags & FLAG_RECEIVER) !== 0) {
            const slot = pool.$[item].recvSlot;
            myP = myP + recvBonus(slot, 0, adjNodes);
            myQ = myQ + recvBonus(slot, 1, adjNodes);
            myE = myE + recvBonus(slot, 2, adjNodes);
        }

        if (nfCount > 0) {
            myP = d.i32((4 * myP + nfCount * nfPerf) / 4);
            myQ = d.i32((4 * myQ + nfCount * nfQual) / 4);
            myE = d.i32((4 * myE + nfCount * nfEff) / 4);
        }

        pDelta = pDelta + myP;
        qDelta = qDelta + myQ;
        eDelta = eDelta + myE;

        const statScore = scoreStat(pDelta, s.fillP.$, 0) + scoreStat(qDelta, s.fillQ.$, 1) + scoreStat(eDelta, s.fillE.$, 2);

        s.scoreOk.$ = 1;
        let major = statScore;
        if (statScore <= 0) major = NO_SCORE_MAJOR;
        s.scoreMajor.$ = major - negativeContacts * NEGATIVE_CONTACT_PENALTY;
        s.scoreMinor.$ = adjNodes;
    };

    const neighborCell = (x: number, y: number, dir: number) => {
        'use gpu';
        let nx = x;
        let ny = y;
        if (dir === 0) ny = y - 1;
        else if (dir === 1) ny = y + 1;
        else if (dir === 2) nx = x - 1;
        else nx = x + 1;
        if (nx < 0 || nx >= BOARD_W || ny < 0 || ny >= BOARD_H) return d.i32(-1);
        return ny * BOARD_W + nx;
    };

    const slotOfItem = (count: number, item: number) => {
        'use gpu';
        for (let k = 0; k < count; k++) {
            if (s.pieceItem.$[k] === item) return k;
        }
        return d.i32(-1);
    };

    const collectPieces = () => {
        'use gpu';
        let count = 0;
        for (let i = 0; i < BOARD_CELLS; i++) {
            const item = s.test.$[i];
            if (item < 0) continue;
            const y = d.i32(i / BOARD_W);
            const x = i - y * BOARD_W;
            // A piece is connected, so most of its cells have the cell to their left or above them in the same piece
            let slot = -1;
            if (x > 0 && s.test.$[i - 1] === item) slot = s.cellSlot.$[i - 1];
            else if (y > 0 && s.test.$[i - BOARD_W] === item) slot = s.cellSlot.$[i - BOARD_W];
            if (slot === -1) slot = slotOfItem(count, item);
            if (slot === -1) {
                slot = count;
                count = count + 1;
                s.pieceItem.$[slot] = item;
                s.pieceMinX.$[slot] = x;
                s.pieceMinY.$[slot] = y;
                s.pieceCellCount.$[slot] = 0;
                s.pieceAdjNodes.$[slot] = 0;
            } else {
                if (x < s.pieceMinX.$[slot]) s.pieceMinX.$[slot] = x;
                if (y < s.pieceMinY.$[slot]) s.pieceMinY.$[slot] = y;
            }
            s.pieceCells.$[slot * MAX_PIECE_CELLS + s.pieceCellCount.$[slot]] = i;
            s.pieceCellCount.$[slot] = s.pieceCellCount.$[slot] + 1;
            s.cellSlot.$[i] = slot;
        }
        return count;
    };

    // Adds what one node earns from the distinct non-node pieces around it and counts the node against each of them
    const addNodeBonus = (slot: number) => {
        'use gpu';
        let nodeP = 0;
        let nodeQ = 0;
        let nodeE = 0;
        let seenCount = 0;
        for (let c = 0; c < s.pieceCellCount.$[slot]; c++) {
            const idx = s.pieceCells.$[slot * MAX_PIECE_CELLS + c];
            const y = d.i32(idx / BOARD_W);
            const x = idx - y * BOARD_W;
            for (let dir = 0; dir < 4; dir++) {
                const cell = neighborCell(x, y, dir);
                if (cell < 0) continue;
                const adj = s.test.$[cell];
                if (adj < 0 || (pool.$[adj].flags & FLAG_WHITE) !== 0) continue;

                let dup = false;
                for (let k = 0; k < seenCount; k++) {
                    if (s.seen.$[k] === adj) { dup = true; break; }
                }
                if (dup) continue;
                s.seen.$[seenCount] = adj;
                seenCount = seenCount + 1;

                const adjSlot = s.cellSlot.$[cell];
                s.pieceAdjNodes.$[adjSlot] = s.pieceAdjNodes.$[adjSlot] + 1;
                nodeP = nodeP + pool.$[adj].p;
                nodeQ = nodeQ + pool.$[adj].q;
                nodeE = nodeE + pool.$[adj].e;
            }
        }
        s.totP.$ = s.totP.$ + d.i32(nodeP / 5);
        s.totQ.$ = s.totQ.$ + d.i32(nodeQ / 5);
        s.totE.$ = s.totE.$ + d.i32(nodeE / 5);
    };

    const addPieceStats = (slot: number) => {
        'use gpu';
        const item = s.pieceItem.$[slot];
        const flags = pool.$[item].flags;
        let p = pool.$[item].p;
        let q = pool.$[item].q;
        let e = pool.$[item].e;
        if (s.pieceMinX.$[slot] === 0 && (flags & FLAG_SIDE_MOUNT) !== 0) {
            p = p + pool.$[item].p20; q = q + pool.$[item].q20; e = e + pool.$[item].e20;
        }
        if (s.pieceMinY.$[slot] === 0 && (flags & FLAG_TOP_MOUNT) !== 0) {
            p = p + pool.$[item].p20; q = q + pool.$[item].q20; e = e + pool.$[item].e20;
        }
        if ((flags & FLAG_RECEIVER) !== 0) {
            const rs = pool.$[item].recvSlot;
            const adjNodes = s.pieceAdjNodes.$[slot];
            p = p + recvBonus(rs, 0, adjNodes);
            q = q + recvBonus(rs, 1, adjNodes);
            e = e + recvBonus(rs, 2, adjNodes);
        }

        const nfCount = (flags >> d.u32(NF_SHIFT)) & 3;
        if (nfCount > 0) {
            let nfP = 0;
            let nfQ = 0;
            let nfE = 0;
            let seenCount = 0;
            for (let c = 0; c < s.pieceCellCount.$[slot]; c++) {
                const idx = s.pieceCells.$[slot * MAX_PIECE_CELLS + c];
                const y = d.i32(idx / BOARD_W);
                const x = idx - y * BOARD_W;
                for (let dir = 0; dir < 4; dir++) {
                    const cell = neighborCell(x, y, dir);
                    if (cell < 0) continue;
                    const adj = s.test.$[cell];
                    if (adj < 0 || adj === item || (pool.$[adj].flags & FLAG_WHITE) !== 0) continue;

                    let dup = false;
                    for (let k = 0; k < seenCount; k++) {
                        if (s.seen.$[k] === adj) { dup = true; break; }
                    }
                    if (dup) continue;
                    s.seen.$[seenCount] = adj;
                    seenCount = seenCount + 1;

                    if (pool.$[adj].p < 0) nfP = nfP + pool.$[adj].p;
                    if (pool.$[adj].q < 0) nfQ = nfQ + pool.$[adj].q;
                    if (pool.$[adj].e < 0) nfE = nfE + pool.$[adj].e;
                }
            }
            p = d.i32((4 * p + nfCount * nfP) / 4);
            q = d.i32((4 * q + nfCount * nfQ) / 4);
            e = d.i32((4 * e + nfCount * nfE) / 4);
        }
        s.totP.$ = s.totP.$ + p;
        s.totQ.$ = s.totQ.$ + q;
        s.totE.$ = s.totE.$ + e;
    };

    // The totals of the test board, as ../boardTotals.ts computes them
    const boardTotals = () => {
        'use gpu';
        const count = collectPieces();
        s.totP.$ = 0; s.totQ.$ = 0; s.totE.$ = 0;
        for (let slot = 0; slot < count; slot++) {
            if ((pool.$[s.pieceItem.$[slot]].flags & FLAG_WHITE) !== 0) addNodeBonus(slot);
        }
        for (let slot = 0; slot < count; slot++) {
            if ((pool.$[s.pieceItem.$[slot]].flags & FLAG_WHITE) === 0) addPieceStats(slot);
        }
        s.totPieces.$ = count;
    };

    const refreshFillTotals = () => {
        'use gpu';
        boardTotals();
        s.fillP.$ = s.totP.$; s.fillQ.$ = s.totQ.$; s.fillE.$ = s.totE.$;
    };

    const statTotal = (stat: number) => {
        'use gpu';
        if (stat === 0) return s.totP.$;
        if (stat === 1) return s.totQ.$;
        return s.totE.$;
    };

    const objectiveTiers = () => {
        'use gpu';
        for (let i = 0; i < TIER_VECTOR_LENGTH; i++) s.curTiers.$[i] = 0;
        for (let stat = 0; stat < 3; stat++) {
            const ti = statParam(STAT_TIER_OF, stat);
            if (ti < 0) continue;
            const t = statTotal(stat);
            const target = statParam(STAT_TARGET, stat);
            if (statParam(STAT_HAS_TARGET, stat) !== 0 && t < target) s.curTiers.$[ti] = s.curTiers.$[ti] - (target - t) * 10000;
            if (statParam(STAT_MAXIMIZE, stat) !== 0) s.curTiers.$[ti] = s.curTiers.$[ti] + t * 10;
        }
        s.curTiers.$[params.$.tierCount] = -s.totPieces.$ * DENSITY_TIER_WEIGHT;
    };

    const compareCurToEpoch = () => {
        'use gpu';
        for (let i = 0; i <= params.$.tierCount; i++) {
            if (s.curTiers.$[i] < s.epochTiers.$[i]) return d.i32(-1);
            if (s.curTiers.$[i] > s.epochTiers.$[i]) return d.i32(1);
        }
        return d.i32(0);
    };

    const curBeatsBest = () => {
        'use gpu';
        for (let i = 0; i <= params.$.tierCount; i++) {
            if (s.curTiers.$[i] < s.bestTiers.$[i]) return false;
            if (s.curTiers.$[i] > s.bestTiers.$[i]) return true;
        }
        return false;
    };

    const compactFreeCells = () => {
        'use gpu';
        let write = 0;
        for (let c = 0; c < s.freeCount.$; c++) {
            if (s.test.$[s.freeCells.$[c]] === EMPTY) {
                s.freeCells.$[write] = s.freeCells.$[c];
                write = write + 1;
            }
        }
        s.freeCount.$ = write;
    };

    const placeBestFit = (item: number, incumbent: number) => {
        'use gpu';
        let bestEntry = -1;
        let haveScore = incumbent === -1;
        let bestMajor = 0;
        let bestMinor = 0;

        if (incumbent !== -1) {
            evalPlacement(incumbent, item);
            if (s.scoreOk.$ !== 0) {
                haveScore = true;
                bestMajor = s.scoreMajor.$; bestMinor = s.scoreMinor.$;
                bestEntry = incumbent;
            }
        }

        const orientStart = pool.$[item].orientStart;
        const orientEnd = orientStart + pool.$[item].orientCount;
        for (let c = 0; c < s.freeCount.$; c++) {
            const anchor = s.freeCells.$[c];
            for (let g = orientStart; g < orientEnd; g++) {
                const entry = g * BOARD_CELLS + anchor;
                if ((geometry.$[GEO_META + entry] & PLACE_VALID) === 0) continue;
                evalPlacement(entry, item);
                if (s.scoreOk.$ === 0) continue;
                if (!haveScore || s.scoreMajor.$ > bestMajor || (s.scoreMajor.$ === bestMajor && s.scoreMinor.$ > bestMinor)) {
                    haveScore = true;
                    bestMajor = s.scoreMajor.$; bestMinor = s.scoreMinor.$;
                    bestEntry = entry;
                }
            }
        }

        if (bestEntry === -1) return false;

        const cellCount = geometry.$[GEO_CELL_COUNT + bestEntry];
        for (let i = 0; i < cellCount; i++) s.test.$[geometry.$[GEO_CELLS + bestEntry * MAX_PIECE_CELLS + i]] = item;
        s.occLo.$ = s.occLo.$ | geometry.$[GEO_MASK_LO + bestEntry];
        s.occHi.$ = s.occHi.$ | geometry.$[GEO_MASK_HI + bestEntry];
        compactFreeCells();
        return true;
    };

    // The shift test of shapeFitsFree in ../geometry.ts, so the draw never has to offer a module of a shape with nowhere to go
    const shapeFits = (shape: number) => {
        'use gpu';
        const freeLo = d.u32(~s.occLo.$);
        const hiBits = d.u32(~s.occHi.$ & 7);
        const orientStart = geometry.$[GEO_ORIENT_START + shape];
        const orientEnd = orientStart + geometry.$[GEO_ORIENT_COUNT + shape];
        for (let g = orientStart; g < orientEnd; g++) {
            let lo = d.u32(geometry.$[GEO_ORIENT_CORNERS_LO + g]);
            let hi = d.u32(geometry.$[GEO_ORIENT_CORNERS_HI + g]);
            const cellCount = geometry.$[GEO_ORIENT_CELL_COUNT + g];
            for (let i = 0; i < cellCount; i++) {
                const c = d.u32(geometry.$[GEO_ORIENT_OFFSETS + g * MAX_PIECE_CELLS + i]);
                lo = lo & ((freeLo >>> c) | ((hiBits << (d.u32(31) - c)) << d.u32(1)));
                hi = hi & (hiBits >>> c);
            }
            if ((lo | hi) !== d.u32(0)) return true;
        }
        return false;
    };

    const homeEntryOf = (fixed: number) => {
        'use gpu';
        const item = s.fixedItem.$[fixed];
        const cellCount = s.fixedCellCount.$[fixed];
        const anchor = s.fixedCells.$[fixed * MAX_PIECE_CELLS];
        const orientEnd = pool.$[item].orientStart + pool.$[item].orientCount;
        for (let g = pool.$[item].orientStart; g < orientEnd; g++) {
            const entry = g * BOARD_CELLS + anchor;
            if ((geometry.$[GEO_META + entry] & PLACE_VALID) === 0 || geometry.$[GEO_CELL_COUNT + entry] !== cellCount) continue;
            let matches = true;
            for (let i = 0; i < cellCount; i++) {
                if (s.fixedCells.$[fixed * MAX_PIECE_CELLS + i] !== geometry.$[GEO_CELLS + entry * MAX_PIECE_CELLS + i]) { matches = false; break; }
            }
            if (matches) return entry;
        }
        return d.i32(-1);
    };

    const shuffleRemovable = (count: number) => {
        'use gpu';
        for (let i = count - 1; i > 0; i--) {
            const j = rngBelow(i + 1);
            const tmp = s.removable.$[i];
            s.removable.$[i] = s.removable.$[j];
            s.removable.$[j] = tmp;
        }
    };

    // Returns how many fixed pieces the board holds, after recording the removable ones in blocked and removable
    const scanBoard = () => {
        'use gpu';
        s.removableCount.$ = 0;
        let fixedCount = 0;
        for (let i = 0; i < BOARD_CELLS; i++) {
            const item = s.test.$[i];
            if (item < 0) continue;
            if ((pool.$[item].flags & FLAG_FIXED) !== 0) {
                let f = 0;
                while (f < fixedCount && s.fixedItem.$[f] !== item) f = f + 1;
                if (f === fixedCount) {
                    s.fixedItem.$[fixedCount] = item;
                    s.fixedCellCount.$[fixedCount] = 0;
                    fixedCount = fixedCount + 1;
                }
                s.fixedCells.$[f * MAX_PIECE_CELLS + s.fixedCellCount.$[f]] = i;
                s.fixedCellCount.$[f] = s.fixedCellCount.$[f] + 1;
                continue;
            }
            if (!s.bitIsSet(item)) {
                s.setBit(item);
                s.removable.$[s.removableCount.$] = item;
                s.removableCount.$ = s.removableCount.$ + 1;
            }
        }
        return fixedCount;
    };

    const ruin = (isStagnant: boolean) => {
        'use gpu';
        const removableCount = s.removableCount.$;
        let removeCount = 0;
        if (removableCount > 0) {
            removeCount = rngBelow(Math.min(3, removableCount)) + 1;
            if (isStagnant) removeCount = Math.max(1, d.i32((removableCount * (50 + rngBelow(40))) / 100));

            shuffleRemovable(removableCount);
            for (let i = 0; i < removeCount; i++) s.clearBit(s.removable.$[i]);

            for (let i = 0; i < BOARD_CELLS; i++) {
                const item = s.test.$[i];
                if (item >= 0 && (pool.$[item].flags & FLAG_FIXED) === 0 && !s.bitIsSet(item)) s.test.$[i] = EMPTY;
            }
        }

        for (let shape = 0; shape < SHAPE_COUNT; shape++) s.shapeBlocked.$[shape] = 0;
        for (let i = removeCount; i < removableCount; i++) {
            const item = s.removable.$[i];
            if (pool.$[item].drawable !== 0) {
                const shape = pool.$[item].shape;
                s.shapeBlocked.$[shape] = s.shapeBlocked.$[shape] + 1;
            }
        }
    };

    // The free cells in the scan order of ../engine.ts: from a random cell, with a random stride coprime to the board size
    const collectFreeCells = () => {
        'use gpu';
        let n = 0;
        s.occLo.$ = 0;
        s.occHi.$ = 0;
        const scanStart = rngBelow(BOARD_CELLS);
        const scanStride = geometry.$[GEO_SCAN_STRIDES + rngBelow(SCAN_STRIDES.length)];
        for (let k = 0; k < BOARD_CELLS; k++) {
            const i = (scanStart + k * scanStride) % BOARD_CELLS;
            if (s.test.$[i] === EMPTY) {
                s.freeCells.$[n] = i;
                n = n + 1;
            } else if (i < 32) {
                s.occLo.$ = s.occLo.$ | s.bit32(i);
            } else {
                s.occHi.$ = s.occHi.$ | s.bit32(i - 32);
            }
        }
        s.freeCount.$ = n;
    };

    const relocateFixed = (fixedCount: number) => {
        'use gpu';
        for (let f = 0; f < fixedCount; f++) {
            const home = homeEntryOf(f);
            for (let c = 0; c < s.fixedCellCount.$[f]; c++) {
                const idx = s.fixedCells.$[f * MAX_PIECE_CELLS + c];
                s.test.$[idx] = EMPTY;
                if (idx < 32) s.occLo.$ = s.occLo.$ & ~s.bit32(idx);
                else s.occHi.$ = s.occHi.$ & ~s.bit32(idx - 32);
                s.freeCells.$[s.freeCount.$] = idx;
                s.freeCount.$ = s.freeCount.$ + 1;
            }
            placeBestFit(s.fixedItem.$[f], home);
            s.boardIsEmpty.$ = 0;
            if (params.$.needsTotals !== 0) refreshFillTotals();
        }
    };

    const metTargetMask = () => {
        'use gpu';
        let mask = 0;
        for (let stat = 0; stat < 3; stat++) {
            const ti = statParam(STAT_TARGETED_INDEX, stat);
            if (ti < 0) continue;
            let t = s.curP.$;
            if (stat === 1) t = s.curQ.$;
            if (stat === 2) t = s.curE.$;
            if (t >= statParam(STAT_TARGET, stat)) mask = mask | s.bit32(ti);
        }
        return mask;
    };

    const drawRank = (metMask: number, pos: number) => {
        'use gpu';
        return aux.$[params.$.drawRankOffset + metMask * params.$.drawCount + pos];
    };

    const shapeRun = (shape: number) => {
        'use gpu';
        return aux.$[params.$.shapeStartOffset + shape + 1] - aux.$[params.$.shapeStartOffset + shape];
    };

    const shapeOffered = (shape: number, infeasible: number) => {
        'use gpu';
        return (infeasible & s.bit32(shape)) === 0 && s.shapeBlocked.$[shape] < shapeRun(shape);
    };

    // Only the shapes the draw could still offer are worth settling
    const infeasibleShapesNow = (known: number) => {
        'use gpu';
        let infeasible = known;
        for (let shape = 0; shape < SHAPE_COUNT; shape++) {
            if (shapeOffered(shape, infeasible) && !shapeFits(shape)) infeasible = infeasible | s.bit32(shape);
        }
        return infeasible;
    };

    const drawWeight = (infeasible: number) => {
        'use gpu';
        let weight = 0;
        for (let shape = 0; shape < SHAPE_COUNT; shape++) {
            if (shapeOffered(shape, infeasible)) weight = weight + shapeRun(shape);
        }
        return weight;
    };

    const drawPosition = (infeasible: number, r: number) => {
        'use gpu';
        let rest = r;
        for (let shape = 0; shape < SHAPE_COUNT; shape++) {
            if (!shapeOffered(shape, infeasible)) continue;
            const run = shapeRun(shape);
            if (rest < run) return aux.$[params.$.shapeStartOffset + shape] + rest;
            rest = rest - run;
        }
        return -1;
    };

    const drawTournament = (infeasible: number, weight: number, metMask: number) => {
        'use gpu';
        let pick = -1;
        for (let t = 0; t < DRAW_TOURNAMENT; t++) {
            const pos = drawPosition(infeasible, rngBelow(weight));
            if (s.bitIsSet(aux.$[params.$.drawListOffset + pos])) continue;
            if (pick === -1 || drawRank(metMask, pos) > drawRank(metMask, pick)) pick = pos;
        }
        return pick;
    };

    const fill = () => {
        'use gpu';
        const metMask = metTargetMask();
        let infeasible = infeasibleShapesNow(0);
        let weight = drawWeight(infeasible);
        for (let drawn = 0; drawn < MAX_DRAWS; drawn++) {
            if (weight === 0) break;
            const pos = drawTournament(infeasible, weight, metMask);
            if (pos === -1) continue;
            const item = aux.$[params.$.drawListOffset + pos];
            const shape = pool.$[item].shape;

            if (placeBestFit(item, -1)) {
                s.setBit(item);
                s.shapeBlocked.$[shape] = s.shapeBlocked.$[shape] + 1;
                s.boardIsEmpty.$ = 0;
                if (params.$.needsTotals !== 0) refreshFillTotals();
                infeasible = infeasibleShapesNow(infeasible);
            } else {
                infeasible = infeasible | s.bit32(shape);
            }
            weight = drawWeight(infeasible);
        }
    };

    const acceptTest = () => {
        'use gpu';
        for (let i = 0; i < BOARD_CELLS; i++) s.cur.$[i] = s.test.$[i];
        s.curP.$ = s.totP.$; s.curQ.$ = s.totQ.$; s.curE.$ = s.totE.$; s.curPieces.$ = s.totPieces.$;
    };

    const recordBest = (t: number) => {
        'use gpu';
        for (let i = 0; i < TIER_VECTOR_LENGTH; i++) s.bestTiers.$[i] = s.curTiers.$[i];
        s.hasRecord.$ = 1;
        for (let i = 0; i < BOARD_CELLS; i++) state.$[t].best[i] = s.cur.$[i];
    };

    const restart = () => {
        'use gpu';
        s.hasEpoch.$ = 0;
        for (let i = 0; i < BOARD_CELLS; i++) s.cur.$[i] = aux.$[params.$.initialBoardOffset + i];
        s.curPieces.$ = -1;
    };

    // Totals of the accepted board are recomputed lazily, marked by a negative piece count, so restarts and the first iteration share one path
    const ensureCurTotals = () => {
        'use gpu';
        if (s.curPieces.$ >= 0) return;
        for (let i = 0; i < BOARD_CELLS; i++) s.test.$[i] = s.cur.$[i];
        boardTotals();
        s.curP.$ = s.totP.$; s.curQ.$ = s.totQ.$; s.curE.$ = s.totE.$; s.curPieces.$ = s.totPieces.$;
    };

    const iterate = (t: number) => {
        'use gpu';
        ensureCurTotals();
        const isStagnant = s.stagnation.$ >= params.$.stagnationLimit;

        for (let i = 0; i < BOARD_CELLS; i++) s.test.$[i] = s.cur.$[i];
        s.clearBlocked();
        const fixedCount = scanBoard();
        ruin(isStagnant);
        collectFreeCells();
        s.boardIsEmpty.$ = 0;
        if (s.freeCount.$ === params.$.openCellCount) s.boardIsEmpty.$ = 1;

        s.fillP.$ = 0; s.fillQ.$ = 0; s.fillE.$ = 0;
        if (params.$.needsTotals !== 0) refreshFillTotals();
        relocateFixed(fixedCount);
        fill();

        boardTotals();
        objectiveTiers();

        let ordering = 1;
        if (s.hasEpoch.$ !== 0) ordering = compareCurToEpoch();
        const improved = ordering > 0;
        if (improved || (ordering === 0 && rngCoinFlip())) {
            if (improved) {
                for (let i = 0; i < TIER_VECTOR_LENGTH; i++) s.epochTiers.$[i] = s.curTiers.$[i];
                s.hasEpoch.$ = 1;
            }
            acceptTest();
            if (improved && (s.hasRecord.$ === 0 || curBeatsBest())) {
                recordBest(t);
                s.stagnation.$ = 0;
            } else {
                s.stagnation.$ = s.stagnation.$ + 1;
            }
        } else {
            s.stagnation.$ = s.stagnation.$ + 1;
        }

        if (isStagnant) {
            s.stagnation.$ = 0;
            s.stagnations.$ = s.stagnations.$ + 1;
            if (s.stagnations.$ >= params.$.restartAfter) {
                s.stagnations.$ = 0;
                restart();
            }
        }
    };

    const loadState = (t: number) => {
        'use gpu';
        rngCtr.$ = state.$[t].rngCtr;
        rngInc.$ = state.$[t].rngInc;
        s.stagnation.$ = state.$[t].stagnation;
        s.stagnations.$ = state.$[t].stagnations;
        s.hasEpoch.$ = state.$[t].hasEpoch;
        s.hasRecord.$ = state.$[t].hasRecord;
        s.curP.$ = state.$[t].curP; s.curQ.$ = state.$[t].curQ; s.curE.$ = state.$[t].curE; s.curPieces.$ = state.$[t].curPieces;
        for (let i = 0; i < TIER_VECTOR_LENGTH; i++) {
            s.epochTiers.$[i] = state.$[t].epochTiers[i];
            s.bestTiers.$[i] = state.$[t].bestTiers[i];
        }
        for (let i = 0; i < BOARD_CELLS; i++) s.cur.$[i] = state.$[t].cur[i];
    };

    const storeState = (t: number) => {
        'use gpu';
        state.$[t].rngCtr = rngCtr.$;
        state.$[t].rngInc = rngInc.$;
        state.$[t].stagnation = s.stagnation.$;
        state.$[t].stagnations = s.stagnations.$;
        state.$[t].hasEpoch = s.hasEpoch.$;
        state.$[t].hasRecord = s.hasRecord.$;
        state.$[t].curP = s.curP.$; state.$[t].curQ = s.curQ.$; state.$[t].curE = s.curE.$; state.$[t].curPieces = s.curPieces.$;
        for (let i = 0; i < TIER_VECTOR_LENGTH; i++) {
            state.$[t].epochTiers[i] = s.epochTiers.$[i];
            state.$[t].bestTiers[i] = s.bestTiers.$[i];
        }
        for (let i = 0; i < BOARD_CELLS; i++) state.$[t].cur[i] = s.cur.$[i];
        for (let i = 0; i < TIER_VECTOR_LENGTH; i++) {
            let v = s.bestTiers.$[i];
            if (s.hasRecord.$ === 0) v = 0;
            if (s.hasRecord.$ === 0 && i === 0) v = NO_RECORD;
            scores.$[t * TIER_VECTOR_LENGTH + i] = v;
        }
    };

    const runThread = (t: number) => {
        'use gpu';
        loadState(t);
        for (let k = 0; k < params.$.itersPerDispatch; k++) iterate(t);
        storeState(t);
    };

    const searchStep = tgpu.computeFn({ workgroupSize: [WORKGROUP_SIZE], in: { gid: d.builtin.globalInvocationId } })((input) => {
        'use gpu';
        const t = d.i32(input.gid.x);
        if (t >= params.$.threadCount) return;
        runThread(t);
    });

    const extractChampion = tgpu.computeFn({ workgroupSize: [WORKGROUP_SIZE], in: { gid: d.builtin.globalInvocationId } })((input) => {
        'use gpu';
        const i = d.i32(input.gid.x);
        if (i >= BOARD_CELLS) return;
        champion.$[i] = state.$[params.$.championIdx].best[i];
    });

    return { tables, params, state, scores, champion, searchStep, extractChampion, runThread };
};

export type SearchKernel = ReturnType<typeof createSearchKernel>;
