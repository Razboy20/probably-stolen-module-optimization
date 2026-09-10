import { NEIGHBOR_DX, NEIGHBOR_DY } from './board';
import { BOARD_CELLS, BOARD_H, BOARD_W, MAX_PIECE_CELLS, MAX_PIECE_NEIGHBORS } from './geometry';
import type { IndexBoard } from './indexBoard';
import { FLAG_RECEIVER, FLAG_SIDE_MOUNT, FLAG_TOP_MOUNT, FLAG_WHITE, nfCountOf, type PoolTables, recvBonus } from './tables';
import { MAX_PIECES_PER_BOARD } from './pool';

export interface BoardTotals {
    p: number;
    q: number;
    e: number;
    pieces: number;
}

// Per-piece scratch for one evaluation, indexed by the order pieces are first met in a row-major scan
const pieceItem = new Int32Array(MAX_PIECES_PER_BOARD);
const pieceMinX = new Int32Array(MAX_PIECES_PER_BOARD);
const pieceMinY = new Int32Array(MAX_PIECES_PER_BOARD);
const pieceCellCount = new Int32Array(MAX_PIECES_PER_BOARD);
const pieceCells = new Int32Array(MAX_PIECES_PER_BOARD * MAX_PIECE_CELLS);
const pieceAdjNodes = new Int32Array(MAX_PIECES_PER_BOARD);
const seen = new Int32Array(MAX_PIECE_NEIGHBORS);

const slotOfItem = (count: number, item: number) => {
    for (let s = 0; s < count; s++) if (pieceItem[s] === item) return s;
    return -1;
};

// The same totals the report computes, on an index board and without allocating
// Node bonuses are 20% of the neighbours' internal stats and Negative Feedback a quarter of the absorbed negatives per effect,
// each rounded toward zero as one quantity, which is what the report's roundStat does on integer inputs
export const boardTotals = (tables: PoolTables, board: IndexBoard, out: BoardTotals) => {
    let count = 0;
    for (let i = 0; i < BOARD_CELLS; i++) {
        const item = board[i];
        if (item < 0) continue;
        const x = i % BOARD_W;
        const y = (i - x) / BOARD_W;
        let s = slotOfItem(count, item);
        if (s === -1) {
            s = count++;
            pieceItem[s] = item;
            pieceMinX[s] = x;
            pieceMinY[s] = y;
            pieceCellCount[s] = 0;
            pieceAdjNodes[s] = 0;
        } else {
            if (x < pieceMinX[s]) pieceMinX[s] = x;
            if (y < pieceMinY[s]) pieceMinY[s] = y;
        }
        pieceCells[s * MAX_PIECE_CELLS + pieceCellCount[s]++] = i;
    }

    let totalP = 0, totalQ = 0, totalE = 0;

    for (let s = 0; s < count; s++) {
        const item = pieceItem[s];
        if ((tables.flags[item] & FLAG_WHITE) === 0) continue;

        let nodeP = 0, nodeQ = 0, nodeE = 0;
        let seenCount = 0;
        for (let c = 0; c < pieceCellCount[s]; c++) {
            const idx = pieceCells[s * MAX_PIECE_CELLS + c];
            const x = idx % BOARD_W;
            const y = (idx - x) / BOARD_W;
            for (let d = 0; d < 4; d++) {
                const nx = x + NEIGHBOR_DX[d];
                const ny = y + NEIGHBOR_DY[d];
                if (nx < 0 || nx >= BOARD_W || ny < 0 || ny >= BOARD_H) continue;
                const adj = board[ny * BOARD_W + nx];
                if (adj < 0 || (tables.flags[adj] & FLAG_WHITE) !== 0) continue;

                let dup = false;
                for (let k = 0; k < seenCount; k++) if (seen[k] === adj) { dup = true; break; }
                if (dup) continue;
                seen[seenCount++] = adj;

                pieceAdjNodes[slotOfItem(count, adj)]++;
                nodeP += tables.p[adj];
                nodeQ += tables.q[adj];
                nodeE += tables.e[adj];
            }
        }
        totalP += Math.trunc(nodeP / 5);
        totalQ += Math.trunc(nodeQ / 5);
        totalE += Math.trunc(nodeE / 5);
    }

    for (let s = 0; s < count; s++) {
        const item = pieceItem[s];
        const flags = tables.flags[item];
        if ((flags & FLAG_WHITE) !== 0) continue;

        let p = tables.p[item], q = tables.q[item], e = tables.e[item];
        let pBonus = 0, qBonus = 0, eBonus = 0;
        if (pieceMinX[s] === 0 && (flags & FLAG_SIDE_MOUNT) !== 0) {
            pBonus += tables.p20[item]; qBonus += tables.q20[item]; eBonus += tables.e20[item];
        }
        if (pieceMinY[s] === 0 && (flags & FLAG_TOP_MOUNT) !== 0) {
            pBonus += tables.p20[item]; qBonus += tables.q20[item]; eBonus += tables.e20[item];
        }
        if ((flags & FLAG_RECEIVER) !== 0) {
            const slot = tables.recvSlot[item];
            const adjNodes = pieceAdjNodes[s];
            pBonus += recvBonus(tables, slot, 0, adjNodes);
            qBonus += recvBonus(tables, slot, 1, adjNodes);
            eBonus += recvBonus(tables, slot, 2, adjNodes);
        }
        p += pBonus; q += qBonus; e += eBonus;

        const nfCount = nfCountOf(flags);
        if (nfCount > 0) {
            let nfP = 0, nfQ = 0, nfE = 0;
            let seenCount = 0;
            for (let c = 0; c < pieceCellCount[s]; c++) {
                const idx = pieceCells[s * MAX_PIECE_CELLS + c];
                const x = idx % BOARD_W;
                const y = (idx - x) / BOARD_W;
                for (let d = 0; d < 4; d++) {
                    const nx = x + NEIGHBOR_DX[d];
                    const ny = y + NEIGHBOR_DY[d];
                    if (nx < 0 || nx >= BOARD_W || ny < 0 || ny >= BOARD_H) continue;
                    const adj = board[ny * BOARD_W + nx];
                    if (adj < 0 || adj === item || (tables.flags[adj] & FLAG_WHITE) !== 0) continue;

                    let dup = false;
                    for (let k = 0; k < seenCount; k++) if (seen[k] === adj) { dup = true; break; }
                    if (dup) continue;
                    seen[seenCount++] = adj;

                    if (tables.p[adj] < 0) nfP += tables.p[adj];
                    if (tables.q[adj] < 0) nfQ += tables.q[adj];
                    if (tables.e[adj] < 0) nfE += tables.e[adj];
                }
            }
            p = Math.trunc((4 * p + nfCount * nfP) / 4);
            q = Math.trunc((4 * q + nfCount * nfQ) / 4);
            e = Math.trunc((4 * e + nfCount * nfE) / 4);
        }

        totalP += p; totalQ += q; totalE += e;
    }

    out.p = totalP;
    out.q = totalQ;
    out.e = totalE;
    out.pieces = count;
};
