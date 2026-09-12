import type { IndexBoard } from './indexBoard';
import {
    MAX_PIECE_NEIGHBORS, PLACE_LEFT_COL, PLACE_MASK_HI, PLACE_MASK_LO, PLACE_META, PLACE_NBR_COUNT, PLACE_NBRS,
    PLACE_TOP_ROW, PLACE_TOUCHES_EDGE, PLACE_VALID
} from './geometry';
import { type ScoringParams, scoreStat } from './scoring';
import {
    FLAG_PURE_NEGATIVE, FLAG_RECEIVER, FLAG_SIDE_MOUNT, FLAG_TOP_MOUNT, FLAG_WHITE, nfCountOf, type PoolTables, recvBonus
} from './tables';

/* The worth of one placement, as an ordered pair rather than a float
 * `major` is the stat score less a thousand per contact between a node and a purely negative module (or a flat -10000 less those when the placement scores nothing)
 * `minor` breaks ties by adjacent nodes: more of them for a placement that scores, fewer for one that does not
 */
export interface PlaceScore {
    ok: boolean;
    major: number;
    minor: number;
}

export const NO_SCORE_MAJOR = -10000;
const NEGATIVE_CONTACT_PENALTY = 1000;

// Distinct neighbouring modules of the candidate placement
const seenNeighbors = new Int32Array(MAX_PIECE_NEIGHBORS);

export const evalPlacement = (
    tables: PoolTables,
    entry: number,
    item: number,
    board: IndexBoard,
    occupiedLo: number, occupiedHi: number,
    boardIsEmpty: boolean,
    params: ScoringParams,
    currentP: number, currentQ: number, currentE: number,
    out: PlaceScore
) => {
    out.ok = false;
    const meta = PLACE_META[entry];
    if ((meta & PLACE_VALID) === 0) return;
    if (((PLACE_MASK_LO[entry] & occupiedLo) | (PLACE_MASK_HI[entry] & occupiedHi)) !== 0) return;

    const flags = tables.flags[item];
    const isWhite = (flags & FLAG_WHITE) !== 0;
    const isPureNegative = (flags & FLAG_PURE_NEGATIVE) !== 0;
    const nfCount = nfCountOf(flags);

    let isConnected = (meta & PLACE_TOUCHES_EDGE) !== 0;
    let negativeContacts = 0;
    let seenCount = 0;

    const nbrCount = PLACE_NBR_COUNT[entry];
    for (let k = 0; k < nbrCount; k++) {
        const adj = board[PLACE_NBRS[entry * MAX_PIECE_NEIGHBORS + k]];
        if (adj < 0) continue;
        isConnected = true;

        const adjWhite = (tables.flags[adj] & FLAG_WHITE) !== 0;
        if (isWhite) {
            if (!adjWhite && (tables.flags[adj] & FLAG_PURE_NEGATIVE) !== 0) negativeContacts++;
        } else if (isPureNegative && adjWhite) {
            negativeContacts++;
        }

        let seen = false;
        for (let s = 0; s < seenCount; s++) {
            if (seenNeighbors[s] === adj) { seen = true; break; }
        }
        if (!seen) seenNeighbors[seenCount++] = adj;
    }

    if (!isConnected && !boardIsEmpty) {
        out.ok = true;
        out.major = NO_SCORE_MAJOR;
        out.minor = 0;
        return;
    }

    let adjNodes = 0;
    let pDelta = 0, qDelta = 0, eDelta = 0;
    let nfPerf = 0, nfQual = 0, nfEff = 0;

    for (let s = 0; s < seenCount; s++) {
        const adj = seenNeighbors[s];
        const adjWhite = (tables.flags[adj] & FLAG_WHITE) !== 0;
        if (!isWhite && adjWhite) {
            adjNodes++;
            pDelta += tables.p20[item];
            qDelta += tables.q20[item];
            eDelta += tables.e20[item];
        } else if (isWhite && !adjWhite) {
            pDelta += tables.p20[adj];
            qDelta += tables.q20[adj];
            eDelta += tables.e20[adj];
        }

        if (nfCount > 0 && !adjWhite) {
            if (tables.p[adj] < 0) nfPerf += tables.p[adj];
            if (tables.q[adj] < 0) nfQual += tables.q[adj];
            if (tables.e[adj] < 0) nfEff += tables.e[adj];
        }
    }

    let myP = tables.p[item];
    let myQ = tables.q[item];
    let myE = tables.e[item];

    let pBonus = 0, qBonus = 0, eBonus = 0;
    if ((flags & FLAG_SIDE_MOUNT) !== 0 && (meta & PLACE_LEFT_COL) !== 0) {
        pBonus += tables.p20[item];
        qBonus += tables.q20[item];
        eBonus += tables.e20[item];
    }
    if ((flags & FLAG_TOP_MOUNT) !== 0 && (meta & PLACE_TOP_ROW) !== 0) {
        pBonus += tables.p20[item];
        qBonus += tables.q20[item];
        eBonus += tables.e20[item];
    }
    if ((flags & FLAG_RECEIVER) !== 0) {
        const slot = tables.recvSlot[item];
        pBonus += recvBonus(tables, slot, 0, adjNodes);
        qBonus += recvBonus(tables, slot, 1, adjNodes);
        eBonus += recvBonus(tables, slot, 2, adjNodes);
    }

    myP += pBonus;
    myQ += qBonus;
    myE += eBonus;

    // A quarter of the absorbed negatives per Negative Feedback, rounded toward zero as one quantity
    if (nfCount > 0) {
        myP = Math.trunc((4 * myP + nfCount * nfPerf) / 4);
        myQ = Math.trunc((4 * myQ + nfCount * nfQual) / 4);
        myE = Math.trunc((4 * myE + nfCount * nfEff) / 4);
    }

    pDelta += myP;
    qDelta += myQ;
    eDelta += myE;

    const statScore =
        scoreStat(pDelta, currentP, params.w[0], params.target[0], params.hasTarget[0], params.maximize[0])
        + scoreStat(qDelta, currentQ, params.w[1], params.target[1], params.hasTarget[1], params.maximize[1])
        + scoreStat(eDelta, currentE, params.w[2], params.target[2], params.hasTarget[2], params.maximize[2]);

    out.ok = true;
    out.major = (statScore <= 0 ? NO_SCORE_MAJOR : statScore) - negativeContacts * NEGATIVE_CONTACT_PENALTY;
    // A placement that earns nothing wants the nodes least: every node it touches is a slot a paying module could have had
    out.minor = statScore <= 0 ? -adjNodes : adjNodes;
};
