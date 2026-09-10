import type { InventoryItem, Stats } from '../types';
import { type Orientation, PRECOMPUTED_ORIENTATIONS, roundStat } from '../utils';
import { type Board, NEIGHBOR_DX, NEIGHBOR_DY } from './board';
import { countEffect } from './boardStats';
import { type MachineConfig, statIsIgnored } from './objective';

export interface PlacementContext {
    piece: InventoryItem;
    internal: Stats;
    isWhite: boolean;
    hasSideMount: boolean;
    hasTopMount: boolean;
    hasReceiver: boolean;
    nfCount: number;
    isPureNegative: boolean;
    size: number;
}

export const buildPlacementContext = (piece: InventoryItem, precomputedInternal: Map<string, Stats>): PlacementContext => {
    const internal = precomputedInternal.get(piece.id)!;
    const { Performance: p, Quality: q, Efficiency: e } = internal;
    const orientations = PRECOMPUTED_ORIENTATIONS.get(piece.shape);

    return {
        piece,
        internal,
        isWhite: piece.color === 'White',
        hasSideMount: piece.effects.includes('Side Mount'),
        hasTopMount: piece.effects.includes('Top Mount'),
        hasReceiver: piece.effects.includes('Receiver'),
        nfCount: countEffect(piece, 'Negative Feedback'),
        isPureNegative: p <= 0 && q <= 0 && e <= 0 && (p < 0 || q < 0 || e < 0),
        size: orientations ? orientations[0].count : 0
    };
};

// Scratch buffer for the distinct neighbours of a candidate placement
// A piece covers at most 5 cells, so the neighbour count is small and bounded
// reusing one array keeps this function allocation-free, which matters because it is the solver's innermost loop
const NEIGHBOR_SCRATCH: InventoryItem[] = [];

export const evaluatePlacementDelta = (
    ctx: PlacementContext,
    x: number, y: number,
    orientation: Orientation,
    testBoard: Board,
    isBoardEmpty: boolean,
    precomputedInternal: Map<string, Stats>,
    dynWp: number, dynWq: number, dynWe: number,
    currentP: number, currentQ: number, currentE: number,
    config: MachineConfig
) => {
    if (x + orientation.minX < 0 || x + orientation.maxX > 6 ||
        y + orientation.minY < 0 || y + orientation.maxY > 4) {
        return -Infinity;
    }

    const { xs, ys, count } = orientation;
    const { internal, isWhite, nfCount, isPureNegative } = ctx;

    let isConnected = false;
    let adjNodes = 0;
    let negativeContactCount = 0;
    let neighborCount = 0;

    for (let i = 0; i < count; i++) {
        const px = x + xs[i];
        const py = y + ys[i];

        if (testBoard[py][px] !== null) return -Infinity;
        if (px === 0 || px === 6 || py === 0 || py === 4) isConnected = true;

        for (let d = 0; d < 4; d++) {
            const nx = px + NEIGHBOR_DX[d];
            const ny = py + NEIGHBOR_DY[d];
            if (nx < 0 || nx >= 7 || ny < 0 || ny >= 5) continue;

            const adjCell = testBoard[ny][nx];
            if (!adjCell || adjCell === 'Locked') continue;

            isConnected = true;

            const adjIsWhite = adjCell.color === 'White';
            if (isWhite) {
                if (!adjIsWhite) {
                    const adjInt = precomputedInternal.get(adjCell.id);
                    if (adjInt !== undefined &&
                        adjInt.Performance <= 0 && adjInt.Quality <= 0 && adjInt.Efficiency <= 0 &&
                        (adjInt.Performance < 0 || adjInt.Quality < 0 || adjInt.Efficiency < 0)) {
                        negativeContactCount++;
                    }
                }
            } else if (isPureNegative && adjIsWhite) {
                negativeContactCount++;
            }

            let seen = false;
            for (let k = 0; k < neighborCount; k++) {
                if (NEIGHBOR_SCRATCH[k].id === adjCell.id) { seen = true; break; }
            }
            if (!seen) NEIGHBOR_SCRATCH[neighborCount++] = adjCell;
        }
    }

    if (!isConnected && !isBoardEmpty) return -10000;

    let pDelta = 0, qDelta = 0, eDelta = 0;
    let nfPerf = 0, nfQual = 0, nfEff = 0;

    for (let k = 0; k < neighborCount; k++) {
        const adjPiece = NEIGHBOR_SCRATCH[k];
        const adjInternal = precomputedInternal.get(adjPiece.id);
        if (adjInternal === undefined) continue;

        const adjIsWhite = adjPiece.color === 'White';
        if (!isWhite && adjIsWhite) {
            adjNodes++;
            pDelta += roundStat(internal.Performance * 0.20);
            qDelta += roundStat(internal.Quality * 0.20);
            eDelta += roundStat(internal.Efficiency * 0.20);
        } else if (isWhite && !adjIsWhite) {
            pDelta += roundStat(adjInternal.Performance * 0.20);
            qDelta += roundStat(adjInternal.Quality * 0.20);
            eDelta += roundStat(adjInternal.Efficiency * 0.20);
        }

        if (nfCount > 0 && !adjIsWhite) {
            if (adjInternal.Performance < 0) nfPerf += adjInternal.Performance;
            if (adjInternal.Quality < 0) nfQual += adjInternal.Quality;
            if (adjInternal.Efficiency < 0) nfEff += adjInternal.Efficiency;
        }
    }

    let myP = internal.Performance;
    let myQ = internal.Quality;
    let myE = internal.Efficiency;

    let pBonus = 0, qBonus = 0, eBonus = 0;
    if (ctx.hasSideMount && x + orientation.minX === 0) {
        pBonus += roundStat(myP * 0.20);
        qBonus += roundStat(myQ * 0.20);
        eBonus += roundStat(myE * 0.20);
    }
    if (ctx.hasTopMount && y + orientation.minY === 0) {
        pBonus += roundStat(myP * 0.20);
        qBonus += roundStat(myQ * 0.20);
        eBonus += roundStat(myE * 0.20);
    }
    if (ctx.hasReceiver) {
        pBonus += roundStat(myP * 0.10 * adjNodes);
        qBonus += roundStat(myQ * 0.10 * adjNodes);
        eBonus += roundStat(myE * 0.10 * adjNodes);
    }

    myP += pBonus;
    myQ += qBonus;
    myE += eBonus;

    if (nfCount > 0) {
        myP += nfCount * 0.25 * nfPerf;
        myQ += nfCount * 0.25 * nfQual;
        myE += nfCount * 0.25 * nfEff;
    }

    pDelta += roundStat(myP);
    qDelta += roundStat(myQ);
    eDelta += roundStat(myE);

    let statScore = 0;
    const scoreStat = (key: keyof Stats, delta: number, current: number, dynW: number) => {
        if (statIsIgnored(config, key)) return 0;
        if (delta === 0) return 0;

        const target = config.targetStats[key];
        const maximize = config.maximizeStats[key];

        if (target !== null && !maximize) {
            const before = current;
            const after = current + delta;

            if (before >= target && after >= target) return 0;
            if (before >= target && after < target) return delta * dynW * 100;
            if (before < target) {
                if (after <= target) return delta * dynW;
                else return (target - before) * dynW;
            }
        } else if (target !== null && maximize) {
            const before = current;
            const after = current + delta;
            if (before >= target) {
                return delta * dynW;
            } else {
                if (after <= target) return delta * dynW * 10;
                else return ((target - before) * dynW * 10) + ((after - target) * dynW);
            }
        } else if (maximize) {
            return delta * dynW;
        }
        return 0;
    };

    statScore += scoreStat('Performance', pDelta, currentP, dynWp);
    statScore += scoreStat('Quality', qDelta, currentQ, dynWq);
    statScore += scoreStat('Efficiency', eDelta, currentE, dynWe);

    const tiebreakers = (adjNodes * 0.05) - (negativeContactCount * 1000);
    if (statScore <= 0) return -10000 + tiebreakers;
    return statScore + tiebreakers;
};
