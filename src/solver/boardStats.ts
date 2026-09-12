import type { InventoryItem, ItemEffect, Stats } from '../types';
import { applyInternalEffects, roundStat } from '../utils';
import { type Board, NEIGHBOR_DX, NEIGHBOR_DY } from './board';

// effects is a fixed 2-slot tuple
// counting it directly avoids the closure + intermediate array that Array.prototype.filter allocates on every call.
export const countEffect = (item: InventoryItem, effect: ItemEffect) => {
    let n = 0;
    if (item.effects[0] === effect) n++;
    if (item.effects[1] === effect) n++;
    return n;
};

interface PlacedPiece {
    item: InventoryItem;
    minX: number;
    minY: number;
    cells: number[];
    adjNodes: number;
}

export const indexInventoryById = (inventory: InventoryItem[]) => {
    const byId = new Map<string, InventoryItem>();
    for (const item of inventory) byId.set(item.id, item);
    return byId;
};

export interface BoardStats {
    totals: Stats;
    pieceStats: Map<string, Stats>;
    placedPiecesCount: number;
}

// `inventoryById` and `internalStats` are only optimisations:
// building them costs one pass over the whole inventory,
// which the solver would otherwise repeat on every iteration even though a board never holds more than 35 cells
// Callers that already have them should pass them
export const calculateBoardStats = (
    currentBoard: Board,
    currentInventory: InventoryItem[],
    inventoryById?: Map<string, InventoryItem>,
    internalStats?: Map<string, Stats>
): BoardStats => {
    const totals: Stats = { Performance: 0, Quality: 0, Efficiency: 0 };
    const pieceStats = new Map<string, Stats>();
    let placedPiecesCount = 0;

    const invById = inventoryById ?? indexInventoryById(currentInventory);

    // applyInternalEffects is pure per item but was recomputed for every adjacency test; memoise it for the duration of the call
    const internalCache = internalStats ?? new Map<string, Stats>();
    const internalOf = (item: InventoryItem): Stats => {
        let cached = internalCache.get(item.id);
        if (cached === undefined) {
            cached = applyInternalEffects(item);
            internalCache.set(item.id, cached);
        }
        return cached;
    };

    const placedPieces = new Map<string, PlacedPiece>();
    const nodeAdjacencies = new Map<string, Set<string>>();
    const gridItems: (InventoryItem | null)[] = new Array(35).fill(null);

    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 7; x++) {
            const boardCell = currentBoard[y][x];
            if (boardCell && boardCell !== 'Locked') {
                const cell = invById.get(boardCell.id) || boardCell;
                const idx = y * 7 + x;
                gridItems[idx] = cell;

                const existing = placedPieces.get(cell.id);
                if (existing === undefined) {
                    placedPieces.set(cell.id, { item: cell, minX: x, minY: y, cells: [idx], adjNodes: 0 });
                    placedPiecesCount++;
                } else {
                    if (x < existing.minX) existing.minX = x;
                    if (y < existing.minY) existing.minY = y;
                    existing.cells.push(idx);
                }
            }
        }
    }

    for (const { item, cells } of placedPieces.values()) {
        if (item.color !== 'White') continue;

        let adjSet = nodeAdjacencies.get(item.id);
        if (adjSet === undefined) {
            adjSet = new Set<string>();
            nodeAdjacencies.set(item.id, adjSet);
        }

        for (const idx of cells) {
            const x = idx % 7;
            const y = (idx - x) / 7;
            for (let d = 0; d < 4; d++) {
                const nx = x + NEIGHBOR_DX[d];
                const ny = y + NEIGHBOR_DY[d];
                if (nx < 0 || nx >= 7 || ny < 0 || ny >= 5) continue;

                const adj = gridItems[ny * 7 + nx];
                if (adj === null || adj.color === 'White') continue;

                if (!adjSet.has(adj.id)) {
                    adjSet.add(adj.id);
                    const adjPiece = placedPieces.get(adj.id);
                    if (adjPiece !== undefined) adjPiece.adjNodes++;
                }
            }
        }
    }

    const absorptionStats = new Map<string, Stats>();
    for (const { item, cells } of placedPieces.values()) {
        if (item.color === 'White') continue;

        let absorbP = 0, absorbQ = 0, absorbE = 0;

        const nfCount = countEffect(item, 'Negative Feedback');
        if (nfCount > 0) {
            let nfPerf = 0, nfQual = 0, nfEff = 0;
            const counted = new Set<string>();

            for (const idx of cells) {
                const x = idx % 7;
                const y = (idx - x) / 7;
                for (let d = 0; d < 4; d++) {
                    const nx = x + NEIGHBOR_DX[d];
                    const ny = y + NEIGHBOR_DY[d];
                    if (nx < 0 || nx >= 7 || ny < 0 || ny >= 5) continue;

                    const neighbor = gridItems[ny * 7 + nx];
                    if (neighbor === null || neighbor.id === item.id || neighbor.color === 'White') continue;
                    if (counted.has(neighbor.id)) continue;
                    counted.add(neighbor.id);

                    const neighborBase = internalOf(neighbor);
                    if (neighborBase.Performance < 0) nfPerf += neighborBase.Performance;
                    if (neighborBase.Quality < 0) nfQual += neighborBase.Quality;
                    if (neighborBase.Efficiency < 0) nfEff += neighborBase.Efficiency;
                }
            }

            absorbP = nfCount * 0.25 * nfPerf;
            absorbQ = nfCount * 0.25 * nfQual;
            absorbE = nfCount * 0.25 * nfEff;
        }

        absorptionStats.set(item.id, {
            Performance: absorbP,
            Quality: absorbQ,
            Efficiency: absorbE
        });
    }

    for (const { item, minX, minY, adjNodes } of placedPieces.values()) {
        if (item.color === 'White') continue;

        let { Performance: p, Quality: q, Efficiency: e } = internalOf(item);

        let pBonus = 0, qBonus = 0, eBonus = 0;
        if (minX === 0 && item.effects.includes('Side Mount')) {
            pBonus += roundStat(p * 0.20);
            qBonus += roundStat(q * 0.20);
            eBonus += roundStat(e * 0.20);
        }
        if (minY === 0 && item.effects.includes('Top Mount')) {
            pBonus += roundStat(p * 0.20);
            qBonus += roundStat(q * 0.20);
            eBonus += roundStat(e * 0.20);
        }
        if (item.effects.includes('Receiver')) {
            pBonus += roundStat(p * 0.10 * adjNodes);
            qBonus += roundStat(q * 0.10 * adjNodes);
            eBonus += roundStat(e * 0.10 * adjNodes);
        }

        p += pBonus;
        q += qBonus;
        e += eBonus;

        const absorb = absorptionStats.get(item.id)!;
        p += absorb.Performance;
        q += absorb.Quality;
        e += absorb.Efficiency;

        const finalStats = { Performance: roundStat(p), Quality: roundStat(q), Efficiency: roundStat(e) };

        pieceStats.set(item.id, finalStats);
        totals.Performance += finalStats.Performance;
        totals.Quality += finalStats.Quality;
        totals.Efficiency += finalStats.Efficiency;
    }

    for (const [nodeId, adjIds] of nodeAdjacencies) {
        let nodeP = 0, nodeQ = 0, nodeE = 0;

        for (const adjId of adjIds) {
            const adjacentItemData = placedPieces.get(adjId);
            if (adjacentItemData) {
                const baseAdj = internalOf(adjacentItemData.item);
                nodeP += baseAdj.Performance;
                nodeQ += baseAdj.Quality;
                nodeE += baseAdj.Efficiency;
            }
        }

        const nodeStat = {
            Performance: roundStat(nodeP * 0.20),
            Quality: roundStat(nodeQ * 0.20),
            Efficiency: roundStat(nodeE * 0.20)
        };

        pieceStats.set(nodeId, nodeStat);
        totals.Performance += nodeStat.Performance;
        totals.Quality += nodeStat.Quality;
        totals.Efficiency += nodeStat.Efficiency;
    }

    return { totals, pieceStats, placedPiecesCount };
};
