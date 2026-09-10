import type { InventoryItem, Stats } from '../types';
import { applyInternalEffects, PRECOMPUTED_ORIENTATIONS, roundStat } from '../utils';
import type { Board } from './board';
import { countEffect } from './boardStats';
import { MAX_PIECE_NEIGHBORS, ORIENT_COUNT, ORIENT_START, shapeIndexOf } from './geometry';
import { isSpecialModule } from './pool';

export const FLAG_WHITE = 1;
export const FLAG_SIDE_MOUNT = 2;
export const FLAG_TOP_MOUNT = 4;
export const FLAG_RECEIVER = 8;
// A special or user-locked module: the search moves it but never adds or removes one
export const FLAG_FIXED = 16;
export const FLAG_PURE_NEGATIVE = 32;
export const NF_SHIFT = 6;
export const nfCountOf = (flags: number) => (flags >> NF_SHIFT) & 3;

export const RECV_MAX_NODES = MAX_PIECE_NEIGHBORS;
const RECV_STRIDE = RECV_MAX_NODES + 1;

/* Everything the search needs to know about a module, addressed by a dense item index
 * The percentages the board rules apply (20% for edge mounts and node contact, 10% per adjacent node for a Receiver) are rounded per module,
 * so they are computed here once, with the same rounding the report uses, and the search only ever adds integers
 */
export interface PoolTables {
    count: number;
    items: InventoryItem[];
    indexOf: Map<string, number>;
    internal: Map<string, Stats>;
    p: Int32Array;
    q: Int32Array;
    e: Int32Array;
    p20: Int32Array;
    q20: Int32Array;
    e20: Int32Array;
    flags: Int32Array;
    shape: Int32Array;
    size: Int32Array;
    orientStart: Int32Array;
    orientCount: Int32Array;
    // Receiver bonus per stat and adjacent node count; -1 for modules without one
    recvSlot: Int32Array;
    recvTable: Int32Array;
}

export const recvBonus = (tables: PoolTables, slot: number, stat: number, adjNodes: number) =>
    tables.recvTable[(slot * 3 + stat) * RECV_STRIDE + adjNodes];

const collectItems = (inventory: InventoryItem[], board: Board) => {
    const items = [...inventory];
    const seen = new Set(items.map(item => item.id));
    for (const row of board) {
        for (const cell of row) {
            if (cell && cell !== 'Locked' && !seen.has(cell.id)) {
                seen.add(cell.id);
                items.push(cell);
            }
        }
    }
    return items;
};

export const buildPoolTables = (inventory: InventoryItem[], board: Board): PoolTables => {
    const items = collectItems(inventory, board);
    const count = items.length;
    const indexOf = new Map<string, number>();
    const internal = new Map<string, Stats>();
    const p = new Int32Array(count), q = new Int32Array(count), e = new Int32Array(count);
    const p20 = new Int32Array(count), q20 = new Int32Array(count), e20 = new Int32Array(count);
    const flags = new Int32Array(count);
    const shape = new Int32Array(count);
    const size = new Int32Array(count);
    const orientStart = new Int32Array(count);
    const orientCount = new Int32Array(count);
    const recvSlot = new Int32Array(count).fill(-1);
    const recvRows: number[] = [];

    for (let i = 0; i < count; i++) {
        const item = items[i];
        indexOf.set(item.id, i);
        const stats = applyInternalEffects(item);
        internal.set(item.id, stats);
        const { Performance: ip, Quality: iq, Efficiency: ie } = stats;
        if (!Number.isInteger(ip) || !Number.isInteger(iq) || !Number.isInteger(ie)) {
            throw new Error(`Module ${item.id} has non-integer stats`);
        }
        p[i] = ip; q[i] = iq; e[i] = ie;
        p20[i] = roundStat(ip * 0.20); q20[i] = roundStat(iq * 0.20); e20[i] = roundStat(ie * 0.20);

        let f = 0;
        if (item.color === 'White') f |= FLAG_WHITE;
        if (item.effects.includes('Side Mount')) f |= FLAG_SIDE_MOUNT;
        if (item.effects.includes('Top Mount')) f |= FLAG_TOP_MOUNT;
        if (item.effects.includes('Receiver')) f |= FLAG_RECEIVER;
        if (isSpecialModule(item) || item.isLocked) f |= FLAG_FIXED;
        if (ip <= 0 && iq <= 0 && ie <= 0 && (ip < 0 || iq < 0 || ie < 0)) f |= FLAG_PURE_NEGATIVE;
        f |= countEffect(item, 'Negative Feedback') << NF_SHIFT;
        flags[i] = f;

        const s = shapeIndexOf(item.shape);
        shape[i] = s;
        orientStart[i] = ORIENT_START[s];
        orientCount[i] = ORIENT_COUNT[s];
        size[i] = PRECOMPUTED_ORIENTATIONS.get(item.shape)?.[0].count ?? 0;

        if (f & FLAG_RECEIVER) {
            recvSlot[i] = recvRows.length / (3 * RECV_STRIDE);
            for (const v of [ip, iq, ie]) {
                for (let n = 0; n < RECV_STRIDE; n++) recvRows.push(roundStat(v * 0.10 * n));
            }
        }
    }

    return {
        count, items, indexOf, internal, p, q, e, p20, q20, e20, flags, shape, size,
        orientStart, orientCount, recvSlot, recvTable: Int32Array.from(recvRows)
    };
};
