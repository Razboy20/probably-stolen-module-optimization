import type { GridTier, InventoryItem, ItemEffect, ModuleColor, ModuleShape, Stats, TargetStats } from '../types';
import { getBaseStats } from '../utils';
import { MODULE_TEMPLATES } from '../constants';
import { type Board, initializeBoard } from './board';

const SHAPE_MAP: ModuleShape[] = ['Node1x2', 'L3', 'L4_Base', 'T4_Base', 'Square4_Base', 'L4_High', 'T4_High', 'Square4_High', 'P5', 'C5', 'Line4'];
const COLOR_MAP_KEYS: ModuleColor[] = ['White', 'Red', 'Yellow', 'Green', 'Purple', 'DarkRed', 'Grey'];
const EFFECT_MAP: ItemEffect[] = ['None', 'Premium', 'Inferior', 'Overcharged', 'Degrading', 'Negative Feedback', 'Receiver', 'Side Mount', 'Top Mount', 'Learning Algorithm'];

const BASE85_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

function encodeBase85(bytes: Uint8Array): string {
    let num = 1n;
    for (let i = 0; i < bytes.length; i++) {
        num = (num << 8n) | BigInt(bytes[i]);
    }
    let str = "";
    while (num > 0n) {
        str = BASE85_ALPHABET[Number(num % 85n)] + str;
        num /= 85n;
    }
    return str;
}

function decodeBase85(str: string): Uint8Array {
    str = str.trim();
    let num = 0n;
    for (let i = 0; i < str.length; i++) {
        const val = BASE85_ALPHABET.indexOf(str[i]);
        if (val === -1) throw new Error("Invalid base85 character");
        num = (num * 85n) + BigInt(val);
    }
    let hex = num.toString(16);
    if (hex.length % 2 !== 0) hex = '0' + hex;
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < bytes.length; i++) {
        bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
    }
    return bytes.slice(1);
}

class BitWriter {
    bytes: number[] = [];
    currentByte = 0;
    bitPos = 0;

    write(value: number, numBits: number) {
        for (let i = numBits - 1; i >= 0; i--) {
            const bit = (value >> i) & 1;
            this.currentByte = (this.currentByte << 1) | bit;
            this.bitPos++;
            if (this.bitPos === 8) {
                this.bytes.push(this.currentByte);
                this.currentByte = 0;
                this.bitPos = 0;
            }
        }
    }

    toBase85(): string {
        if (this.bitPos > 0) {
            this.bytes.push(this.currentByte << (8 - this.bitPos));
        }
        return encodeBase85(new Uint8Array(this.bytes));
    }
}

class BitReader {
    bytes: Uint8Array;
    bytePos = 0;
    bitPos = 7;

    constructor(base85: string) {
        this.bytes = decodeBase85(base85);
    }

    read(numBits: number): number {
        let value = 0;
        for (let i = 0; i < numBits; i++) {
            if (this.bytePos >= this.bytes.length) throw new Error("EOF");
            const bit = (this.bytes[this.bytePos] >> this.bitPos) & 1;
            value = (value << 1) | bit;
            this.bitPos--;
            if (this.bitPos < 0) {
                this.bitPos = 7;
                this.bytePos++;
            }
        }
        return value;
    }
}

// The code format stores the module count in 8 bits,
// so a larger inventory would encode a wrapped count and produce a code that decodes into something else entirely
// Emit nothing rather than something corrupt.
export const MAX_ENCODABLE_MODULES = 255;

export const generateCodeFromState = (
    currentTier: GridTier,
    maxStats: Record<keyof Stats, boolean>,
    tarStats: TargetStats,
    inv: InventoryItem[],
    brd: Board
) => {
    if (inv.length > MAX_ENCODABLE_MODULES) return '';

    const writer = new BitWriter();

    writer.write(currentTier, 2);

    writer.write(maxStats.Performance ? 1 : 0, 1);
    writer.write(maxStats.Quality ? 1 : 0, 1);
    writer.write(maxStats.Efficiency ? 1 : 0, 1);

    const writeTarget = (val: number | null) => {
        if (val === null) {
            writer.write(0, 1);
        } else {
            writer.write(1, 1);
            writer.write(val + 2048, 12);
        }
    };
    writeTarget(tarStats.Performance);
    writeTarget(tarStats.Quality);
    writeTarget(tarStats.Efficiency);

    writer.write(inv.length, 8);

    const placedItemsMap = new Map<string, number[]>();
    brd.forEach((row, y) => row.forEach((cell, x) => {
        if (cell && cell !== 'Locked') {
            if (!placedItemsMap.has(cell.id)) {
                placedItemsMap.set(cell.id, []);
            }
            placedItemsMap.get(cell.id)!.push(y * 7 + x);
        }
    }));

    inv.forEach(item => {
        writer.write(SHAPE_MAP.indexOf(item.shape), 4);
        writer.write(COLOR_MAP_KEYS.indexOf(item.color), 3);

        const positions = placedItemsMap.get(item.id) || [];
        writer.write(positions.length, 3);
        positions.forEach(p => writer.write(p, 6));

        item.effects.forEach((eff, idx) => {
            if (eff === 'None') {
                writer.write(0, 1);
            } else {
                writer.write(1, 1);
                writer.write(EFFECT_MAP.indexOf(eff), 4);

                if (eff === 'Learning Algorithm' || eff === 'Degrading') {
                    writer.write(1, 1);
                    writer.write(item.effectValues[idx] + 2048, 12);
                } else {
                    writer.write(0, 1);
                }
            }
        });
    });

    return writer.toBase85();
};

// Every code lists the modules the machine is not using too, except unused infinite clones, which would only pad it out
export const inventoryForCode = (inventory: InventoryItem[], board: Board) => {
    const usedCloneIds = new Set<string>();
    board.forEach(row => row.forEach(cell => {
        if (cell && cell !== 'Locked' && cell.id.includes('_clone_')) {
            usedCloneIds.add(cell.id);
        }
    }));
    return inventory.filter(item => !item.id.includes('_clone_') || usedCloneIds.has(item.id));
};

export interface DecodedSolution {
    tier: GridTier;
    maximizeStats: Record<keyof Stats, boolean>;
    targetStats: TargetStats;
    inventory: InventoryItem[];
    board: Board;
}

export const decodeSolution = (code: string): DecodedSolution => {
    const reader = new BitReader(code);
    const tier = reader.read(2) as GridTier;

    const maxP = reader.read(1) === 1;
    const maxQ = reader.read(1) === 1;
    const maxE = reader.read(1) === 1;
    const maximizeStats = { Performance: maxP, Quality: maxQ, Efficiency: maxE };

    const readTarget = () => {
        const hasTarget = reader.read(1) === 1;
        return hasTarget ? reader.read(12) - 2048 : null;
    };
    const targetStats = { Performance: readTarget(), Quality: readTarget(), Efficiency: readTarget() };

    const inventory: InventoryItem[] = [];
    const board = initializeBoard(tier);
    const numModules = reader.read(8);

    for (let i = 0; i < numModules; i++) {
        const shapeIdx = reader.read(4);
        const colorIdx = reader.read(3);
        const shape = SHAPE_MAP[shapeIdx];
        const color = COLOR_MAP_KEYS[colorIdx];

        const posCount = reader.read(3);
        const positions: number[] = [];
        for (let p = 0; p < posCount; p++) positions.push(reader.read(6));

        const template = shape === 'Node1x2' ? { displayName: 'Node' } : MODULE_TEMPLATES.find(m => m.shape === shape && m.color === color) || { displayName: 'Unknown Module' };
        const reconstructedEffects: [ItemEffect, ItemEffect] = ['None', 'None'];
        const base = getBaseStats({ shape, color, displayName: template.displayName });
        const maxBaseValue = Math.max(Math.abs(base.Performance), Math.abs(base.Quality), Math.abs(base.Efficiency));
        const reconstructedValues: [number, number] = [maxBaseValue * 2, maxBaseValue * 2];

        for (let eIdx = 0; eIdx < 2; eIdx++) {
            const hasEffect = reader.read(1) === 1;
            if (hasEffect) {
                const effIdx = reader.read(4);
                const eff = EFFECT_MAP[effIdx];
                reconstructedEffects[eIdx] = eff;

                const hasValue = reader.read(1) === 1;
                if ((eff === 'Learning Algorithm' || eff === 'Degrading') && hasValue) {
                    reconstructedValues[eIdx] = reader.read(12) - 2048;
                }
            }
        }

        const newItem: InventoryItem = { id: `${shape}_${color}_${Math.random().toString(36).substring(2, 8)}`, shape, color, displayName: template.displayName, effects: reconstructedEffects, effectValues: reconstructedValues };
        inventory.push(newItem);
        positions.forEach((pos: number) => board[Math.floor(pos / 7)][pos % 7] = newItem);
    }

    return { tier, maximizeStats, targetStats, inventory, board };
};
