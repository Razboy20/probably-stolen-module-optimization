import type { GridTier, InventoryItem, ItemEffect, ModuleColor, ModuleShape, Stats, TargetStats } from '../types';
import { getBaseStats } from '../utils';
import { EVAPORATOR_GRID_SIZE, MODULE_TEMPLATES, NODE_TEMPLATE, STANDARD_GRID_SIZE } from '../constants';
import { type Board, initializeBoard, isEvaporatorBoard } from './board';
import { isLockedModule } from './locked';

const SHAPE_MAP: ModuleShape[] = ['Node1x2', 'L3', 'L4_Base', 'T4_Base', 'Square4_Base', 'L4_High', 'T4_High', 'Square4_High', 'P5', 'C5', 'Line4', 'Line1x2'];
const COLOR_MAP_KEYS: ModuleColor[] = ['White', 'Red', 'Yellow', 'Green', 'Purple', 'DarkRed', 'Grey'];
// Blast and Junk Processing furnaces are both grey Line4s, so Blast takes the one colour index the three bits leave unused
const BLAST_COLOR_INDEX = 7;
const BLAST_NAME = 'Furnace Module (Blast)';
// The specials the three colour bits cannot name are written as grey; the identity section after the lock state says which they really are
const FALLBACK_COLOR_INDEX = COLOR_MAP_KEYS.indexOf('Grey');
const EFFECT_MAP: ItemEffect[] = ['None', 'Premium', 'Inferior', 'Overcharged', 'Degrading', 'Negative Feedback', 'Receiver', 'Side Mount', 'Top Mount', 'Learning Algorithm'];

// The special module numbering of the upstream site's version 1 codes, which the identity section reuses so the two never disagree
const SPECIAL_MODULE_IDS: string[] = [
    'Neural Core Module (Uncapped)',
    'Neural Core Module (Capped)',
    'Alarm Transmitter Module',
    'Furnace Module (Junk Processing)',
    'Furnace Module (Blast)',
    'Evaporator Solvent Recovery Module',
    'Dream Cap Neutralizer Hydroponic Module',
    'Dream Cap Nightmare Hydroponic Module',
    'Hydroponic Smuggler Module',
    'Kotton Auto Loom Hydroponic Module',
    'Chemical Equipment Module',
    'Negative Trait Filter Module',
    'Positive Trait Filter Module',
    'Precision Extraction Module',
    'Genetic Stabilizer Module',
    'Filterweed Production Hydroponic Module',
    'Hydroreed Filter Hydroponic Module',
    'Nutrifruit Hydroponic Module',
    'Computer Numerical Control Module',
    'Melee Batch Module',
    'Metal Printer Module',
    'Glitched Module'
];

// Upstream version 1 codes open with two zero bits where a tier (1-3) would otherwise be, and number shapes and colours their own way
const UPSTREAM_CODE_VERSION = 1;
const UPSTREAM_SHAPE_MAP: ModuleShape[] = ['Node1x2', 'Line1x2', 'L3', 'L4_Base', 'T4_Base', 'Square4_Base', 'L4_High', 'T4_High', 'Square4_High', 'P5', 'C5', 'Line4'];
const UPSTREAM_COLOR_MAP: ModuleColor[] = ['White', 'Red', 'Yellow', 'Green', 'Purple', 'DarkRed', 'Grey', 'Blue', 'Gold', 'Brown', 'DarkGreen', 'DarkPurple'];

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

    hasBits(numBits: number) {
        return (this.bytes.length - this.bytePos) * 8 - (7 - this.bitPos) >= numBits;
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
        writer.write(legacyColorIndex(item), 3);

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

    // Lock state trails the modules behind a set marker bit, so older decoders never read it and older codes, padded with zeros, read as having none
    writer.write(1, 1);
    inv.forEach(item => writer.write(isLockedModule(item) ? 1 : 0, 1));

    // Then, behind a marker of its own, the grid and what each module is beyond its shape and colour
    writer.write(1, 1);
    writer.write(isEvaporatorBoard(brd) ? 1 : 0, 1);
    inv.forEach(item => {
        writer.write(item.isStuck ? 1 : 0, 1);
        const specialIdx = SPECIAL_MODULE_IDS.indexOf(item.displayName);
        writer.write(specialIdx >= 0 ? 1 : 0, 1);
        if (specialIdx >= 0) writer.write(specialIdx, 6);
    });

    return writer.toBase85();
};

const legacyColorIndex = (item: InventoryItem) => {
    if (item.displayName === BLAST_NAME) return BLAST_COLOR_INDEX;
    const idx = COLOR_MAP_KEYS.indexOf(item.color);
    return idx >= 0 ? idx : FALLBACK_COLOR_INDEX;
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
    isEvaporator: boolean;
    maximizeStats: Record<keyof Stats, boolean>;
    targetStats: TargetStats;
    inventory: InventoryItem[];
    board: Board;
}

interface DecodedModule {
    item: InventoryItem;
    // Row-major cells of the standard grid
    positions: number[];
}

interface DecodedModules {
    tier: GridTier;
    isEvaporator: boolean;
    maximizeStats: Record<keyof Stats, boolean>;
    targetStats: TargetStats;
    modules: DecodedModule[];
}

const readObjective = (reader: BitReader) => {
    const maximizeStats = { Performance: reader.read(1) === 1, Quality: reader.read(1) === 1, Efficiency: reader.read(1) === 1 };
    const readTarget = () => reader.read(1) === 1 ? reader.read(12) - 2048 : null;
    const targetStats = { Performance: readTarget(), Quality: readTarget(), Efficiency: readTarget() };
    return { maximizeStats, targetStats };
};

const readPositions = (reader: BitReader) => {
    const count = reader.read(3);
    const positions: number[] = [];
    for (let p = 0; p < count; p++) positions.push(reader.read(6));
    return positions;
};

const templateName = (shape: ModuleShape, color: ModuleColor) =>
    color === 'White' ? NODE_TEMPLATE.displayName
        : MODULE_TEMPLATES.find(m => m.shape === shape && m.color === color)?.displayName ?? 'Unknown Module';

const specialTemplate = (idx: number) => {
    const template = MODULE_TEMPLATES.find(t => t.displayName === SPECIAL_MODULE_IDS[idx]);
    if (!template) throw new Error('Unknown special module');
    return template;
};

// The effects close out every module in both formats
const readModule = (reader: BitReader, shape: ModuleShape, color: ModuleColor, displayName: string): InventoryItem => {
    const effects: [ItemEffect, ItemEffect] = ['None', 'None'];
    const base = getBaseStats({ shape, color, displayName });
    const maxPositiveBase = Math.max(base.Performance, base.Quality, base.Efficiency, 0);
    const effectValues: [number, number] = [maxPositiveBase * 2, maxPositiveBase * 2];

    for (let eIdx = 0; eIdx < 2; eIdx++) {
        if (reader.read(1) !== 1) continue;
        const eff = EFFECT_MAP[reader.read(4)];
        effects[eIdx] = eff;
        const hasValue = reader.read(1) === 1;
        if ((eff === 'Learning Algorithm' || eff === 'Degrading') && hasValue) {
            effectValues[eIdx] = reader.read(12) - 2048;
        }
    }

    return { id: `${shape}_${color}_${Math.random().toString(36).substring(2, 8)}`, shape, color, displayName, effects, effectValues };
};

const decodeOwnFormat = (reader: BitReader, tier: GridTier): DecodedModules => {
    const { maximizeStats, targetStats } = readObjective(reader);
    const modules: DecodedModule[] = [];
    const numModules = reader.read(8);

    for (let i = 0; i < numModules; i++) {
        const shapeIdx = reader.read(4);
        const colorIdx = reader.read(3);
        const isBlast = shapeIdx === SHAPE_MAP.indexOf('Line4') && colorIdx === BLAST_COLOR_INDEX;
        const shape = SHAPE_MAP[shapeIdx];
        const color = isBlast ? 'Grey' : COLOR_MAP_KEYS[colorIdx];
        if (shape === undefined || color === undefined) throw new Error('Invalid module');

        const positions = readPositions(reader);
        const item = readModule(reader, shape, color, isBlast ? BLAST_NAME : templateName(shape, color));
        modules.push({ item, positions });
    }

    if (reader.hasBits(1 + numModules) && reader.read(1) === 1) {
        for (const { item } of modules) item.isLocked = reader.read(1) === 1;
    }

    let isEvaporator = false;
    if (reader.hasBits(2 + 2 * numModules) && reader.read(1) === 1) {
        isEvaporator = reader.read(1) === 1;
        for (const { item } of modules) {
            if (reader.read(1) === 1) item.isStuck = true;
            if (reader.read(1) === 1) {
                const template = specialTemplate(reader.read(6));
                item.shape = template.shape;
                item.color = template.color;
                item.displayName = template.displayName;
            }
        }
    }

    return { tier, isEvaporator, maximizeStats, targetStats, modules };
};

const decodeUpstreamFormat = (reader: BitReader): DecodedModules => {
    if (reader.read(4) !== UPSTREAM_CODE_VERSION) throw new Error('Unsupported code version');
    const isEvaporator = reader.read(1) === 1;
    const tier = reader.read(2) as GridTier;
    const { maximizeStats, targetStats } = readObjective(reader);
    const cols = isEvaporator ? EVAPORATOR_GRID_SIZE.cols : STANDARD_GRID_SIZE.cols;
    const modules: DecodedModule[] = [];
    const numModules = reader.read(8);

    for (let i = 0; i < numModules; i++) {
        let shape: ModuleShape;
        let color: ModuleColor;
        let displayName: string;
        if (reader.read(1) === 1) {
            ({ shape, color, displayName } = specialTemplate(reader.read(6)));
        } else {
            shape = UPSTREAM_SHAPE_MAP[reader.read(4)];
            color = UPSTREAM_COLOR_MAP[reader.read(4)];
            if (shape === undefined || color === undefined) throw new Error('Invalid module');
            if (shape === 'Node1x2' && color !== 'White') shape = 'Line1x2';
            displayName = templateName(shape, color);
        }

        // Upstream numbers cells across the machine's own grid width
        const positions = readPositions(reader).map(pos => Math.floor(pos / cols) * STANDARD_GRID_SIZE.cols + pos % cols);
        const isLocked = reader.read(1) === 1;
        const isStuck = reader.read(1) === 1;
        const item = readModule(reader, shape, color, displayName);
        // Upstream never hands a special to its optimizer, so an unset lock bit leaves the special's own default in place
        if (isLocked) item.isLocked = true;
        if (isStuck) item.isStuck = true;
        modules.push({ item, positions });
    }

    return { tier, isEvaporator, maximizeStats, targetStats, modules };
};

export const decodeSolution = (code: string): DecodedSolution => {
    const reader = new BitReader(code);
    const header = reader.read(2);
    const { tier, isEvaporator, maximizeStats, targetStats, modules } = header === 0
        ? decodeUpstreamFormat(reader)
        : decodeOwnFormat(reader, header as GridTier);

    const board = initializeBoard(tier, undefined, undefined, isEvaporator);
    const cells = STANDARD_GRID_SIZE.rows * STANDARD_GRID_SIZE.cols;
    for (const { item, positions } of modules) {
        for (const pos of positions) {
            const y = Math.floor(pos / STANDARD_GRID_SIZE.cols);
            const x = pos % STANDARD_GRID_SIZE.cols;
            if (pos >= cells || board[y][x] === 'Locked') throw new Error('Invalid position');
            board[y][x] = item;
        }
    }

    return { tier, isEvaporator, maximizeStats, targetStats, inventory: modules.map(m => m.item), board };
};
