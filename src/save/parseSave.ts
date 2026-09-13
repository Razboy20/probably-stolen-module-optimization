import { PRECOMPUTED_OFFSETS, getBaseStats } from '../utils';
import type { InventoryItem, ModuleShape, ItemEffect, ModuleColor, Point, GridTier } from '../types';

export interface ImportedMachine {
    id: string;
    boardIds: (string | null)[][];
    machineType: string;
    tier: GridTier;
}

export interface ImportedSave {
    items: InventoryItem[];
    machines: ImportedMachine[];
}

interface SaveValue {
    '<identifier>k__BackingField'?: string;
    internalValueString?: string;
    valueInt?: number;
}

interface SaveShape {
    '<width>k__BackingField'?: number;
    '<height>k__BackingField'?: number;
    '<minX>k__BackingField'?: number;
    '<minY>k__BackingField'?: number;
    '<orientation>k__BackingField'?: number;
    '<flipped>k__BackingField'?: boolean;
    data?: number[];
}

interface SaveItem {
    uuid: number;
    identifier?: string;
    name?: string;
    itemTypes?: string[];
    childItems?: number[];
    itemShape?: SaveShape;
    itemModifiedShape?: SaveShape;
    _keys?: string[];
    _values?: SaveValue[];
    numberedName?: string;
}

const isHex = (c: string | undefined) => c !== undefined && /[0-9a-fA-F]/.test(c);

const isValidEscape = (json: string, backslashIdx: number) => {
    const next = json[backslashIdx + 1];
    if (next === undefined) return false;
    if (next === 'u') {
        return isHex(json[backslashIdx + 2]) && isHex(json[backslashIdx + 3]) &&
            isHex(json[backslashIdx + 4]) && isHex(json[backslashIdx + 5]);
    }
    return '"\\/bfnrt'.includes(next);
};

// Localized names reach the save with the backslashes their loc strings escaped
// quotes with, e.g. \“本源\”费洛蒙香水, which no JSON parser will accept.
const dropStrayEscapes = (json: string): string => {
    let out = '';
    let copiedTo = 0;

    for (let i = 0; i < json.length; i++) {
        if (json[i] !== '\\') continue;
        if (isValidEscape(json, i)) { i++; continue; }
        out += json.substring(copiedTo, i);
        copiedTo = i + 1;
    }

    return copiedTo === 0 ? json : out + json.substring(copiedTo);
};

const readSaveItems = (raw: string): SaveItem[] => {
    const text = dropStrayEscapes(raw);

    const keyIdx = text.indexOf('"mainInvJSON"');
    if (keyIdx === -1) throw new Error("Could not find mainInvJSON in save file.");

    const colonIdx = text.indexOf(':', keyIdx);
    const quoteStart = text.indexOf('"', colonIdx);

    let quoteEnd = -1;
    for (let i = quoteStart + 1; i < text.length; i++) {
        if (text[i] === '\\') i++;
        else if (text[i] === '"') { quoteEnd = i; break; }
    }

    if (quoteEnd === -1) throw new Error("Malformed mainInvJSON string.");

    const invStr = JSON.parse(text.substring(quoteStart, quoteEnd + 1)) as string;
    const invData = JSON.parse(dropStrayEscapes(invStr)) as { saveItems?: SaveItem[] };

    return invData.saveItems || [];
};

interface ModuleIdentity {
    color: ModuleColor;
    isHighTier: boolean;
    displayName: string;
}

// Item identifiers are the same in every localization, so they - not the displayed
// name - decide what a module is. The names they resolve to are the ones the rest
// of the app keys off (see MODULE_TEMPLATES, getBaseStats, isSpecialModule).
const identityFromIdentifier = (identifier: string): ModuleIdentity | null => {
    if (identifier.includes('node')) return { color: 'White', isHighTier: false, displayName: 'Node (Medium)' };
    if (identifier.includes('neural_core')) {
        const uncapped = identifier.includes('uncapped');
        return { color: 'Purple', isHighTier: uncapped, displayName: `Neural Core Module (${uncapped ? 'Uncapped' : 'Capped'})` };
    }
    if (identifier.includes('alarm')) return { color: 'DarkRed', isHighTier: false, displayName: 'Alarm Transmitter Module' };
    if (identifier.includes('junk')) return { color: 'Grey', isHighTier: false, displayName: 'Furnace Module (Junk Processing)' };
    if (identifier.includes('blast')) return { color: 'Grey', isHighTier: false, displayName: 'Furnace Module (Blast)' };
    if (identifier.includes('overclock')) return { color: 'Red', isHighTier: true, displayName: 'Overclock' };
    if (identifier.includes('performance')) return { color: 'Red', isHighTier: false, displayName: 'Performance' };
    if (identifier.includes('fineness')) return { color: 'Yellow', isHighTier: true, displayName: 'Refinement' };
    if (identifier.includes('quality')) return { color: 'Yellow', isHighTier: false, displayName: 'Quality' };
    if (identifier.includes('eco')) return { color: 'Green', isHighTier: true, displayName: 'Eco' };
    if (identifier.includes('efficianc') || identifier.includes('efficienc')) return { color: 'Green', isHighTier: false, displayName: 'Efficiency' };
    return null;
};

const identityFromName = (name: string): ModuleIdentity => {
    const lower = name.toLowerCase();
    const isHighTier = lower.includes('overclock') || lower.includes('refinement') || lower.includes('eco') || lower.includes('uncapped');

    let color: ModuleColor = 'White';
    if (lower.includes('performance') || lower.includes('overclock')) color = 'Red';
    else if (lower.includes('quality') || lower.includes('refinement')) color = 'Yellow';
    else if (lower.includes('efficiency') || lower.includes('eco')) color = 'Green';
    else if (lower.includes('neural core')) color = 'Purple';
    else if (lower.includes('alarm transmitter')) color = 'DarkRed';
    else if (lower.includes('(junk processing)') || lower.includes('(blast)')) color = 'Grey';

    return { color, isHighTier, displayName: name };
};

const hasModuleName = (item: SaveItem): boolean => {
    const lower = (item.name || '').toLowerCase();
    const typesStr = item.itemTypes?.join(' ').toUpperCase() || '';
    const allKeysString = ((item._keys || []).join(' ') + ' ' +
        ((item._values || []).map(v => v.internalValueString || '').join(' '))).toUpperCase();

    return (allKeysString.includes("MODULE") ||
            allKeysString.includes("BONUS_PERCENTAGE_PERFORMANCE_INT") && typesStr.includes("MODULE")) &&
        (lower.includes('module') || lower.includes('node') || lower.includes('core') ||
            lower.includes('alarm transmitter module') || lower.includes('(junk processing)') || lower.includes('(blast)'));
};

const isModule = (item: SaveItem): boolean => {
    const identifier = item.identifier || '';
    const lower = (item.name || '').toLowerCase();
    if (identifier.includes('ruined') || identifier.includes('corrupted')) return false;
    if (lower.includes('ruined') || lower.includes('corrupted')) return false;

    const types = (item.itemTypes || []).map(t => t.toUpperCase());
    return types.includes('MODULE') || types.includes('NODE') || hasModuleName(item);
};

const MACHINE_NAMES: [string, string][] = [
    ['purifier', 'Water Purifier'],
    ['furnace', 'Furnace'],
    ['moisture_farm', 'Moisture Farm'],
    ['wine_rack', 'AgeWell'],
    ['mirage_projector', 'Mirage Projector'],
    ['desequencer', 'Cryptographic Desequencer'],
    ['alarm', 'Alarm System'],
];

const machineNameFromIdentifier = (identifier: string): string | null =>
    MACHINE_NAMES.find(([keyword]) => identifier.includes(keyword))?.[1] ?? null;

const machineNameFromName = (name: string): string | null => {
    const lower = name.toLowerCase();
    if (lower.includes("bay") || lower.includes("box") || lower.includes("crate")) return null;
    if (lower.includes("purifier")) return "Water Purifier";
    if (lower.includes("furnace")) return "Furnace";
    if (lower.includes("farm")) return "Moisture Farm";
    if (lower.includes("agewell")) return "AgeWell";
    if (lower.includes("projector")) return "Mirage Projector";
    if (lower.includes("desequencer")) return "Cryptographic Desequencer";
    if (lower.includes("alarm")) return "Alarm System";
    return null;
};

// Only machines with a module grid count; power blocks and rechargers are machines too
const moduleMachineName = (item: SaveItem): string | null => {
    const typesStr = item.itemTypes?.join(' ').toLowerCase() || '';
    const keysStr = (item._keys || []).join(' ').toLowerCase();

    const isMachine = typesStr.includes("machine") || keysStr.includes("standard_machine_tag") || keysStr.includes("machinery");
    if (!isMachine) return null;

    return machineNameFromIdentifier(item.identifier || '') ?? machineNameFromName(item.name || '');
};

const getShapeFromData = (width: number, data: number[], isHighTier: boolean, blocks: number): ModuleShape | null => {
    if (blocks <= 2) return 'Node1x2';

    const points: Point[] = [];
    for (let i = 0; i < data.length; i++) {
        if (data[i] === 1) points.push({ x: i % width, y: Math.floor(i / width) });
    }

    const minX = Math.min(...points.map(p => p.x));
    const minY = Math.min(...points.map(p => p.y));
    const normalized = points.map(p => ({ x: p.x - minX, y: p.y - minY })).sort((a, b) => a.y === b.y ? a.x - b.x : a.y - b.y);
    const hash = JSON.stringify(normalized);

    for (const [shape, offsetsList] of PRECOMPUTED_OFFSETS.entries()) {
        for (const offsets of offsetsList) {
            const offMinX = Math.min(...offsets.map(p => p.x));
            const offMinY = Math.min(...offsets.map(p => p.y));
            const offNorm = offsets.map(p => ({ x: p.x - offMinX, y: p.y - offMinY })).sort((a, b) => a.y === b.y ? a.x - b.x : a.y - b.y);

            if (JSON.stringify(offNorm) === hash) {
                if (shape.includes('Base') && isHighTier) continue;
                if (shape.includes('High') && !isHighTier) continue;
                return shape;
            }
        }
    }
    return null;
};

const getTransformedPoints = (w: number, h: number, data: number[], orientation: number, flipped: boolean) => {
    const pts: Point[] = [];
    for (let i = 0; i < data.length; i++) {
        if (data[i] === 1) {
            let cx = i % w;
            let cy = Math.floor(i / w);

            if (flipped) cx = (w - 1) - cx;

            let cw = w;
            let ch = h;

            for (let r = 0; r < orientation; r++) {
                const nx = cy;
                const ny = cw - 1 - cx;
                cx = nx;
                cy = ny;
                const temp = cw; cw = ch; ch = temp;
            }
            pts.push({ x: cx, y: cy });
        }
    }

    if (pts.length === 0) return pts;

    const minX = Math.min(...pts.map(p => p.x));
    const minY = Math.min(...pts.map(p => p.y));

    return pts.map(p => ({ x: p.x - minX, y: p.y - minY }));
};

const parseEffectStr = (t: string): ItemEffect | null => {
    if (!t) return null;
    if (t.includes("PREMIUM")) return "Premium";
    if (t.includes("INFERIOR")) return "Inferior";
    if (t.includes("OVERVOLTED")) return "Overcharged";
    if (t.includes("DEGRADING")) return "Degrading";
    if (t.includes("NEGATIVE_FEEDBACK")) return "Negative Feedback";
    if (t.includes("RECEIVER")) return "Receiver";
    if (t.includes("SIDE_MOUNT")) return "Side Mount";
    if (t.includes("TOP_MOUNT")) return "Top Mount";
    if (t.includes("LEARN")) return "Learning Algorithm";
    return null;
};

const findValue = (item: SaveItem, key: string): SaveValue | undefined => {
    const idx = (item._keys || []).indexOf(key);
    if (idx !== -1 && item._values && item._values[idx] !== undefined) return item._values[idx];
    return item._values?.find(v => v?.['<identifier>k__BackingField'] === key);
};

const getTagStr = (item: SaveItem, key: string): string =>
    findValue(item, key)?.internalValueString?.toUpperCase() || '';

const getIntVal = (item: SaveItem, key: string): number | null =>
    findValue(item, key)?.valueInt ?? null;

const readEffects = (item: SaveItem): [ItemEffect, ItemEffect] => {
    const eff1 = parseEffectStr(getTagStr(item, 'MODULE_EFFECT1_TAG')) || 'None';
    const eff2 = parseEffectStr(getTagStr(item, 'MODULE_EFFECT2_TAG')) || 'None';
    if (eff1 !== 'None' || eff2 !== 'None') return [eff1, eff2];

    const found: ItemEffect[] = [];
    const tags = [...(item._keys || []), ...(item._values || []).map(v => v.internalValueString || '')];
    tags.forEach(tag => {
        const parsed = parseEffectStr(tag.toUpperCase());
        if (parsed && !found.includes(parsed)) found.push(parsed);
    });

    return [found[0] || 'None', found[1] || 'None'];
};

// Containers number their contents in grid order so two "储藏区" read as distinct paths
const numberSiblings = (children: SaveItem[]) => {
    children.sort((a, b) => {
        const aShape = a.itemModifiedShape || a.itemShape || {};
        const bShape = b.itemModifiedShape || b.itemShape || {};
        const aY = aShape['<minY>k__BackingField'] ?? 0;
        const bY = bShape['<minY>k__BackingField'] ?? 0;
        if (aY !== bY) return aY - bY;
        const aX = aShape['<minX>k__BackingField'] ?? 0;
        const bX = bShape['<minX>k__BackingField'] ?? 0;
        return aX - bX;
    });

    const baseCounters: Record<string, number> = {};
    const customCounters: Record<string, number> = {};

    children.forEach(child => {
        const baseName = child.name || 'Unknown';
        const customName = findValue(child, 'CUSTOM_NAME_TAG')?.internalValueString;
        const resolvedName = customName || baseName;

        if (child.identifier === 'save_bag' || baseName === 'Save Bag') {
            child.numberedName = 'Inv.';
        } else if (customName) {
            if (customCounters[resolvedName] === undefined) {
                customCounters[resolvedName] = 0;
                child.numberedName = resolvedName;
            } else {
                customCounters[resolvedName] += 1;
                child.numberedName = `${resolvedName} ${customCounters[resolvedName]}`;
            }
        } else {
            baseCounters[resolvedName] = (baseCounters[resolvedName] || 0) + 1;
            child.numberedName = `${resolvedName} ${baseCounters[resolvedName]}`;
        }
    });
};

const buildPath = (startId: number | undefined, itemMap: Map<number, SaveItem>, parentMap: Map<number, number>): string => {
    let path = '';
    let currId = startId;

    while (currId !== undefined) {
        const node = itemMap.get(currId);
        if (!node) break;
        const nodeName = node.numberedName || node.name || 'Unknown';
        path = path ? `${nodeName} > ${path}` : nodeName;
        currId = parentMap.get(node.uuid);
    }

    return path;
};

const readMachineTier = (machine: SaveItem): GridTier => {
    const stageVal = getIntVal(machine, 'MODULE_UPGRADE_STAGE_INT');
    if (stageVal === null || stageVal < 0 || stageVal > 2) return 1;
    return (stageVal + 1) as GridTier;
};

const placeOnBoard = (boardIds: (string | null)[][], invId: string, shape: SaveShape) => {
    const w = shape['<width>k__BackingField'] ?? 1;
    const h = shape['<height>k__BackingField'] ?? 1;
    const minX = shape['<minX>k__BackingField'] ?? 0;
    const minY = shape['<minY>k__BackingField'] ?? 0;
    const orientation = shape['<orientation>k__BackingField'] || 0;
    const flipped = shape['<flipped>k__BackingField'] || false;

    for (const p of getTransformedPoints(w, h, shape.data || [], orientation, flipped)) {
        const finalX = minX + p.x;
        const finalY = minY + p.y;
        if (finalY >= 0 && finalY < 5 && finalX >= 0 && finalX < 7) boardIds[finalY][finalX] = invId;
    }
};

export const parseSaveFile = (raw: string): ImportedSave => {
    const saveItems = readSaveItems(raw);

    const itemMap = new Map<number, SaveItem>();
    saveItems.forEach(item => itemMap.set(item.uuid, item));

    const parentMap = new Map<number, number>();
    saveItems.forEach(item => {
        item.childItems?.forEach(childId => parentMap.set(childId, item.uuid));
    });

    const childrenMap = new Map<number | undefined, SaveItem[]>();
    saveItems.forEach(item => {
        const pId = parentMap.get(item.uuid);
        if (!childrenMap.has(pId)) childrenMap.set(pId, []);
        childrenMap.get(pId)!.push(item);
    });
    childrenMap.forEach(numberSiblings);

    const items: InventoryItem[] = [];
    const machines: ImportedMachine[] = [];
    const machineContents = new Map<number, { invId: string, modifiedShape: SaveShape }[]>();
    const machineNames = new Map<number, string>();

    saveItems.forEach(item => {
        const machineName = moduleMachineName(item);
        if (machineName === null) return;
        machineContents.set(item.uuid, []);
        machineNames.set(item.uuid, machineName);
    });

    saveItems.forEach(item => {
        if (!isModule(item)) return;

        const shapeData = item.itemShape?.data;
        const w = item.itemShape?.['<width>k__BackingField'];
        if (!shapeData || w === undefined) return;

        const { color, isHighTier, displayName } = identityFromIdentifier(item.identifier || '') ?? identityFromName(item.name || '');

        const blocks = shapeData.filter(b => b === 1).length;
        let shape = getShapeFromData(w, shapeData, isHighTier, blocks);
        if (!shape) {
            if (blocks <= 2) shape = 'Node1x2';
            else if (blocks === 3) shape = 'L3';
            else if (blocks === 4) shape = isHighTier ? 'L4_High' : 'L4_Base';
            else shape = 'P5';
        }

        const [eff1, eff2] = readEffects(item);

        let primaryStat: number | null = null;
        if (color === 'Red' || color === 'Purple') primaryStat = getIntVal(item, "BONUS_PERCENTAGE_PERFORMANCE_INT");
        if (color === 'Yellow') primaryStat = getIntVal(item, "BONUS_PERCENTAGE_QUALITY_INT");
        if (color === 'Green') primaryStat = getIntVal(item, "BONUS_PERCENTAGE_EFFICIENCY_INT");

        const base = getBaseStats({ shape, color, displayName });
        const maxPositiveBase = Math.max(
            base.Performance > 0 ? base.Performance : 0,
            base.Quality > 0 ? base.Quality : 0,
            base.Efficiency > 0 ? base.Efficiency : 0
        );

        const defaultDoubleBase = Math.floor(maxPositiveBase * 2);
        const customVal = primaryStat !== null ? Math.abs(primaryStat) : 0;
        const effValue = (eff: ItemEffect) => (eff === 'Learning Algorithm' || eff === 'Degrading') ? customVal : defaultDoubleBase;

        const invId = `${shape}_${color}_${Math.random().toString(36).substring(2, 8)}`;

        const path = buildPath(parentMap.get(item.uuid), itemMap, parentMap);
        const originalPath = !path ? 'Inv.' : path.startsWith('Inv.') ? path : `Inv. > ${path}`;

        items.push({
            id: invId,
            shape,
            color,
            displayName,
            effects: [eff1, eff2],
            effectValues: [effValue(eff1), effValue(eff2)],
            isInfinite: false,
            isLocked: false,
            originalPath
        } as InventoryItem);

        const parentId = parentMap.get(item.uuid);
        const contents = parentId !== undefined ? machineContents.get(parentId) : undefined;
        contents?.push({ invId, modifiedShape: item.itemModifiedShape || item.itemShape || {} });
    });

    machineContents.forEach((contents, machineId) => {
        const boardIds = Array.from({ length: 5 }, () => Array.from({ length: 7 }, () => null as string | null));
        contents.forEach(({ invId, modifiedShape }) => placeOnBoard(boardIds, invId, modifiedShape));

        const machine = itemMap.get(machineId);
        const path = buildPath(machineId, itemMap, parentMap);

        machines.push({
            id: `m_${Math.random().toString(36).substring(2, 8)}`,
            boardIds,
            machineType: path || machineNames.get(machineId) || "Select Machine...",
            tier: machine ? readMachineTier(machine) : 1
        });
    });

    return { items, machines };
};
