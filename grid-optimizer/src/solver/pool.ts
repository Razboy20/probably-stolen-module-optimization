import type { InventoryItem, ItemEffect, Stats } from '../types';
import { isLockedModule } from './locked';
import { type MachineConfig, STAT_KEYS, statIsScored } from './objective';

// Effects whose value depends on where the module ends up (edge contact, adjacent nodes, adjacent negatives)
// Modules carrying different ones cannot be compared on their stats alone, so they are only ever compared against modules carrying the same ones
const PLACEMENT_DEPENDENT_EFFECTS: ItemEffect[] = ['Side Mount', 'Top Mount', 'Receiver', 'Negative Feedback'];

// A 35-cell board cannot hold more than 17 pieces (the 2-cell Node is the smallest),
// so this many candidates per group is always enough to build any layout.
export const MAX_PIECES_PER_BOARD = 18;

const dominates = (a: number[], b: number[]) => {
    let strictlyBetter = false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] < b[i]) return false;
        if (a[i] > b[i]) strictlyBetter = true;
    }
    return strictlyBetter;
};

/* Drops modules the search can never benefit from trying.
 *
 * A module is only worth considering if no other module of the same shape is at least as good on every stat that actually scores
 * Swapping a dominated module for the one that dominates it occupies the same cells and scores at least as well,
 * so any layout using the dominated one is matched by a layout without it, and trying it only costs time
 *
 * Three things keep this from throwing away real options: modules are only compared within the same shape and the same placement-dependent effects,
 * stats the machine does not score on are left out of the comparison entirely, and enough candidates are kept per group to fill the board,
 * so pruning can never make a layout unreachable for lack of copies
 * Nodes are never dropped, and locked modules are left out of the pool entirely
 * The pool is what the fill draws NEW modules from, and those are never the solver's to add
 * Where a locked module already on a board sits is still the solver's problem. A Blast module against a Node costs real stats, so it is free to move around its board
 */
export const buildSearchPool = (
    inventory: InventoryItem[],
    precomputedInternal: Map<string, Stats>,
    machine: MachineConfig
): InventoryItem[] => {
    const scoredKeys = STAT_KEYS.filter(key => statIsScored(machine, key));

    if (scoredKeys.length === 0) return inventory.filter(item => !isLockedModule(item));

    const keepCap = MAX_PIECES_PER_BOARD;
    const kept: InventoryItem[] = [];
    const groups = new Map<string, Map<string, InventoryItem[]>>();

    for (const item of inventory) {
        if (isLockedModule(item)) continue;

        if (item.color === 'White') {
            kept.push(item);
            continue;
        }

        const signature = PLACEMENT_DEPENDENT_EFFECTS.filter(eff => item.effects.includes(eff)).join('+');
        const groupKey = `${item.shape}|${signature}`;

        const stats = precomputedInternal.get(item.id)!;
        const tupleKey = scoredKeys.map(key => stats[key]).join(',');

        let group = groups.get(groupKey);
        if (group === undefined) {
            group = new Map<string, InventoryItem[]>();
            groups.set(groupKey, group);
        }
        const bucket = group.get(tupleKey);
        if (bucket === undefined) group.set(tupleKey, [item]);
        else bucket.push(item);
    }

    for (const group of groups.values()) {
        const tuples = [...group.keys()];
        const parsed = tuples.map(t => t.split(',').map(Number));

        const frontier: InventoryItem[] = [];
        const rest: InventoryItem[] = [];
        for (let i = 0; i < tuples.length; i++) {
            let isDominated = false;
            for (let j = 0; j < tuples.length && !isDominated; j++) {
                if (i !== j && dominates(parsed[j], parsed[i])) isDominated = true;
            }
            const items = group.get(tuples[i])!;
            if (isDominated) rest.push(...items);
            else frontier.push(...items);
        }

        // Keep the undominated candidates first; top up from the rest so a group never ends up with fewer modules than a board could actually use
        for (const item of frontier) kept.push(item);
        for (let i = 0; i < rest.length && frontier.length + i < keepCap; i++) kept.push(rest[i]);
    }

    return kept;
};
