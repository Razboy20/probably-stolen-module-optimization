import type { GridTier, Stats, TargetStats } from '../types';

export type MachineConfig = {
    id: string;
    tier: GridTier;
    targetStats: TargetStats;
    maximizeStats: Record<keyof Stats, boolean>;
    ignoreStats?: Partial<Record<keyof Stats, boolean>>;
    // Rank per stat, 1 being the most important
    // Stats sharing a rank are traded off against each other exactly as they always were
    // A lower rank is only ever consulted once every higher one is tied, so a met high-priority target can never be dropped to meet a lower-priority one
    // Absent or all-equal means the single combined objective as before
    statPriority?: Partial<Record<keyof Stats, number>>;
};

export const STAT_KEYS: (keyof Stats)[] = ['Performance', 'Quality', 'Efficiency'];

// An ignored stat is worth nothing to this machine in either direction
export const statIsIgnored = (m: MachineConfig, key: keyof Stats) => Boolean(m.ignoreStats?.[key]);
export const DEFAULT_STAT_PRIORITY = 1;
export const priorityOf = (m: MachineConfig, key: keyof Stats) =>
    m.statPriority?.[key] ?? DEFAULT_STAT_PRIORITY;

export const statIsScored = (m: MachineConfig, key: keyof Stats) =>
    !statIsIgnored(m, key) && (Boolean(m.maximizeStats?.[key]) || m.targetStats[key] !== null);

// How much harder the placement heuristic leans on a stat per rank it is above the least important one
// The acceptance test is strictly ordered on its own; this only points the greedy fill in the same direction so it does not spend the search fighting the objective
export const PRIORITY_WEIGHT_STEP = 4;

// Lexicographic: the first rank that differs decides, so nothing below it can outvote it.
export const compareTiers = (a: Float64Array, b: Float64Array) => {
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
};
