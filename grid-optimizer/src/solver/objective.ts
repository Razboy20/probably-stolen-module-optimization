import type { GridTier, Stats, TargetStats } from '../types';
import type { BoardTotals } from './boardTotals';
import type { ScoringParams } from './scoring';

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
export const compareTiers = (a: Int32Array, b: Int32Array, length: number) => {
    for (let i = 0; i < length; i++) {
        if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
};

// Which objective tier each stat lands in (-1 when ignored), most important first
// One tier per distinct rank in play; with no priorities set this collapses to a single tier holding everything, which is the original objective
export interface TierPlan {
    tierCount: number;
    tierOf: Int32Array;
}

export const buildTierPlan = (machine: MachineConfig): TierPlan => {
    const activeRanks = new Set<number>();
    for (const key of STAT_KEYS) {
        if (!statIsIgnored(machine, key)) activeRanks.add(priorityOf(machine, key));
    }
    if (activeRanks.size === 0) activeRanks.add(DEFAULT_STAT_PRIORITY);
    const rankOrder = [...activeRanks].sort((a, b) => a - b);

    const tierOf = new Int32Array(3).fill(-1);
    for (let s = 0; s < 3; s++) {
        const key = STAT_KEYS[s];
        if (!statIsIgnored(machine, key)) tierOf[s] = rankOrder.indexOf(priorityOf(machine, key));
    }
    return { tierCount: rankOrder.length, tierOf };
};

// How much harder the placement heuristic leans on a stat for being in a higher tier
export const tierBoost = (plan: TierPlan, stat: number) =>
    plan.tierOf[stat] < 0 ? 1 : Math.pow(PRIORITY_WEIGHT_STEP, plan.tierCount - 1 - plan.tierOf[stat]);

// A tier per rank plus one more for the density reward, which is a general tiebreak and so can never take a ranked stat out of its own tier
export const TIER_VECTOR_LENGTH = 4;
export const DENSITY_TIER_WEIGHT = 5;

export const objectiveTiers = (totals: BoardTotals, plan: TierPlan, params: ScoringParams, out: Int32Array) => {
    out.fill(0);
    for (let s = 0; s < 3; s++) {
        const ti = plan.tierOf[s];
        if (ti < 0) continue;
        const t = s === 0 ? totals.p : s === 1 ? totals.q : totals.e;
        if (params.hasTarget[s] !== 0 && t < params.target[s]) out[ti] -= (params.target[s] - t) * 10000;
        if (params.maximize[s] !== 0) out[ti] += t * 10;
    }
    out[plan.tierCount] -= totals.pieces * DENSITY_TIER_WEIGHT;
};

/* A set of machines is scored on the sum of their tier vectors
 * Tier i of every machine lands at index i, so a machine without priorities has all its stats in the top tier next to the other machines' rank-1 stats,
 * and every machine's density term lands at the one index past the deepest machine's tiers
 */
export const addMachineTiers = (sum: Int32Array, tiers: Int32Array, tierCount: number, densityIndex: number) => {
    for (let i = 0; i < tierCount; i++) sum[i] += tiers[i];
    sum[densityIndex] += tiers[tierCount];
};
