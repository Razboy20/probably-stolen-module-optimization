import type { Stats } from '../types';
import { type MachineConfig, STAT_KEYS, statIsIgnored, type TierPlan, tierBoost } from './objective';
import { FLAG_WHITE, type PoolTables } from './tables';
/* What each pool entry is worth to this machine per cell it occupies, used to decide which modules the fill is offered
 * Board space is the scarce resource, so per-cell is the comparison that matters
 *
 * These weights are not the placement weights, and the difference is the point
 * The score pays nothing for overshooting a target, so once a target is met, another point of that stat is worth next to nothing for a stat that is only being held to a target
 * A machine sitting comfortably above P 500 and Q 150 should be offered Efficiency modules, not more of what it already has
 * The placement heuristic keeps the full target weight in every case, and that is what stops a met target being undercut. This only decides what gets offered to it
 *
 * Which targets are met changes as the board moves, so there is one table per combination of met targets (at most eight), built once here, picked by bitmask per fill
 * Rebuilding a table whenever the totals shifted would cost more than the bias is worth
 *
 * The draw only ever asks which of two entries is worth more, so each table holds the rank of the entry's value rather than the value itself
 */
export const TARGET_MET_DRAW_SCALE = 0.25;

const rankValues = (values: Float64Array) => {
    const distinct = [...new Set(values)].sort((a, b) => a - b);
    const rankOf = new Map(distinct.map((v, i) => [v, i]));
    return Int32Array.from(values, v => rankOf.get(v)!);
};

const drawValuesFor = (tables: PoolTables, drawList: Int32Array, w: Stats) => {
    const values = new Float64Array(drawList.length);
    const stated: number[] = [];
    for (let i = 0; i < drawList.length; i++) {
        const item = drawList[i];
        if ((tables.flags[item] & FLAG_WHITE) !== 0 || tables.size[item] === 0) continue;
        values[i] = (tables.p[item] * w.Performance + tables.q[item] * w.Quality + tables.e[item] * w.Efficiency) / tables.size[item];
        stated.push(values[i]);
    }
    // Nodes carry no stats of their own: all their worth is the 20% they add to whatever ends up beside them, which only the placement scorer can see
    // Scoring them at the median leaves them drawn about as often as an ordinary module instead of never
    if (stated.length > 0) {
        stated.sort((a, b) => a - b);
        const median = stated[stated.length >> 1];
        for (let i = 0; i < drawList.length; i++) {
            if ((tables.flags[drawList[i]] & FLAG_WHITE) !== 0) values[i] = median;
        }
    }
    return rankValues(values);
};

export const targetedStats = (machine: MachineConfig) => {
    const out: number[] = [];
    for (let s = 0; s < 3; s++) if (machine.targetStats[STAT_KEYS[s]] !== null) out.push(s);
    return out;
};

export const buildDrawRanks = (tables: PoolTables, drawList: Int32Array, machine: MachineConfig, plan: TierPlan): Int32Array[] => {
    const targeted = targetedStats(machine);
    const ranks: Int32Array[] = [];
    for (let mask = 0; mask < (1 << targeted.length); mask++) {
        const w: Stats = { Performance: 0, Quality: 0, Efficiency: 0 };
        for (let s = 0; s < 3; s++) {
            const key = STAT_KEYS[s];
            if (statIsIgnored(machine, key)) continue;
            let v = 0;
            if (machine.maximizeStats[key]) v += 10;
            const ti = targeted.indexOf(s);
            // Bit set means the target is already met, so the push for it is cut back, but never to nothing
            // The draw decides which modules the fill is even offered,
            // so a stat whose modules stop being drawn cannot be rebuilt when a ruin knocks it below its target, and every repair after that is rejected
            if (ti !== -1) v += 15 * ((mask & (1 << ti)) !== 0 ? TARGET_MET_DRAW_SCALE : 1);
            w[key] = v * tierBoost(plan, s);
        }
        ranks.push(drawValuesFor(tables, drawList, w));
    }
    return ranks;
};
