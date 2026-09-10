import type { Stats } from '../types';
import { type MachineConfig, STAT_KEYS, statIsIgnored } from './objective';

// The heuristic weights and targets of one machine, ready for the placement scorer
// Targets are rounded up: totals are integers, so a total meets a decimal target exactly when it meets the next integer
// An ignored stat keeps its target (it still decides which draw table is used) but weighs nothing
// Targets are capped so the objective's shortfall penalty stays within 32 bits
export const MAX_TARGET = 50000;
export interface ScoringParams {
    w: Int32Array;
    target: Int32Array;
    hasTarget: Int32Array;
    maximize: Int32Array;
}

export const buildScoringParams = (machine: MachineConfig, weights: Stats): ScoringParams => {
    const params: ScoringParams = {
        w: new Int32Array(3), target: new Int32Array(3), hasTarget: new Int32Array(3), maximize: new Int32Array(3)
    };
    for (let s = 0; s < 3; s++) {
        const key = STAT_KEYS[s];
        if (!statIsIgnored(machine, key)) params.w[s] = weights[key];
        const target = machine.targetStats[key];
        if (target !== null) {
            params.hasTarget[s] = 1;
            params.target[s] = Math.min(MAX_TARGET, Math.ceil(target));
        }
        params.maximize[s] = machine.maximizeStats[key] ? 1 : 0;
    }
    return params;
};

// How much one stat moving by `delta` is worth
// Progress toward an unmet target is paid in full (ten times over when the stat is also maximized), overshoot only at the maximize rate,
// and undercutting a met target is punished a hundredfold
export const scoreStat = (delta: number, current: number, w: number, target: number, hasTarget: number, maximize: number) => {
    if (delta === 0 || w === 0) return 0;
    const after = current + delta;

    if (hasTarget !== 0 && maximize === 0) {
        if (current >= target) return after >= target ? 0 : delta * w * 100;
        return after <= target ? delta * w : (target - current) * w;
    }
    if (hasTarget !== 0) {
        if (current >= target) return delta * w;
        return after <= target ? delta * w * 10 : (target - current) * w * 10 + (after - target) * w;
    }
    if (maximize !== 0) return delta * w;
    return 0;
};
