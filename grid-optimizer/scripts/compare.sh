#!/bin/bash
# usage: compare.sh <baseline checkout> <candidate checkout> [ms]
# Runs the CPU bench on both checkouts over SEEDS (default 1-8) and every bench case, and prints the record tiers at each checkpoint side by side,
# then how often the candidate came out ahead, level or behind at the last checkpoint, with the mean difference on the first tier that differed
# The search is stochastic, so a change is judged on this table and never on one run
# MACHINES is passed to both sides; BASE_ENV and CAND_ENV are extra VAR=value settings for one side, so INDEPENDENT=1 or PRESOLVE=1 can be compared to a joint solve of the same checkout
set -e
base=$1; cand=$2; ms=${3:-3000}
seeds=${SEEDS:-1 2 3 4 5 6 7 8}
cases=${CASES:-0 1 2}
machines=${MACHINES:-1}

run() {
    (cd "$1" && env $4 SEED=$2 TARGETS=$3 MS=$ms MACHINES=$machines bun scripts/bench.ts 2>/dev/null | tail -1)
}

results=""
for t in $cases; do for seed in $seeds; do
    a=$(run "$base" "$seed" "$t" "$BASE_ENV")
    b=$(run "$cand" "$seed" "$t" "$CAND_ENV")
    results+=$(printf '%s\n%s\n' "$a" "$b")$'\n'
done; done

echo "$results" | bun -e '
    const lines = (await Bun.stdin.text()).trim().split("\n").map(l => JSON.parse(l));
    const tiers = (j, ms) => j.checkpoints[ms]?.tiers ?? [];
    const fmt = (j) => Object.keys(j.checkpoints).map(Number).sort((x, y) => x - y).map(ms => `${ms}:[${tiers(j, ms).join(",")}]`).join(" ");
    const cmp = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1; return 0; };
    // The tiers above the first one that differs are equal, so the delta there is the whole difference between the two runs
    const firstDelta = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return { tier: i, delta: b[i] - a[i] }; return { tier: -1, delta: 0 }; };
    const tally = {};
    for (let i = 0; i < lines.length; i += 2) {
        const a = lines[i], b = lines[i + 1];
        const lastA = tiers(a, Math.max(...Object.keys(a.checkpoints).map(Number)));
        const lastB = tiers(b, Math.max(...Object.keys(b.checkpoints).map(Number)));
        const verdict = cmp(lastB, lastA);
        const t = (tally[a.targets] ??= { ahead: 0, level: 0, behind: 0, delta: 0, tiers: new Set() });
        t[verdict > 0 ? "ahead" : verdict < 0 ? "behind" : "level"]++;
        const d = firstDelta(lastA, lastB);
        t.delta += d.delta;
        if (d.tier >= 0) t.tiers.add(d.tier);
        console.log(`case=${a.targets} seed=${a.seed}`);
        console.log(`  base ${String(a.itersPerSec).padStart(7)} it/s  ${fmt(a)}`);
        console.log(`  cand ${String(b.itersPerSec).padStart(7)} it/s  ${fmt(b)}  ${verdict > 0 ? "+" : verdict < 0 ? "-" : "="}`);
    }
    for (const [c, t] of Object.entries(tally)) {
        const at = t.tiers.size === 0 ? "" : ` on tier ${[...t.tiers].sort().join("/")}`;
        console.log(`case=${c}: ahead ${t.ahead}, level ${t.level}, behind ${t.behind}, mean delta ${(t.delta / (t.ahead + t.level + t.behind)).toFixed(1)}${at}`);
    }
'
