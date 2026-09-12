import { readFileSync } from 'node:fs';

const profile = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const top = Number(process.argv[3] ?? 25);

const byId = new Map(profile.nodes.map(n => [n.id, n]));
const parentOf = new Map();
for (const n of profile.nodes) for (const c of n.children ?? []) parentOf.set(c, n.id);

const selfTicks = new Map();
for (const id of profile.samples) selfTicks.set(id, (selfTicks.get(id) ?? 0) + 1);

const name = n => `${n.callFrame.functionName || '(anonymous)'} ${n.callFrame.url.split('/').pop()}:${n.callFrame.lineNumber + 1}`;

const self = new Map();
const total = new Map();
for (const [id, ticks] of selfTicks) {
    const seen = new Set();
    let cur = id;
    self.set(name(byId.get(id)), (self.get(name(byId.get(id))) ?? 0) + ticks);
    while (cur !== undefined) {
        const key = name(byId.get(cur));
        if (!seen.has(key)) {
            seen.add(key);
            total.set(key, (total.get(key) ?? 0) + ticks);
        }
        cur = parentOf.get(cur);
    }
}

const all = profile.samples.length;
const pct = v => `${(100 * v / all).toFixed(1).padStart(5)}%`;
console.log('self   total  function');
for (const [key, s] of [...self].sort((a, b) => b[1] - a[1]).slice(0, top)) {
    console.log(`${pct(s)} ${pct(total.get(key))}  ${key}`);
}
