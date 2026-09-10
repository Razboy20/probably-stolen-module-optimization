/* A Weyl sequence hashed through lowbias32, all in 32-bit unsigned arithmetic so the same stream comes out of JavaScript and WGSL
 * Each stream has its own odd increment, so thousands of them seeded from consecutive thread ids never share a cycle
 */
export interface Rng {
    ctr: number;
    inc: number;
}

export const lowbias32 = (x: number) => {
    let z = x >>> 0;
    z ^= z >>> 16;
    z = Math.imul(z, 0x7feb352d) >>> 0;
    z ^= z >>> 15;
    z = Math.imul(z, 0x846ca68b) >>> 0;
    z ^= z >>> 16;
    return z >>> 0;
};

export const seedRng = (seed: number, thread: number): Rng => ({
    ctr: lowbias32((Math.imul(thread, 0x9e3779b9) + seed) >>> 0),
    inc: (lowbias32((thread ^ seed) >>> 0) | 1) >>> 0
});

export const rngNext = (rng: Rng) => {
    rng.ctr = (rng.ctr + rng.inc) >>> 0;
    return lowbias32(rng.ctr);
};

// Uniform in [0, n) for the small n the search draws
export const rngBelow = (rng: Rng, n: number) => rngNext(rng) % n;

export const rngCoinFlip = (rng: Rng) => (rngNext(rng) & 1) === 1;

export const randomSeed = () => (Math.random() * 0x100000000) >>> 0;
