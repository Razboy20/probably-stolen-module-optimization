/* A Weyl sequence hashed through lowbias32, all in 32-bit unsigned arithmetic so the same stream comes out of JavaScript and WGSL
 * Each stream has its own odd increment, so thousands of them seeded from consecutive thread ids never share a cycle
 * The state is a two-word typed array, counter then increment: the engine draws tens of millions of times a second, and a typed array keeps the step to an add
 */
export type Rng = Uint32Array;
export const RNG_CTR = 0;
export const RNG_INC = 1;

export const lowbias32 = (x: number) => {
    let z = x >>> 0;
    z ^= z >>> 16;
    z = Math.imul(z, 0x7feb352d) >>> 0;
    z ^= z >>> 15;
    z = Math.imul(z, 0x846ca68b) >>> 0;
    z ^= z >>> 16;
    return z >>> 0;
};

export const seedRng = (seed: number, thread: number): Rng => Uint32Array.of(
    lowbias32((Math.imul(thread, 0x9e3779b9) + seed) >>> 0),
    lowbias32((thread ^ seed) >>> 0) | 1
);

export const rngNext = (rng: Rng) => {
    rng[RNG_CTR] += rng[RNG_INC];
    return lowbias32(rng[RNG_CTR]);
};

// Uniform in [0, n) for the small n the search draws: the top sixteen bits scaled to n, which stays in integer arithmetic where a modulo of a full 32-bit word does not
export const rngBelow = (rng: Rng, n: number) => ((rngNext(rng) >>> 16) * n) >>> 16;

export const rngCoinFlip = (rng: Rng) => (rngNext(rng) & 1) === 1;

export const randomSeed = () => (Math.random() * 0x100000000) >>> 0;
