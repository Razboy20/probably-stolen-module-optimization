import tgpu, { d } from 'typegpu';

/* The same Weyl + lowbias32 stream as ../rng.ts, written so it runs both as JavaScript and as WGSL
 * WGSL has no Math.imul, so the 32-bit product is built from 16-bit halves
 */
export const rngCtr = tgpu.privateVar(d.u32, 0);
export const rngInc = tgpu.privateVar(d.u32, 0);

const mulU32 = (a: number, b: number) => {
    'use gpu';
    const lo = d.u32(a & 0xffff);
    const hi = d.u32(a >>> 16);
    return d.u32((lo * d.u32(b) + ((hi * d.u32(b)) << 16)) >>> 0);
};

export const lowbias32 = (x: number) => {
    'use gpu';
    let z = d.u32(x);
    z = d.u32((z ^ (z >>> 16)) >>> 0);
    z = mulU32(z, d.u32(0x7feb352d));
    z = d.u32((z ^ (z >>> 15)) >>> 0);
    z = mulU32(z, d.u32(0x846ca68b));
    return d.u32((z ^ (z >>> 16)) >>> 0);
};

export const rngNext = () => {
    'use gpu';
    rngCtr.$ = d.u32(rngCtr.$ + rngInc.$);
    return lowbias32(rngCtr.$);
};

export const rngBelow = (n: number) => {
    'use gpu';
    const top = d.u32(rngNext() >>> 16);
    return d.i32((top * d.u32(n)) >>> 16);
};

export const rngCoinFlip = () => {
    'use gpu';
    return (rngNext() & d.u32(1)) === d.u32(1);
};
