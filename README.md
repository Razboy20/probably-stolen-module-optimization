# Module Optimizer

Places modules on machine grids to hit target Performance, Quality and Efficiency stats. Import a `.es3` save, pick targets per machine, and run. Several machines can be solved together so they share one module inventory.

Fork of [hoydoy/probably-stolen-module-optimization](https://github.com/hoydoy/probably-stolen-module-optimization) with a rewritten solver.

## Solver

The search is an iterated local search over placements. It runs on one of four backends, chosen automatically or from the UI:

- `gpu`: TypeGPU population search in WebGPU
- `population`: one Web Worker per core
- `workers`: a single Web Worker
- `inline`: on the main thread

## Development

```sh
pnpm install
pnpm dev
pnpm build
pnpm deploy   # Cloudflare Workers
```

## Benchmarks

Scripts in `scripts/` run the solver headlessly under Bun.

```sh
pnpm bench                       # one machine, 3s, seed 1
SEED=3 MS=10000 pnpm bench       # longer run on another case
MACHINES=3 pnpm bench            # joint solve over three machines
scripts/compare.sh <base> <cand> # A/B two checkouts over seeds 1-8
```

`bench.ts` reads `SEED`, `N`, `MS`, `ITERS`, `TARGETS`, `IMPL`, `WORKERS`, `MACHINES`, `INDEPENDENT` and `PRESOLVE` from the environment. The search is stochastic, so judge changes on the compare table and never on a single run.
