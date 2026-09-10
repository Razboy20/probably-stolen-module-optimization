import { runOptimizationEngine, type SolveRequest } from './engine';
import { runGpuPopulation } from './gpu/population';
import { gpuMayWork, gpuSupported, markGpuFailed } from './gpu/probe';
import type { SolverHandle, UpdateHandler } from './handle';
import { compareTiers } from './objective';
import { randomSeed } from './rng';
import type { WorkerRequest, WorkerResponse } from './worker';

export type { SolverHandle } from './handle';

export type SolverBackend = 'auto' | 'gpu' | 'population' | 'workers' | 'inline';
export const SOLVER_BACKENDS: { value: SolverBackend; label: string }[] = [
    { value: 'auto', label: 'Auto' },
    { value: 'gpu', label: 'GPU' },
    { value: 'population', label: 'All cores' },
    { value: 'workers', label: 'One worker' },
    { value: 'inline', label: 'Main thread' },
];

const BACKEND_STORAGE_KEY = 'optimizer_backend';

export const readBackendPreference = (): SolverBackend => {
    if (typeof localStorage === 'undefined') return 'auto';
    const saved = localStorage.getItem(BACKEND_STORAGE_KEY);
    return SOLVER_BACKENDS.some(b => b.value === saved) ? saved as SolverBackend : 'auto';
};

export const writeBackendPreference = (backend: SolverBackend) => localStorage.setItem(BACKEND_STORAGE_KEY, backend);

// Passes on only the reports that beat every report before them, so several sources of records read as one improving record
const recordGate = (onUpdate: UpdateHandler): UpdateHandler => {
    let bestTiers: Int32Array | null = null;
    return update => {
        const tiers = Int32Array.from(update.tiers);
        if (bestTiers !== null && compareTiers(tiers, bestTiers, tiers.length) <= 0) return;
        bestTiers = tiers;
        onUpdate(update);
    };
};

const runInline = (request: SolveRequest, onUpdate: UpdateHandler): SolverHandle => {
    const control = { running: true };
    return { stop: () => { control.running = false; }, done: runOptimizationEngine(request, control, onUpdate) };
};

const spawnWorker = () => new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

// The worker is torn down once the solve reports done, so a stop still lets the engine flush its last record
const runInWorker = (request: SolveRequest, onUpdate: UpdateHandler): SolverHandle => {
    const worker = spawnWorker();
    const send = (message: WorkerRequest) => worker.postMessage(message);
    let stopped = false;

    const done = new Promise<{ iterations: number }>((resolve, reject) => {
        worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
            const message = event.data;
            if (message.type === 'update') {
                onUpdate(message.update);
            } else {
                worker.terminate();
                resolve({ iterations: message.iterations });
            }
        };
        worker.onerror = event => {
            worker.terminate();
            reject(event.error ?? new Error(event.message));
        };
    });
    send({ type: 'start', request });

    return {
        stop: () => {
            if (stopped) return;
            stopped = true;
            send({ type: 'stop' });
        },
        done
    };
};

/* Independent solves of the same request on several workers, each on its own random stream
 * Only a report that beats every report so far is passed on, so the caller sees one monotonically improving record
 */
const runPopulation = (request: SolveRequest, onUpdate: UpdateHandler, size: number): SolverHandle => {
    const seed = request.seed ?? randomSeed();
    const gate = recordGate(onUpdate);
    const members = Array.from({ length: size }, (_, thread) => runInWorker({ ...request, seed, thread }, gate));
    return {
        stop: () => members.forEach(m => m.stop()),
        done: Promise.all(members.map(m => m.done)).then(results => ({ iterations: results.reduce((sum, r) => sum + r.iterations, 0) }))
    };
};

// A solve whose backend fails partway carries on elsewhere; the records it already reported stand, and the gate keeps the successor from reporting worse ones
const withFallback = (primary: SolverHandle, makeFallback: () => SolverHandle): SolverHandle => {
    let active = primary;
    let stopped = false;
    const done = primary.done.catch(error => {
        console.warn('Solver backend failed, continuing on workers', error);
        if (stopped) return { iterations: 0 };
        active = makeFallback();
        return active.done;
    });
    return { stop: () => { stopped = true; active.stop(); }, done };
};

const workersAvailable = () => typeof Worker !== 'undefined';
const coreCount = () => (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;

// The first solve running takes the GPU or every spare core; solves started while it runs (Run All) get one worker each
let activeSolves = 0;
let gpuLeased = false;
const populationSize = (requested?: number) => requested ?? (activeSolves === 0 ? Math.max(1, coreCount() - 1) : 1);

const resolveBackend = (preference: SolverBackend): Exclude<SolverBackend, 'auto'> => {
    if (preference === 'inline') return 'inline';
    if (preference === 'gpu') return gpuSupported() && !gpuLeased ? 'gpu' : 'population';
    if (preference === 'auto' && gpuMayWork() && !gpuLeased) return 'gpu';
    if (!workersAvailable()) return 'inline';
    return preference === 'auto' ? 'population' : preference;
};

const track = (handle: SolverHandle): SolverHandle => {
    activeSolves++;
    handle.done.finally(() => { activeSolves--; });
    return handle;
};

const runOnCpu = (request: SolveRequest, onUpdate: UpdateHandler, backend: 'population' | 'workers', workers?: number): SolverHandle => {
    if (!workersAvailable()) return runInline(request, onUpdate);
    const size = backend === 'population' ? populationSize(workers) : 1;
    return size > 1 ? runPopulation(request, onUpdate, size) : runInWorker(request, onUpdate);
};

const runOnGpu = (request: SolveRequest, onUpdate: UpdateHandler, parallelism?: number): SolverHandle => {
    gpuLeased = true;
    const gate = recordGate(onUpdate);
    const gpu = runGpuPopulation(request, gate, parallelism);
    gpu.done.catch(markGpuFailed).finally(() => { gpuLeased = false; });
    return withFallback(gpu, () => runOnCpu(request, gate, 'population'));
};

// parallelism overrides the worker count or GPU thread count, for benchmarks
export const runSolver = (
    request: SolveRequest, onUpdate: UpdateHandler, preference = readBackendPreference(), parallelism?: number
): SolverHandle => {
    const backend = resolveBackend(preference);
    if (backend === 'gpu') return track(runOnGpu(request, onUpdate, parallelism));
    if (backend !== 'inline') {
        try {
            return track(runOnCpu(request, onUpdate, backend, parallelism));
        } catch (error) {
            console.warn('Solver worker unavailable, running on the main thread', error);
        }
    }
    return track(runInline(request, onUpdate));
};
