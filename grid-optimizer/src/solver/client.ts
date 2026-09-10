import { runOptimizationEngine, type SolveRequest, type SolveUpdate } from './engine';
import { compareTiers } from './objective';
import { randomSeed } from './rng';
import type { WorkerRequest, WorkerResponse } from './worker';

export type SolverBackend = 'auto' | 'population' | 'workers' | 'inline';
export const SOLVER_BACKENDS: { value: SolverBackend; label: string }[] = [
    { value: 'auto', label: 'Auto' },
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

export interface SolverHandle {
    stop: () => void;
    done: Promise<{ iterations: number }>;
}

type UpdateHandler = (update: SolveUpdate) => void;

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
    let bestTiers: Int32Array | null = null;
    const members = Array.from({ length: size }, (_, thread) => runInWorker({ ...request, seed, thread }, update => {
        const tiers = Int32Array.from(update.tiers);
        if (bestTiers !== null && compareTiers(tiers, bestTiers, tiers.length) <= 0) return;
        bestTiers = tiers;
        onUpdate(update);
    }));
    return {
        stop: () => members.forEach(m => m.stop()),
        done: Promise.all(members.map(m => m.done)).then(results => ({ iterations: results.reduce((sum, r) => sum + r.iterations, 0) }))
    };
};

const workersAvailable = () => typeof Worker !== 'undefined';
const coreCount = () => (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;

// The first solve running takes every spare core; solves started while it runs (Run All) get one worker each
let activeSolves = 0;
const populationSize = (requested?: number) => requested ?? (activeSolves === 0 ? Math.max(1, coreCount() - 1) : 1);

const resolveBackend = (preference: SolverBackend): Exclude<SolverBackend, 'auto'> => {
    if (preference === 'inline') return 'inline';
    if (!workersAvailable()) return 'inline';
    return preference === 'auto' ? 'population' : preference;
};

const track = (handle: SolverHandle): SolverHandle => {
    activeSolves++;
    handle.done.finally(() => { activeSolves--; });
    return handle;
};

export const runSolver = (
    request: SolveRequest, onUpdate: UpdateHandler, preference = readBackendPreference(), workers?: number
): SolverHandle => {
    const backend = resolveBackend(preference);
    if (backend !== 'inline') {
        try {
            const size = backend === 'population' ? populationSize(workers) : 1;
            return track(size > 1 ? runPopulation(request, onUpdate, size) : runInWorker(request, onUpdate));
        } catch (error) {
            console.warn('Solver worker unavailable, running on the main thread', error);
        }
    }
    return track(runInline(request, onUpdate));
};
