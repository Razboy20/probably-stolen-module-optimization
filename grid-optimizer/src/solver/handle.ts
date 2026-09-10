import type { SolveUpdate } from './report';

export interface SolverHandle {
    stop: () => void;
    done: Promise<{ iterations: number }>;
}

export type UpdateHandler = (update: SolveUpdate) => void;
