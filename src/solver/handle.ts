import type { SolveUpdate } from './report';

export interface SolverHandle {
    stop: () => void;
    // Hands the search a record found elsewhere, which it takes up if it beats its own
    adopt: (update: SolveUpdate) => void;
    done: Promise<{ iterations: number }>;
}

export type UpdateHandler = (update: SolveUpdate) => void;
