import { runOptimizationEngine, type SolveControl, type SolveRequest, type SolveUpdate } from './engine';

export type WorkerRequest = { type: 'start'; request: SolveRequest } | { type: 'stop' };
export type WorkerResponse = { type: 'update'; update: SolveUpdate } | { type: 'done'; iterations: number };

const control: SolveControl = { running: false };
const post = (message: WorkerResponse) => self.postMessage(message);

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;
    if (message.type === 'stop') {
        control.running = false;
        return;
    }
    control.running = true;
    const { iterations } = await runOptimizationEngine(message.request, control, update => post({ type: 'update', update }));
    post({ type: 'done', iterations });
};
