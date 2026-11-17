import { type Arranger, runArranger } from './arranger';
import type { SolveUpdate } from './report';
import type { SolveRequest } from './setup';

export type ArrangeWorkerRequest = { type: 'start'; request: SolveRequest } | { type: 'offer'; update: SolveUpdate } | { type: 'finish' };
export type ArrangeWorkerResponse = { type: 'update'; update: SolveUpdate } | { type: 'done' };

let arranger: Arranger | null = null;
const post = (message: ArrangeWorkerResponse) => self.postMessage(message);

self.onmessage = async (event: MessageEvent<ArrangeWorkerRequest>) => {
    const message = event.data;
    if (message.type === 'start') {
        arranger = runArranger(message.request, update => post({ type: 'update', update }), true);
    } else if (message.type === 'offer') {
        arranger?.offer(message.update);
    } else {
        await arranger?.finish();
        post({ type: 'done' });
    }
};
