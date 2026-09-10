import { type RefObject, useCallback, useEffect, useRef, useState } from 'react';
import type { Board } from '../solver/board';
import { runSolver, type SolverHandle } from '../solver/client';
import type { MachineConfig } from '../solver/objective';
import type { BoardUpdate } from '../solver/report';
import type { InventoryItem } from '../types';

export interface MachineHandle {
    getState: () => Omit<MachineConfig, 'id'>;
    getBoard: () => Board;
    isLocked: () => boolean;
    applyUpdate: (update: BoardUpdate) => void;
    setWarning: (message: string | null) => void;
}

interface Run {
    ids: string[];
    handle: SolverHandle;
}

const boardItemIds = (board: Board) => {
    const ids = new Set<string>();
    for (const row of board) for (const cell of row) if (cell && cell !== 'Locked') ids.add(cell.id);
    return ids;
};

/* Every machine solving at once is in one joint solve, judged on their combined objective, so a module can move to whichever board gains more from it
 * Starting or stopping one machine while others solve restarts the solve over the new set; the records already shown stand, since each restart begins from them
 */
export const useJointSolve = (machinesRef: RefObject<Record<string, MachineHandle>>, inventory: InventoryItem[]) => {
    const [solvingIds, setSolvingIds] = useState<ReadonlySet<string>>(new Set());
    const runRef = useRef<Run | null>(null);
    const inventoryRef = useRef(inventory);
    useEffect(() => { inventoryRef.current = inventory; }, [inventory]);
    // Transitions are serialised, so a solve is always stopped and flushed before the next one reads the boards
    const queueRef = useRef(Promise.resolve());

    const launch = useCallback((ids: string[]) => {
        const handles = ids.map(id => machinesRef.current![id]);
        const items = inventoryRef.current;
        const usedOutside = new Set<string>();
        for (const [id, machine] of Object.entries(machinesRef.current!)) {
            if (!ids.includes(id)) for (const itemId of boardItemIds(machine.getBoard())) usedOutside.add(itemId);
        }

        const poolIsEmpty = items.every(item => item.isLocked || usedOutside.has(item.id));
        const boardsAreEmpty = handles.every(h => boardItemIds(h.getBoard()).size === 0);
        if (poolIsEmpty && boardsAreEmpty) {
            handles.forEach(h => h.setWarning('Cannot optimize: No unused modules available.'));
            return null;
        }
        handles.forEach(h => h.setWarning(null));

        const request = {
            machines: ids.map((id, k) => ({ machine: { id, ...handles[k].getState() }, initialBoard: handles[k].getBoard() })),
            searchPoolInventory: items.map(item => usedOutside.has(item.id) ? { ...item, isLocked: true } : item),
            fullInventory: items
        };
        const handle = runSolver(request, update => update.boards.forEach((board, k) => handles[k].applyUpdate(board)));
        const run: Run = { ids, handle };
        handle.done.catch(error => {
            console.error(error);
            handles.forEach(h => h.setWarning('The optimizer stopped unexpectedly.'));
        }).finally(() => {
            if (runRef.current !== run) return;
            runRef.current = null;
            setSolvingIds(new Set());
        });
        return run;
    }, [machinesRef]);

    const transition = useCallback((next: (current: string[]) => string[]) => {
        queueRef.current = queueRef.current.then(async () => {
            const previous = runRef.current;
            const ids = next(previous?.ids ?? []);
            if (previous) {
                previous.handle.stop();
                await previous.handle.done.catch(() => undefined);
            }
            runRef.current = ids.length > 0 ? launch(ids) : null;
            setSolvingIds(new Set(runRef.current?.ids ?? []));
        });
    }, [launch]);

    const start = useCallback((ids: string[]) => transition(current => [...new Set([...current, ...ids])]), [transition]);
    const stop = useCallback((id: string) => transition(current => current.filter(other => other !== id)), [transition]);
    const stopAll = useCallback(() => transition(() => []), [transition]);

    return { solvingIds, start, stop, stopAll };
};
