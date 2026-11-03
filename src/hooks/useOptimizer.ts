import { useState, useRef, useEffect } from 'react';
import type { GridTier, InventoryItem, MachineKind, Stats, TargetStats, Point } from '../types';
import { type Board, type GridLayout, initializeBoard, layoutOf } from '../solver/board';
import { calculateBoardStats, indexInventoryById } from '../solver/boardStats';
import { decodeSolution, generateCodeFromState, inventoryForCode } from '../solver/codec';
import type { BoardUpdate } from '../solver/report';
import { isLockedModule } from '../solver/locked';
import { isModuleAllowedInMachine } from '../utils';

export type StatFlags = Record<keyof Stats, boolean>;
export type StatRanks = Record<keyof Stats, number>;

// Solves are run by the joint solve controller, which feeds this machine its record boards; isAnySolving is whether any machine is being solved
export function useOptimizer(
    inventory: InventoryItem[],
    setInventory: React.Dispatch<React.SetStateAction<InventoryItem[]>>,
    machineId: string,
    getUsedItems: (excludeId: string) => Set<string>,
    defaultTier: GridTier = 3,
    isAnySolving: boolean = false
) {
    const getSavedState = () => {
        const saved = localStorage.getItem(`optimizer_machine_${machineId}`);
        if (saved) {
            try { return JSON.parse(saved); } catch { return null; }
        }
        return null;
    };

    const savedState = getSavedState();

    const [tier, setTier] = useState<GridTier>(() => {
        const saved = localStorage.getItem(`optimizer_machine_${machineId}`);
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                if (parsed.tier && [1, 2, 3].includes(parsed.tier)) {
                    return parsed.tier as GridTier;
                }
            } catch (e) {
                console.error(e);
            }
        }
        return defaultTier;
    });

    const [kind, setKind] = useState<MachineKind | null>(savedState?.kind ?? null);
    const [label, setLabel] = useState<string>(savedState?.label ?? '');
    // Which machine of the imported save this is, so the next save of the run hands it back its settings
    const [saveId] = useState<number | undefined>(savedState?.saveId);
    const layout = layoutOf(kind, tier);

    const [targetStats, setTargetStats] = useState<TargetStats>(savedState?.targetStats ?? { Performance: null, Quality: null, Efficiency: null });
    const [maximizeStats, setMaximizeStats] = useState<StatFlags>(savedState?.maximizeStats ?? { Performance: false, Quality: false, Efficiency: false });
    const [ignoreStats, setIgnoreStats] = useState<StatFlags>(savedState?.ignoreStats ?? { Performance: false, Quality: false, Efficiency: false });
    const [statPriority, setStatPriority] = useState<StatRanks>(savedState?.statPriority ?? { Performance: 1, Quality: 1, Efficiency: 1 });

    const [board, setBoard] = useState<Board>(() => initializeBoard(layout, savedState?.boardIds, inventory));
    const boardRef = useRef(board);
    const setBoardSync = (newBoard: Board) => {
        boardRef.current = newBoard;
        setBoard(newBoard);
    };

    const [bestTotals, setBestTotals] = useState<Stats>({ Performance: 0, Quality: 0, Efficiency: 0 });
    const [bestPieceStats, setBestPieceStats] = useState<Map<string, Stats>>(new Map());

    const [warningMsg, setWarningMsg] = useState<string | null>(null);
    const [solutionCode, setSolutionCode] = useState<string>('');

    const startOver = (newLayout: GridLayout) => {
        setBoardSync(initializeBoard(newLayout));
        setBestTotals({ Performance: 0, Quality: 0, Efficiency: 0 });
        setBestPieceStats(new Map());
        setWarningMsg(null);
        setSolutionCode('');
    };

    const handleTierChange = (newTier: GridTier) => {
        setTier(newTier);
        startOver(layoutOf(kind, newTier));
    };

    // A kind with another grid starts the board over; otherwise only the modules the new kind refuses come off
    const changeMachineKind = (newKind: MachineKind, newLabel: string) => {
        setKind(newKind);
        setLabel(newLabel);
        const newLayout = layoutOf(newKind, tier);
        if (newLayout !== layout) {
            startOver(newLayout);
            return;
        }
        const removedNames = new Set<string>();
        const next = boardRef.current.map(row => row.map(cell => {
            if (cell && cell !== 'Locked' && !cell.isStuck && !isModuleAllowedInMachine(cell, newKind)) {
                removedNames.add(cell.displayName);
                return null;
            }
            return cell;
        }));
        if (removedNames.size > 0) {
            setBoardSync(next);
            setWarningMsg(`Removed ${[...removedNames].join(', ')}: incompatible with ${newKind}.`);
        }
    };

    const kindName = kind ?? 'this machine';
    const isModuleAllowed = (item: InventoryItem) => isModuleAllowedInMachine(item, kind);

    const emptyBoard = (keepLockedModules: boolean) => {
        const fresh = initializeBoard(layout);
        if (keepLockedModules) {
            for (let y = 0; y < 5; y++) {
                for (let x = 0; x < 7; x++) {
                    const cell = boardRef.current[y][x];
                    if (cell && cell !== 'Locked' && isLockedModule(cell) && fresh[y][x] !== 'Locked') {
                        fresh[y][x] = cell;
                    }
                }
            }
        }
        setBoardSync(fresh);
        setBestTotals({ Performance: 0, Quality: 0, Efficiency: 0 });
        setBestPieceStats(new Map());
        setWarningMsg(null);
        // Clear maximize settings
        //setTargetStats({ Performance: null, Quality: null, Efficiency: null });
        //setMaximizeStats({ Performance: false, Quality: false, Efficiency: false });
        //setIgnoreStats({ Performance: false, Quality: false, Efficiency: false });
        //setStatPriority({ Performance: 1, Quality: 1, Efficiency: 1 });
        setSolutionCode('');
    };

    // A locked module stays put when its board is cleared; only wiping the inventory it came from takes it off
    const resetBoard = () => emptyBoard(true);
    const resetBoardIncludingLocked = () => emptyBoard(false);

    const applyUpdate = (update: BoardUpdate) => {
        setBoardSync(update.board);
        setBestTotals(update.totals);
        setBestPieceStats(update.pieceStats);
        setSolutionCode(update.code);
    };

    const manuallyPlaceItem = (item: InventoryItem, rootX: number, rootY: number, offsets: Point[]) => {
        const next = boardRef.current.map(row => [...row]);
        const swapItemIds = new Set<string>();

        for (const pt of offsets) {
            const px = rootX + pt.x;
            const py = rootY + pt.y;
            if (px >= 0 && px < 7 && py >= 0 && py < 5) {
                const cell = next[py][px];
                if (cell && cell !== 'Locked' && cell.id !== item.id) {
                    if (cell.shape === item.shape) swapItemIds.add(cell.id);
                }
            }
        }

        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 7; x++) {
                const cell = next[y][x];
                if (cell && cell !== 'Locked') {
                    if (cell.id === item.id || swapItemIds.has(cell.id)) {
                        next[y][x] = null;
                    }
                }
            }
        }

        for (const pt of offsets) {
            const px = rootX + pt.x;
            const py = rootY + pt.y;
            if (px >= 0 && px < 7 && py >= 0 && py < 5) {
                next[py][px] = item;
            }
        }
        setBoardSync(next);
    };

    const manuallyRemoveItem = (itemId: string) => {
        const next = boardRef.current.map(row => [...row]);
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 7; x++) {
                const cell = next[y][x];
                if (cell && cell !== 'Locked' && cell.id === itemId) {
                    next[y][x] = null;
                }
            }
        }
        setBoardSync(next);
    };

    const isValidPlacement = (item: InventoryItem, rootX: number, rootY: number, offsets: Point[]) => {
        if (!isModuleAllowed(item)) return false;
        for (const pt of offsets) {
            const px = rootX + pt.x;
            const py = rootY + pt.y;
            if (px < 0 || px >= 7 || py < 0 || py >= 5) return false;
            const cell = boardRef.current[py][px];
            if (cell === 'Locked') return false;
            if (cell && cell.id !== item.id) return false;
        }
        return true;
    };

    // A module this machine refuses is turned away with a warning; any other invalid drop is silently ignored
    const tryPlace = (item: InventoryItem, rootX: number, rootY: number, offsets: Point[]) => {
        if (!isModuleAllowed(item)) {
            setWarningMsg(`Cannot place ${item.displayName}: incompatible with ${kindName}.`);
            return false;
        }
        if (!isValidPlacement(item, rootX, rootY, offsets)) return false;
        manuallyPlaceItem(item, rootX, rootY, offsets);
        return true;
    };

    const importSolution = (code: string) => {
        try {
            const decoded = decodeSolution(code);
            const codeIsEvaporator = decoded.layout === 'Evaporator';
            if (codeIsEvaporator !== (layout === 'Evaporator')) {
                setWarningMsg(codeIsEvaporator
                    ? 'This solution code was made for an Evaporator (4x4 grid). Select an Evaporator machine to import it.'
                    : 'This solution code was made for a standard (7x5 grid) machine and cannot be imported into an Evaporator.');
                return;
            }
            const placed = decoded.board.flat().filter((cell): cell is InventoryItem => cell !== null && cell !== 'Locked');
            const incompatible = new Set(placed.filter(item => !isModuleAllowed(item)).map(item => item.displayName));
            if (incompatible.size > 0) {
                setWarningMsg(`This solution contains modules that cannot be placed in ${kindName}: ${[...incompatible].join(', ')}.`);
                return;
            }
            if (decoded.layout !== 'Evaporator') setTier(decoded.layout);
            setMaximizeStats(decoded.maximizeStats);
            setTargetStats(decoded.targetStats);

            setInventory(decoded.inventory);
            setBoardSync(decoded.board);
            setSolutionCode(code);
            setWarningMsg(null);

            const { totals, pieceStats } = calculateBoardStats(decoded.board, decoded.inventory);
            setBestTotals(totals);
            setBestPieceStats(new Map(pieceStats));
        } catch {
            setWarningMsg("Failed to import solution code. The code might be broken or from an incompatible version.");
        }
    };

    useEffect(() => {
        if (!isAnySolving) {
            const invById = indexInventoryById(inventory);
            let boardChanged = false;
            const newBoard = boardRef.current.map(row => row.map(cell => {
                if (cell && cell !== 'Locked') {
                    const invMatch = invById.get(cell.id);
                    if (invMatch && invMatch !== cell) {
                        boardChanged = true;
                        return invMatch;
                    }
                }
                return cell;
            }));

            const boardToCalculate = boardChanged ? newBoard : boardRef.current;
            const used = getUsedItems(machineId);
            const availableInventory = inventory.filter(item => !used.has(item.id) && !item.isLocked);
            const { totals, pieceStats } = calculateBoardStats(boardToCalculate, availableInventory, indexInventoryById(availableInventory));

            setBestTotals(totals);
            setBestPieceStats(new Map(pieceStats));
            if (boardChanged) setBoardSync(newBoard);

            const boardIds = boardToCalculate.map(row => row.map(c => c && c !== 'Locked' ? c.id : c));
            localStorage.setItem(`optimizer_machine_${machineId}`, JSON.stringify({
                tier, kind, label, saveId, maximizeStats, targetStats, ignoreStats, statPriority, boardIds
            }));

            if (inventory.length > 0) {
                const availableForCode = inventoryForCode(inventory, boardToCalculate);
                setSolutionCode(generateCodeFromState(layout, maximizeStats, targetStats, availableForCode, boardToCalculate));
            } else {
                setSolutionCode('');
            }
        }
    }, [inventory, tier, kind, label, saveId, layout, maximizeStats, targetStats, ignoreStats, statPriority, machineId, getUsedItems, board, isAnySolving]);

    return {
        tier, setTier, handleTierChange, targetStats, setTargetStats,
        maximizeStats, setMaximizeStats, ignoreStats, setIgnoreStats,
        statPriority, setStatPriority, board, bestTotals, bestPieceStats,
        warningMsg, setWarningMsg, applyUpdate, kind, label, layout, changeMachineKind,
        solutionCode, setSolutionCode, importSolution, resetBoard, resetBoardIncludingLocked,
        tryPlace, manuallyRemoveItem, isValidPlacement, boardRef
    };
}
