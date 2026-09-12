import { useState, useRef, useEffect } from 'react';
import type { GridTier, InventoryItem, Stats, TargetStats, Point } from '../types';
import { saveToDatabase } from '../leaderboard';
import { type Board, initializeBoard } from '../solver/board';
import { calculateBoardStats, indexInventoryById } from '../solver/boardStats';
import { decodeSolution, generateCodeFromState, inventoryForCode } from '../solver/codec';
import type { BoardUpdate } from '../solver/report';
import { isLockedModule } from '../solver/locked';

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

    const [targetStats, setTargetStats] = useState<TargetStats>(savedState?.targetStats ?? { Performance: null, Quality: null, Efficiency: null });
    const [maximizeStats, setMaximizeStats] = useState(savedState?.maximizeStats ?? { Performance: false, Quality: false, Efficiency: false });
    const [ignoreStats, setIgnoreStats] = useState(savedState?.ignoreStats ?? { Performance: false, Quality: false, Efficiency: false });
    const [statPriority, setStatPriority] = useState(savedState?.statPriority ?? { Performance: 1, Quality: 1, Efficiency: 1 });

    const [board, setBoard] = useState<Board>(() => initializeBoard(savedState?.tier ?? defaultTier, savedState?.boardIds, inventory));
    const boardRef = useRef(board);
    const setBoardSync = (newBoard: Board) => {
        boardRef.current = newBoard;
        setBoard(newBoard);
    };

    const [isInitializedFromSave, setIsInitializedFromSave] = useState(false);

    useEffect(() => {
        if (!isInitializedFromSave && inventory.length > 0 && savedState?.boardIds) {
            const initialized = initializeBoard(tier, savedState.boardIds, inventory);
            setBoardSync(initialized);
            setIsInitializedFromSave(true);
        }
    }, [inventory, tier, savedState, isInitializedFromSave]);

    const [bestTotals, setBestTotals] = useState<Stats>({ Performance: 0, Quality: 0, Efficiency: 0 });
    const [bestPieceStats, setBestPieceStats] = useState<Map<string, Stats>>(new Map());

    const [warningMsg, setWarningMsg] = useState<string | null>(null);
    const [solutionCode, setSolutionCode] = useState<string>('');

    const getAvailableInventory = () => {
        if (!getUsedItems || !machineId) return inventory.filter(i => !i.isLocked);
        const used = getUsedItems(machineId);
        return inventory.filter(item => !used.has(item.id) && !item.isLocked);
    };

    const getInventoryForCode = () => {
        return inventory;
    };

    const handleTierChange = (newTier: GridTier) => {
        setTier(newTier);
        setBoardSync(initializeBoard(newTier));
        setBestTotals({ Performance: 0, Quality: 0, Efficiency: 0 });
        setBestPieceStats(new Map());
        setWarningMsg(null);
        setSolutionCode('');
    };

    const emptyBoard = (keepLockedModules: boolean) => {
        const fresh = initializeBoard(tier);
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

    const importSolution = (code: string) => {
        try {
            const decoded = decodeSolution(code);
            setTier(decoded.tier);
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
            const availableInventory = getAvailableInventory();
            const { totals, pieceStats } = calculateBoardStats(boardToCalculate, availableInventory, indexInventoryById(availableInventory));

            setBestTotals(totals);
            setBestPieceStats(new Map(pieceStats));
            if (boardChanged) setBoardSync(newBoard);

            const boardIds = boardToCalculate.map(row => row.map(c => c && c !== 'Locked' ? c.id : c));
            localStorage.setItem(`optimizer_machine_${machineId}`, JSON.stringify({
                tier, maximizeStats, targetStats, ignoreStats, statPriority, boardIds
            }));

            if (inventory.length > 0) {
                const availableForCode = inventoryForCode(getInventoryForCode(), boardToCalculate);
                const newCode = generateCodeFromState(tier, maximizeStats, targetStats, availableForCode, boardToCalculate);
                setSolutionCode(newCode);

                // Never publish a run that has no representable code
                // It is an inventory past the 8-bit module count the format allows
                if (newCode) {
                    if (totals.Performance !== 0 || totals.Quality !== 0 || totals.Efficiency !== 0) {
                        const timer = setTimeout(() => saveToDatabase(tier, totals, newCode, availableForCode), 30000); // save timer
                        return () => clearTimeout(timer);
                    }
                }
            } else {
                setSolutionCode('');
            }
        }
    }, [inventory, tier, maximizeStats, targetStats, ignoreStats, statPriority, machineId, getUsedItems, board, isAnySolving]);

    return {
        tier, setTier, handleTierChange, targetStats, setTargetStats,
        maximizeStats, setMaximizeStats, ignoreStats, setIgnoreStats,
        statPriority, setStatPriority, board, bestTotals, bestPieceStats,
        warningMsg, setWarningMsg, applyUpdate,
        solutionCode, setSolutionCode, importSolution, resetBoard, resetBoardIncludingLocked,
        manuallyPlaceItem, manuallyRemoveItem, isValidPlacement, boardRef
    };
}
