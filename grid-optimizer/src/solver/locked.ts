import type { InventoryItem } from '../types';

// Alarm / Junk Processing / Blast modules exist for reasons the grid does not model
// On stats alone they are neutral at best and negative at worst, so a stat optimizer left to its own devices either ignores them or, worse, treats them as free filler
// Deciding how many of them a build should carry is a separate question from maximising stats, so they start out locked; unlocking one hands it to the optimizer like any other module
export const isSpecialModule = (item: InventoryItem) =>
    item.displayName.includes('Alarm Transmitter Module')
    || item.displayName.includes('(Junk Processing)')
    || item.displayName.includes('(Blast)');

// A locked module belongs to whichever machine holds it: the solver and the user may move it around that board, but neither adds one to a board or takes one off
export const isLockedModule = (item: InventoryItem) => item.isLocked ?? isSpecialModule(item);
