import type { InventoryItem } from '../types';
import { isFixedStatModule } from '../utils';

// Alarm, Furnace and the machine-specific specials exist for reasons the grid does not model
// On stats alone they are neutral at best and negative at worst, so a stat optimizer left to its own devices either ignores them or, worse, treats them as free filler
// Deciding how many of them a build should carry is a separate question from maximising stats, so they start out locked; unlocking one hands it to the optimizer like any other module
export const isSpecialModule = (item: InventoryItem) => isFixedStatModule(item);

// A locked module belongs to whichever machine holds it: the solver and the user may move it around that board, but neither adds one to a board or takes one off
// A stuck module is locked for good, and does not move at all
export const isLockedModule = (item: InventoryItem) => Boolean(item.isStuck) || (item.isLocked ?? isSpecialModule(item));
