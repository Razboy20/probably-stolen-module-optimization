import type { FilterGroup, InventoryItem, ItemEffect, MachineKind, ModuleTemplate } from './types';
import { MACHINE_GROUP_ORDER, MODULE_TEMPLATES } from './constants';
import { compareTemplatesByMachine, getModuleTemplate, PRECOMPUTED_OFFSETS } from './utils';

export type SizeFilter = 'All' | 2 | 3 | 4 | 5;
export type MachineFilter = 'All' | 'None' | MachineKind;
export type InventoryGroupFilter = FilterGroup | 'Placed' | 'NotPlaced';

// Only the machines some module is made for
export const MACHINE_FILTER_OPTIONS = MACHINE_GROUP_ORDER.filter(kind => MODULE_TEMPLATES.some(t => t.allowedMachines?.includes(kind)));

const matchesMachineFilter = (allowedMachines: MachineKind[] | undefined, filter: MachineFilter) => {
    if (filter === 'All') return true;
    if (filter === 'None') return !allowedMachines?.length;
    return !!allowedMachines?.includes(filter);
};

export interface CatalogFilters {
    group: FilterGroup;
    size: SizeFilter;
    machine: MachineFilter;
}

export const filterCatalog = ({ group, size, machine }: CatalogFilters) => MODULE_TEMPLATES.filter(m => {
    if (m.shapeType === 'Node') return false;
    if (group !== 'All' && m.group !== group) return false;
    if (size !== 'All' && m.size !== size) return false;
    return matchesMachineFilter(m.allowedMachines, machine);
}).sort(compareTemplatesByMachine);

export interface InventoryFilters {
    group: InventoryGroupFilter;
    size: SizeFilter;
    effect: ItemEffect | 'All';
    machine: MachineFilter;
}

// An infinite module counts as placed when any of its clones is
const isPlaced = (item: InventoryItem, placedIds: ReadonlySet<string>) =>
    placedIds.has(item.id) || (Boolean(item.isInfinite) && [...placedIds].some(id => id.startsWith(item.id + '_clone_')));

const matchesEffectFilter = (item: InventoryItem, effect: ItemEffect | 'All') => {
    if (effect === 'All') return true;
    if (effect === 'None') return item.effects[0] === 'None' && item.effects[1] === 'None';
    return item.effects.includes(effect);
};

const matchesGroupFilter = (item: InventoryItem, template: ModuleTemplate | undefined, group: InventoryGroupFilter, placedIds: ReadonlySet<string>) => {
    if (group === 'All') return true;
    if (group === 'Placed') return isPlaced(item, placedIds);
    if (group === 'NotPlaced') return !isPlaced(item, placedIds);
    return template?.group === group;
};

export const filterInventory = (inventory: InventoryItem[], { group, size, effect, machine }: InventoryFilters, placedIds: ReadonlySet<string>) =>
    inventory.filter(item => {
        const template = getModuleTemplate(item);
        if (!matchesGroupFilter(item, template, group, placedIds)) return false;
        if (!matchesMachineFilter(template?.allowedMachines, machine)) return false;
        if (size !== 'All' && (PRECOMPUTED_OFFSETS.get(item.shape)?.[0].length ?? 0) !== size) return false;
        return matchesEffectFilter(item, effect);
    });
