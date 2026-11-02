import React from 'react';
import type { InventoryItem, ItemEffect, ModuleColor, Stats } from '../types';
import { COLOR_MAP, EFFECTS_LIST } from '../constants';
import { applyInternalEffects, formatStatValue, getBaseStats, getStatColor, isFixedStatModule } from '../utils';
import { isLockedModule } from '../solver/locked';
import { STAT_KEYS } from '../solver/objective';
import MiniShape from './MiniShape';

const STAT_ABBREVIATIONS: Record<keyof Stats, string> = { Performance: 'Perf', Quality: 'Qual', Efficiency: 'Eff' };

type InventoryItemRowProps = {
    item: InventoryItem;
    isAnySolving: boolean;
    updateItemEffect: (item: InventoryItem, effectIndex: 0 | 1, newEffect: ItemEffect) => void;
    updateItemEffectValue: (itemId: string, effectIndex: 0 | 1, newValue: number) => void;
    handleBlurEffectValue: (item: InventoryItem, effectIndex: 0 | 1, rawValue: number) => void;
    onRemove: (itemId: string) => void;
    onDragStart: (e: React.MouseEvent, item: InventoryItem) => void;
    onToggleInfinite: (isInfinite: boolean) => void;
    onToggleLock: (itemId: string, isLocked: boolean) => void;
};

export const InventoryItemRow = React.memo(({ item, isAnySolving, updateItemEffect, updateItemEffectValue, handleBlurEffectValue, onRemove, onDragStart, onToggleInfinite, onToggleLock }: InventoryItemRowProps) => {
    const locked = isLockedModule(item);
    const fixedBase = isFixedStatModule(item) ? getBaseStats(item) : null;
    const shownStats = item.color !== 'White' ? (fixedBase ?? applyInternalEffects(item)) : null;
    return (
        <div
            onMouseDown={(e) => onDragStart(e, item)}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: '#252526', borderRadius: '4px', borderLeft: `4px solid ${COLOR_MAP[item.color as ModuleColor]}`, cursor: isAnySolving || item.isStuck ? 'default' : 'grab' }}
        >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', width: '100%' }}>
                <MiniShape shape={item.shape} colorHex={COLOR_MAP[item.color as ModuleColor]} size="10px" />

                <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: '4px', pointerEvents: 'none' }}>
                    <span style={{ fontSize: '0.9em', fontWeight: 'bold', color: '#eee' }}>{item.displayName}</span>

                    {shownStats && (
                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', fontSize: '0.75em', fontWeight: 'bold' }}>
                            {STAT_KEYS.map(stat => shownStats[stat] !== 0 && (
                                <span key={stat} style={{ color: getStatColor(shownStats[stat]) }}>
                                    {STAT_ABBREVIATIONS[stat]} {formatStatValue(shownStats[stat])}
                                </span>
                            ))}
                            {STAT_KEYS.every(stat => shownStats[stat] === 0) && <span style={{ color: '#888' }}>No stat changes</span>}
                        </div>
                    )}

                    {fixedBase ? null : item.color !== 'White' ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', width: '100%', pointerEvents: 'auto' }}>
                            {[0, 1].map((effectIdx) => {
                                const currentEffect = item.effects[effectIdx];
                                const showCustomInput = currentEffect === 'Learning Algorithm' || currentEffect === 'Degrading';

                                return (
                                    <div key={effectIdx} style={{ display: 'flex', gap: '6px', alignItems: 'center', width: '100%' }} onMouseDown={(e) => e.stopPropagation()}>
                                        <select
                                            value={currentEffect}
                                            onChange={(e) => updateItemEffect(item, effectIdx as 0 | 1, e.target.value as ItemEffect)}
                                            disabled={isAnySolving}
                                            style={{ flex: 1, padding: '2px', fontSize: '0.7em', backgroundColor: '#111', color: '#eee', border: '1px solid #444', borderRadius: '3px', minWidth: '0', cursor: isAnySolving ? 'not-allowed' : 'pointer', opacity: isAnySolving ? 0.6 : 1 }}
                                        >
                                            {EFFECTS_LIST.filter(eff => eff === 'None' || eff !== item.effects[effectIdx === 0 ? 1 : 0]).map(eff => (
                                                <option key={eff} value={eff}>{eff === 'None' ? 'No Effect' : eff}</option>
                                            ))}
                                        </select>

                                        {showCustomInput && (
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }} title="Custom Percentage Value">
                                                <input
                                                    type="number"
                                                    value={item.effectValues[effectIdx]}
                                                    onChange={(e) => updateItemEffectValue(item.id, effectIdx as 0 | 1, Number(e.target.value))}
                                                    onBlur={(e) => handleBlurEffectValue(item, effectIdx as 0 | 1, Number(e.target.value))}
                                                    disabled={isAnySolving}
                                                    style={{ width: '48px', padding: '1px', fontSize: '0.7em', backgroundColor: '#111', color: '#eee', border: '1px solid #444', borderRadius: '3px', textAlign: 'center', cursor: isAnySolving ? 'not-allowed' : 'auto', opacity: isAnySolving ? 0.6 : 1 }}
                                                />
                                                <span style={{ fontSize: '0.65em', color: '#aaa' }}>%</span>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div style={{ display: 'flex', alignItems: 'center', width: '100%', pointerEvents: 'auto' }} onMouseDown={(e) => e.stopPropagation()}>
                            <label style={{ fontSize: '0.75em', color: '#ccc', display: 'flex', alignItems: 'center', gap: '6px', cursor: isAnySolving ? 'not-allowed' : 'pointer' }}>
                                <input
                                    type="checkbox"
                                    checked={!!item.isInfinite}
                                    onChange={(e) => onToggleInfinite(e.target.checked)}
                                    disabled={isAnySolving}
                                    style={{ margin: 0, cursor: isAnySolving ? 'not-allowed' : 'pointer' }}
                                />
                                Infinite Nodes
                            </label>
                        </div>
                    )}
                </div>
            </div>

            <div style={{ display: 'flex', marginLeft: '8px', alignItems: 'center', gap: '4px' }}>
                <button
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={() => onToggleLock(item.id, !locked)}
                    disabled={isAnySolving || item.isStuck}
                    title={item.isStuck ? 'A stuck module cannot be moved or unlocked' : 'The optimizer never adds, removes or moves a locked module to another machine, only rearranges it on its board; you can place it anywhere'}
                    style={{
                        background: locked ? 'rgba(255, 77, 77, 0.1)' : 'transparent',
                        border: `1px solid ${locked ? '#ff4d4d' : '#555'}`,
                        color: locked ? '#ff4d4d' : '#aaa',
                        borderRadius: '4px',
                        padding: '4px 8px',
                        cursor: isAnySolving ? 'not-allowed' : 'pointer',
                        fontSize: '0.75em',
                        fontWeight: 'bold',
                        minWidth: '60px'
                    }}
                >
                    {item.isStuck ? 'Stuck' : locked ? 'Unlock' : 'Lock'}
                </button>
                <button
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={() => onRemove(item.id)}
                    disabled={isAnySolving}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: isAnySolving ? '#444' : '#666',
                        cursor: isAnySolving ? 'not-allowed' : 'pointer',
                        fontSize: '1.4em',
                        padding: '8px',
                        marginRight: '-5px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}
                >
                    &times;
                </button>
            </div>
        </div>
    );
});
