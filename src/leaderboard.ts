import { createClient } from '@supabase/supabase-js';
import type { GridTier, InventoryItem, Stats } from './types';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://yhiojdutwgfxrgakbrjs.supabase.co';
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InloaW9qZHV0d2dmeHJnYWticmpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODY0ODcwMjMsImV4cCI6MjEwMjA2MzAyM30.lVkU06tLfM64aFYL2Gx-UMPFL9KCRSaadu58TDWMmSI';
const supabase = createClient(supabaseUrl, supabaseKey);

export const saveToDatabase = (
    currentTier: GridTier,
    totals: Stats,
    code: string,
    inv: InventoryItem[]
) => {
    const hasNeuralCore = inv.some(item => item.displayName.includes('Neural Core'));
    const averageStat = (totals.Performance + totals.Quality + totals.Efficiency) / 3;

    const submission = {
        tier: currentTier,
        has_neural_core: hasNeuralCore,
        performance: totals.Performance,
        quality: totals.Quality,
        efficiency: totals.Efficiency,
        average_stat: parseFloat(averageStat.toFixed(2)),
        solution_code: code
    };

    supabase.from('leaderboards').insert([submission]).then(({ error }) => {
        if (error) console.error(error);
    });
};
