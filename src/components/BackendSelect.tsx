import { useState } from 'react';
import { readBackendPreference, SOLVER_BACKENDS, type SolverBackend, writeBackendPreference } from '../solver/client';

const BackendSelect = ({ disabled }: { disabled: boolean }) => {
    const [backend, setBackend] = useState<SolverBackend>(readBackendPreference);
    const onChange = (value: SolverBackend) => {
        writeBackendPreference(value);
        setBackend(value);
    };
    return (
        <label title="Where the optimizer runs. Auto picks the fastest available." style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85em', color: '#aaa' }}>
            Solver:
            <select
                value={backend}
                disabled={disabled}
                onChange={e => onChange(e.target.value as SolverBackend)}
                style={{ padding: '6px 8px', backgroundColor: '#111', color: '#eee', border: '1px solid #444', borderRadius: '6px', fontSize: '0.95em' }}
            >
                {SOLVER_BACKENDS.map(b => <option key={b.value} value={b.value}>{b.label}</option>)}
            </select>
        </label>
    );
};

export default BackendSelect;
