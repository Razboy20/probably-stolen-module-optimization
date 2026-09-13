import React, { useRef, useState } from 'react';
import { parseSaveFile, type ImportedMachine } from '../save/parseSave';
import type { InventoryItem } from '../types';

interface SaveFileImporterProps {
    onImport: (newItems: InventoryItem[], newMachines: ImportedMachine[]) => void;
}

export default function SaveFileImporter({ onImport }: SaveFileImporterProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setErrorMsg(null);

        const reader = new FileReader();
        reader.onload = (evt) => {
            try {
                const { items, machines } = parseSaveFile(evt.target?.result as string);
                onImport(items, machines);
            } catch (err) {
                console.error("Failed to parse save:", err);
                setErrorMsg("Failed to parse save file. Please ensure it is a valid .es3 save string.");
            }

            if (fileInputRef.current) fileInputRef.current.value = '';
        };
        reader.readAsText(file);
    };

    return (
        <div style={{ display: 'inline-block', position: 'relative' }}>
            <input
                type="file"
                accept=".es3"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="save-upload"
            />
            <label
                htmlFor="save-upload"
                style={{
                    padding: '10px 24px',
                    backgroundColor: '#333333',
                    color: '#eee',
                    border: '1px solid #555555',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.95em',
                    display: 'inline-block'
                }}
            >
                Import Save (.es3)
            </label>
            {errorMsg && (
                <div style={{ position: 'absolute', top: '100%', marginTop: '5px', left: 0, color: '#ff4d4d', fontSize: '0.8em', whiteSpace: 'nowrap' }}>
                    {errorMsg}
                </div>
            )}
        </div>
    );
}
