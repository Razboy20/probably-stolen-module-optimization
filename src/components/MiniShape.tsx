import type {ModuleShape} from '../types';
import { SHAPE_DEFINITIONS } from '../constants';

interface MiniShapeProps {
    shape: ModuleShape;
    colorHex: string;
    size?: string;
}

export default function MiniShape({ shape, colorHex, size = '14px' }: MiniShapeProps) {
    const layout = SHAPE_DEFINITIONS[shape];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0px', justifyContent: 'center' }}>
            {layout.map((row, rIdx) => (
                <div key={rIdx} style={{ display: 'flex', gap: '0px' }}>
                    {row.map((cell, cIdx) => (
                        <div
                            key={cIdx}
                            style={{
                                width: size,
                                height: size,
                                backgroundColor: cell ? colorHex : 'transparent',
                                border: cell ? '1px solid #000' : 'none'
                            }}
                        />
                    ))}
                </div>
            ))}
        </div>
    );
}