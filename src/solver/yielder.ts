export const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

// The solver has to hand the event loop back periodically or the page freezes, but `setTimeout(..., 0)` is clamped to ~4ms once it nests a few levels deep,
// which caps the search at a few hundred iterations per second no matter how fast the search itself is
// A MessagePort task is not clamped, so most yields go through one of those
// Leaning on it exclusively is not safe though: a steady stream of port messages can crowd out timer callbacks entirely,
// so every so often we yield through a timer instead to guarantee they still get serviced
export const createYielder = () => {
    const timerYield = () => new Promise<void>(resolve => { setTimeout(resolve, 0); });

    if (typeof MessageChannel === 'undefined') {
        return { portYield: timerYield, timerYield, dispose: () => {} };
    }

    const channel = new MessageChannel();
    let pending: (() => void) | null = null;
    channel.port1.onmessage = () => {
        const resolve = pending;
        pending = null;
        if (resolve) resolve();
    };

    return {
        portYield: () => new Promise<void>(resolve => {
            pending = resolve;
            channel.port2.postMessage(0);
        }),
        timerYield,
        dispose: () => {
            channel.port1.onmessage = null;
            channel.port1.close();
            channel.port2.close();
        }
    };
};

// How long the solver may run before handing control back to the host
// One frame's worth of work per yield keeps the UI responsive while amortising the cost of yielding.
export const FRAME_BUDGET_MS = 12;
// How often one of those yields must be a timer yield, so timer callbacks cannot starve.
export const TIMER_YIELD_INTERVAL_MS = 32;
