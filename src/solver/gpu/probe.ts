// A device that failed once is not tried again for a while, so a broken driver does not cost every solve a failed start
const FAILURE_KEY = 'optimizer_gpu_failed_at';
const RETRY_AFTER_MS = 24 * 60 * 60 * 1000;

export const gpuSupported = () => typeof navigator !== 'undefined' && !!navigator.gpu;

export const gpuMayWork = () => {
    if (!gpuSupported()) return false;
    if (typeof localStorage === 'undefined') return true;
    const failedAt = Number(localStorage.getItem(FAILURE_KEY));
    return !(failedAt > 0 && Date.now() - failedAt < RETRY_AFTER_MS);
};

export const markGpuFailed = () => {
    if (typeof localStorage !== 'undefined') localStorage.setItem(FAILURE_KEY, String(Date.now()));
};
