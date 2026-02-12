import numpy as np

def ied_duration_ms(epoch: np.ndarray,
                    sfreq: float,
                    onset_idx: int,
                    k: float = 4.0,
                    peak_search_ms: float = 30.0,
                    min_ms: float = 5.0) -> float:
    x = np.asarray(epoch, dtype=float)
    if x.ndim != 1 or x.size < 3:
        return 0.0

    x = x - np.median(x)

    mad = np.median(np.abs(x - np.median(x)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0

    thr = k * scale #setting threshold
    active = np.abs(x) >= thr #marks which samples are big enough 
    if not np.any(active):
        return 0.0

    r = int((peak_search_ms / 1000.0) * sfreq)
    onset_idx = int(np.clip(onset_idx, 0, x.size - 1))
    lo = max(0, onset_idx - r)
    hi = min(x.size, onset_idx + r + 1)

    peak_local = int(np.argmax(np.abs(x[lo:hi])))
    peak_idx = lo + peak_local

    if not active[peak_idx]:
        peak_idx = int(np.argmax(np.abs(x)))
        if not active[peak_idx]:
            return 0.0

    left = peak_idx
    while left > 0 and active[left]:
        left -= 1
    right = peak_idx
    while right < x.size - 1 and active[right]:
        right += 1

    dur_samples = (right - 1) - (left + 1) + 1
    if dur_samples <= 0:
        return 0.0

    dur_ms = 1000.0 * dur_samples / float(sfreq)
    return dur_ms if dur_ms >= float(min_ms) else 0.0