import numpy as np
import scipy.stats as sp_stats


def feat_sharpness(epoch_1ch: np.ndarray, sfreq: float, window_ms: float = 5.0) -> float:
    x = np.asarray(epoch_1ch, dtype=float)
    n = x.size
    if n < 3:
        return np.nan

    peak_idx = int(np.argmax(np.abs(x)))
    w = max(1, int(round((window_ms / 1000.0) * sfreq)))

    left_idx = max(0, peak_idx - w)
    right_idx = min(n - 1, peak_idx + w)
    if left_idx == peak_idx or right_idx == peak_idx:
        return np.nan

    dt_left = (peak_idx - left_idx) / sfreq
    dt_right = (right_idx - peak_idx) / sfreq

    slope_left = abs((x[peak_idx] - x[left_idx]) / dt_left)
    slope_right = abs((x[right_idx] - x[peak_idx]) / dt_right)

    return 0.5 * (slope_left + slope_right)
  
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

def get_ptp_amplitude(epochs):
    return np.ptp(epochs, axis=1)
