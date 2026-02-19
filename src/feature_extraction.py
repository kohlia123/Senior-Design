import numpy as np
import scipy.stats as sp_stats

# Sharpness
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


# Duration 
def ied_duration_ms(epoch: np.ndarray,
                    sfreq: float,
                    onset_idx: int,
                    peak_search_ms: float = 30.0,
                    min_ms: float = 0.0) -> float:
    """
    Duration = time (ms) between half-amplitude crossings around the peak.
    Peak is searched in a window around onset_idx.
    """
    x = np.asarray(epoch, dtype=float).ravel()
    if x.size < 3:
        return 0.0

    # Robust baseline correction
    x = x - np.median(x)

    n = x.size
    onset_idx = int(np.clip(onset_idx, 0, n - 1))

    # Search for the peak near onset
    r = int(round((peak_search_ms / 1000.0) * sfreq))
    lo = max(0, onset_idx - r)
    hi = min(n, onset_idx + r + 1)
    peak_idx = lo + int(np.argmax(np.abs(x[lo:hi])))

    peak_amp = float(np.abs(x[peak_idx]))
    if peak_amp <= 1e-12:
        return 0.0

    half = 0.5 * peak_amp

    # Walk left/right until dropping below half amplitude
    left = peak_idx
    while left > 0 and np.abs(x[left]) >= half:
        left -= 1

    right = peak_idx
    while right < n - 1 and np.abs(x[right]) >= half:
        right += 1

    dur_samples = max(0, right - left - 1)
    dur_ms = 1000.0 * dur_samples / float(sfreq)

    return dur_ms if dur_ms >= float(min_ms) else 0.0


# Amplitude
def get_ptp_amplitude(epochs):
    return np.ptp(epochs, axis=1)