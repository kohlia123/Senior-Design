import numpy as np

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

