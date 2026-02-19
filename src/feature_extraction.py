import numpy as np
from scipy.signal import butter, filtfilt, welch

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


def _bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def feat_slow_afterwave(epoch_1ch: np.ndarray,
                        sfreq: float,
                        spike_index: int,
                        min_latency_ms: float = 20.0,
                        max_latency_ms: float = 300.0,
                        min_ratio: float = 0.2,
                        min_duration_ms: float = 40.0):

    x = np.asarray(epoch_1ch, dtype=float)
    n = x.size

    if n < 3:
        return False, {}

    spike_index = int(np.clip(spike_index, 0, n-1))
    spike_amp = np.abs(x[spike_index])

    if spike_amp == 0:
        return False, {}

    #search window after spike
    min_samples = int((min_latency_ms / 1000.0) * sfreq)
    max_samples = int((max_latency_ms / 1000.0) * sfreq)

    start = spike_index + min_samples
    end = min(spike_index + max_samples, n)

    if start >= n:
        return False, {}

    window = x[start:end]

    # Bandpass 
    slow = _bandpass(window, sfreq, 1.0, 4.0)

    peak_idx = np.argmax(np.abs(slow))
    slow_amp = np.abs(slow[peak_idx])

    latency_samples = peak_idx + min_samples
    latency_ms = (latency_samples / sfreq) * 1000

    # Duration estimation
    threshold = 0.5 * slow_amp
    duration_samples = np.sum(np.abs(slow) > threshold)
    duration_ms = (duration_samples / sfreq) * 1000

    amp_ratio = slow_amp / spike_amp

    slow_present = (
        amp_ratio >= min_ratio and
        duration_ms >= min_duration_ms and
        min_latency_ms <= latency_ms <= max_latency_ms
    )

    features = {
        "slow_amplitude": slow_amp,
        "spike_amplitude": spike_amp,
        "amplitude_ratio": amp_ratio,
        "latency_ms": latency_ms,
        "duration_ms": duration_ms
    }

    return slow_present, features


def feat_background_disruption(epoch_1ch: np.ndarray,
                               sfreq: float,
                               spike_index: int,
                               pre_ms: float = 300.0,
                               post_ms: float = 300.0,
                               guard_ms: float = 50.0):

    x = np.asarray(epoch_1ch, dtype=float)
    n = x.size
    if n < 10:
        return {}

    
    x = x - np.mean(x)

    spike_index = int(np.clip(spike_index, 0, n - 1))

    guard = int(round((guard_ms / 1000.0) * sfreq))
    pre_len = int(round((pre_ms / 1000.0) * sfreq))
    post_len = int(round((post_ms / 1000.0) * sfreq))

    # pre background
    pre_end = max(0, spike_index - guard)
    pre_start = max(0, pre_end - pre_len)
    pre = x[pre_start:pre_end]

    if len(pre) < int(0.05 * sfreq):
        return {}

    # post background
    post_start = min(n, spike_index + guard)
    post_end = min(n, post_start + post_len)
    post = x[post_start:post_end]

    event_start = max(0, spike_index - int(0.025 * sfreq))
    event_end = min(n, spike_index + int(0.05 * sfreq))
    event = x[event_start:event_end]
    # time-domain
    background_rms = np.sqrt(np.mean(pre**2))
    event_rms = np.sqrt(np.mean(event**2))

    eps = 1e-12
    rms_ratio_event_bg = event_rms / (background_rms + eps)

    background_std = np.std(pre)
    background_line_length = np.sum(np.abs(np.diff(pre)))

    # frequency-domain
    f, psd = welch(pre, fs=sfreq, nperseg=len(pre))

    delta_power = np.mean(psd[(f >= 1) & (f <= 4)])
    alpha_power = np.mean(psd[(f >= 8) & (f <= 13)])


    return {
        "background_rms": background_rms,
        "background_std": background_std,
        "background_line_length": background_line_length,
        "background_delta_power": delta_power,
        "background_alpha_power": alpha_power,
        "event_rms_ratio_bg": rms_ratio_event_bg

    }
