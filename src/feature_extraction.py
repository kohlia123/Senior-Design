import numpy as np
from scipy.signal import butter, filtfilt, welch
from src.utils.plotting import plot_epoch
import matplotlib.pyplot as plt
import scipy.stats as sp_stats

# Sharpness
def feat_sharpness(epoch_1ch: np.ndarray,
                   sfreq: float,
                   window_ms: float = 5.0) -> float:
    x = np.asarray(epoch_1ch, dtype=float)

    # Epoch sample size
    n = x.size

    # Not enough points to compute slope → return NaN
    if n < 3:
        return np.nan

    # Find index of the peak (maximum absolute amplitude)
    peak_idx = int(np.argmax(np.abs(x)))

    # Convert window size from ms → samples
    w = max(1, int(round((window_ms / 1000.0) * sfreq)))

    # Define indices around the peak for slope computation
    left_idx = max(0, peak_idx - w)
    right_idx = min(n - 1, peak_idx + w)

    # If the window coincides with the peak itself (too small) → return NaN
    if left_idx == peak_idx or right_idx == peak_idx:
        return np.nan

    # Compute time differences (in seconds) between peak and window boundaries
    dt_left = (peak_idx - left_idx) / sfreq
    dt_right = (right_idx - peak_idx) / sfreq

    # Compute slope on left side of peak
    slope_left = abs((x[peak_idx] - x[left_idx]) / dt_left)
    slope_right = abs((x[right_idx] - x[peak_idx]) / dt_right)

    # Return the average slope as the sharpness measure
    avg_slope = 0.5 * (slope_left + slope_right)

    return avg_slope


def _bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)


# Slow afterwave
def feat_slow_afterwave(epoch_1ch: np.ndarray,
                        sfreq: float,
                        spike_index: int,
                        min_latency_ms: float = 20.0,
                        max_latency_ms: float = 300.0,
                        min_ratio: float = 0.2,
                        min_duration_ms: float = 40.0):

    x = np.asarray(epoch_1ch, dtype=float)

    # Epoch sample size
    n = x.size

    # Not enough points to analyze afterwave → return False with empty features
    if n < 3:
        return False, {}

    # Ensure spike index is within valid bounds
    spike_index = int(np.clip(spike_index, 0, n-1))

    # Compute spike amplitude (absolute, since spike can be positive or negative)
    spike_amp = np.abs(x[spike_index])

    # If spike amplitude is zero, cannot compute meaningful ratio → return
    if spike_amp == 0:
        return False, {}

    # Define search window after the spike
    # Convert latency bounds from milliseconds to samples
    min_samples = int((min_latency_ms / 1000.0) * sfreq)
    max_samples = int((max_latency_ms / 1000.0) * sfreq)

    # Define search window starting after the spike
    start = spike_index + min_samples
    end = min(spike_index + max_samples, n)

    # If window is outside signal bounds → no afterwave possible
    if start >= n:
        return False, {}

    # Extract post-spike segment
    window = x[start:end]

    # Isolate slow activity (afterwave)
    # Bandpass filter to keep slow frequencies (1–4 Hz typical slow wave)
    slow = _bandpass(window, sfreq, 1.0, 4.0)

    # Find peak of slow wave (max absolute amplitude)
    peak_idx = np.argmax(np.abs(slow))
    slow_amp = np.abs(slow[peak_idx])

    # Compute latency of slow wave peak relative to spike (in ms)
    latency_samples = peak_idx + min_samples
    latency_ms = (latency_samples / sfreq) * 1000

    # Estimate slow wave duration
    # Define threshold as 50% of slow wave peak amplitude
    threshold = 0.5 * slow_amp

    # Count samples where slow signal exceeds half amplitude
    duration_samples = np.sum(np.abs(slow) > threshold)

    # Convert duration to milliseconds
    duration_ms = (duration_samples / sfreq) * 1000

    # Compute amplitude ratio
    # Ratio between slow wave amplitude and spike amplitude
    amp_ratio = slow_amp / spike_amp

    # Determine if slow afterwave is present
    slow_present = (
        amp_ratio >= min_ratio and                      # slow wave is large enough
        duration_ms >= min_duration_ms and              # slow wave lasts long enough
        min_latency_ms <= latency_ms <= max_latency_ms  # occurs in expected time window
    )

    # Store extracted features
    features = {
        "slow_amplitude": slow_amp,
        "spike_amplitude": spike_amp,
        "amplitude_ratio": amp_ratio,
        "latency_ms": latency_ms,
        "duration_ms": duration_ms
    }

    return slow_present, features


# Background disruption
def feat_background_disruption(epoch_1ch: np.ndarray,
                               sfreq: float,
                               spike_index: int,
                               pre_ms: float = 500.0,
                               post_ms: float = 500.0,
                               guard_ms: float = 50.0,
                               prev_epoch: np.ndarray = None,
                               next_epoch: np.ndarray = None):

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

    # If PRE is too short, borrow from prev_epoch (take from its end)
    if len(pre) < pre_len and prev_epoch is not None:
        prev = np.asarray(prev_epoch, dtype=float)
        prev = prev - np.mean(prev)
        need = pre_len - len(pre)
        take = min(len(prev), need)
        if take > 0:
            pre = np.concatenate([prev[-take:], pre])

    if len(pre) < int(0.05 * sfreq):
        return {}

    # post background
    post_start = min(n, spike_index + guard)
    post_end = min(n, post_start + post_len)
    post = x[post_start:post_end]

     # If POST is too short, borrow from next_epoch (take from its beginning)
    if len(post) < post_len and next_epoch is not None:
        nxt = np.asarray(next_epoch, dtype=float)
        nxt = nxt - np.mean(nxt)
        need = post_len - len(post)
        take = min(len(nxt), need)
        if take > 0:
            post = np.concatenate([post, nxt[:take]])

    event_start = max(0, spike_index - int(0.025 * sfreq))
    event_end = min(n, spike_index + int(0.05 * sfreq))
    event = x[event_start:event_end]
    if len(event) < 2:
        return {}
    
    # time-domain
    background_rms = np.sqrt(np.mean(pre**2))
    background_std = np.std(pre)
    background_line_length = np.sum(np.abs(np.diff(pre)))

    eps = 1e-12
    event_rms = np.sqrt(np.mean(event**2))
    event_rms_ratio_bg = event_rms / (background_rms + eps)

    post_rms = np.sqrt(np.mean(post**2)) if len(post) >= int(0.05 * sfreq) else np.nan
    event_rms_ratio_post = event_rms / (post_rms + eps) if np.isfinite(post_rms) else np.nan
    post_rms_ratio_pre = post_rms / (background_rms + eps) if np.isfinite(post_rms) else np.nan

    # frequency-domain
    nperseg = min(len(pre), 256)
    f, psd = welch(pre, fs=sfreq, nperseg=nperseg)

    delta_power = np.mean(psd[(f >= 1) & (f <= 4)]) if np.any((f >= 1) & (f <= 4)) else 0.0
    alpha_power = np.mean(psd[(f >= 8) & (f <= 13)]) if np.any((f >= 8) & (f <= 13)) else 0.0


    return {
        "background_rms": background_rms,
        "background_std": background_std,
        "background_line_length": background_line_length,
        "background_delta_power": delta_power,
        "background_alpha_power": alpha_power,
        "event_rms_ratio_bg": event_rms_ratio_bg,
        "post_rms_ratio_pre": post_rms_ratio_pre,
        "event_rms_ratio_post":event_rms_ratio_post,

    }


# Duration 
def ied_duration_ms(epoch: np.ndarray,
                    sfreq: float,
                    onset_idx: int = None,
                    k: float = 3.0,
                    peak_search_ms: float = 30.0,
                    min_ms: float = 5.0) -> float:
    """
    Estimate the duration of an interictal epileptiform discharge (IED) in milliseconds.

    The duration is defined as the contiguous time interval around the spike peak
    during which the signal amplitude exceeds a threshold based on the estimate
    of baseline variability (Median Absolute Deviation, MAD).

    Specifically:
    - A threshold is computed as k × (1.4826 × MAD), where MAD is calculated from the epoch.
    - Samples with absolute amplitude above this threshold are considered "active".
    - The spike peak is identified either globally or within a window around a provided onset.
    - The duration is computed by expanding left and right from the peak across contiguous
      active samples.
    - Only durations above `min_ms` are returned; otherwise, 0 is returned.

    Notes
    -----
    - This method captures the high-amplitude (sharp) component of the IED, rather than
      the full baseline-to-baseline duration typically reported in clinical definitions
      (e.g., 20–70 ms).
    - The input signal is assumed to be preprocessed (e.g., z-score normalized), so
      additional baseline correction is not applied.

    Parameters
    ----------
    epoch : np.ndarray
        1D array containing the EEG signal for a single epoch.
    sfreq : float
        Sampling frequency in Hz.
    onset_idx : int, optional
        Approximate index of the spike onset. If provided, the peak is searched within
        ±peak_search_ms around this index. If None, the global maximum is used.
    k : float, default=3.0
        Multiplier for the MAD-based threshold.
    peak_search_ms : float, default=30.0
        Time window (in milliseconds) around onset_idx to search for the spike peak.
    min_ms : float, default=5.0
        Minimum duration (in milliseconds) required to consider a valid IED.

    Returns
    -------
    float
        Estimated IED duration in milliseconds. Returns 0.0 if no valid IED is detected.
    """

    # Ensure epoch is a float array and it has enough samples to compute duration
    x = np.asarray(epoch, dtype=float)
    if x.ndim != 1 or x.size < 3:  # Not enough samples to compute duration
        return 0.0

    # x = x - np.median(x)  # keep commented! since data is already z-scored, so median should be ~0

    # Estimate signal scale using Median Absolute Deviation (MAD)
    mad = np.median(np.abs(x - np.median(x)))
    scale = 1.4826 * mad  # Converts MAD to standard deviation equivalent
    if scale <= 1e-12:  # Avoid division by zero / flat signal
        return 0.0

    # Define threshold for “active” spike samples
    thr = k * scale
    active = np.abs(x) >= thr  # marks which samples are big enough
    if not np.any(active):  # No samples exceed threshold → no IED
        return 0.0

    # Find peak index: if onset is given, search for local peak around it; otherwise, take global max
    if onset_idx is None:
        peak_idx = int(np.argmax(np.abs(x)))
    else:
        # Clip onset index to valid range
        onset_idx = int(np.clip(onset_idx, 0, x.size - 1))

        # Convert peak search window from ms to samples
        r = int((peak_search_ms / 1000.0) * sfreq)

        # Define local window around onset to find the true peak
        lo = max(0, onset_idx - r)
        hi = min(x.size, onset_idx + r + 1)

        # Find local peak within this window
        peak_local = int(np.argmax(np.abs(x[lo:hi])))
        peak_idx = lo + peak_local

    # If the identified peak is not active (below threshold), fallback to strongest active sample
    if not active[peak_idx]:
        # fallback: pick the strongest active sample
        active_indices = np.where(active)[0]
        if active_indices.size == 0:
            return 0.0  # safety check (shouldn't happen)

        peak_idx = active_indices[np.argmax(np.abs(x[active_indices]))]

    # Expand left from peak until signal falls below threshold
    left = peak_idx
    while left > 0 and active[left]:
        left -= 1

    # Expand right from peak until signal falls below threshold
    right = peak_idx
    while right < x.size - 1 and active[right]:
        right += 1

    # Compute duration in samples
    dur_samples = (right - 1) - (left + 1) + 1
    if dur_samples <= 0:
        return 0.0

    # Return duration if above minimum threshold, otherwise return 0
    dur_ms = 1000.0 * dur_samples / float(sfreq)
    if dur_ms < min_ms:
        return 0.0

    # Auxiliary function for testing: visualize the epoch with the spike marked
    # fig, ax = plot_epoch(x, sfreq, spike_idx=peak_idx, mad=scale, k=k, active=active, title=f"k = {k}")
    # plt.show()

    return dur_ms


# Amplitude
def get_ptp_amplitude(epochs):
    return np.ptp(epochs, axis=1)


#%%
