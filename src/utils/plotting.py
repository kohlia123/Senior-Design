import numpy as np
import matplotlib.pyplot as plt

def plot_ied(
    epochs_dict,
    labels_dict,
    ied_onsets_dict,
    epoch_idx,
    spacing=5.0,
):
    """
    Plot ied epoch with all channels stacked. IED channels shown in red.
    """

    channels = list(epochs_dict.keys())
    n_epochs = len(next(iter(epochs_dict.values())))
    window_size = len(next(iter(epochs_dict.values()))[epoch_idx])

    # Determine valid epoch indices
    context_epochs = [epoch_idx - 1, epoch_idx, epoch_idx + 1]
    context_epochs = [ep for ep in context_epochs if 0 <= ep < n_epochs]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)

    offsets = np.arange(len(channels)) * spacing

    for i, chan in enumerate(channels):
        signals = []

        for ep in context_epochs:
            sig = epochs_dict[chan][ep]
            signals.append(sig)

        # Concatenate signals
        full_signal = np.concatenate(signals)

        # Red for annotated ied
        color = "red" if labels_dict[chan][epoch_idx] == 1 else "black"

        ax.plot(
            full_signal + offsets[i],
            color=color,
            linewidth=0.8
        )

    # Vertical separators between epochs
    for i in range(1, len(context_epochs)):
        ax.axvline(
            i * window_size,
            color="gray",
            linestyle=":",
            linewidth=1
        )

    # IED marker
    for chan in channels:
        if labels_dict[chan][epoch_idx] == 1:
            onset = ied_onsets_dict[chan][epoch_idx][0]
            ax.axvline(
                window_size + onset,
                color="red",
                linestyle="--",
                linewidth=1
            )
            break

    ax.set_xlim(0, window_size * len(context_epochs))
    ax.set_ylim(-spacing, offsets[-1] + spacing)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channels, fontsize=8)

    ax.set_xlabel("Samples")
    ax.set_title(
        f"IED epoch {epoch_idx} (with context)",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


def plot_epoch(epoch, sfreq, spike_idx=None, mad=None, k=None, active=None,
               slow_wave_idx=None,
               slow_wave=None,
               slow_wave_duration=None,
               latency=None,
               title="IED Epoch"):
    # Plot a single epoch.
    times = np.arange(len(epoch)) / sfreq  # time in seconds

    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    ax.plot(times, epoch, color='black', linewidth=1, label='EEG signal')

    if spike_idx is not None:
        ax.axvline(x=spike_idx / sfreq, color='red', linestyle='--', label='Spike onset')

    if mad is not None and k is not None:
        # Add MAD threshold area
        ax.fill_between(
            np.arange(len(epoch)) / sfreq,  # x-axis in seconds
            -k * mad,  # lower bound
            k * mad,  # upper bound
            color='red',
            alpha=0.2,
            label=f'±{k}×MAD threshold'
        )

    if active is not None:
        # Highlight active samples (above threshold)
        t = np.arange(len(epoch)) / sfreq
        ax.scatter(
            t[active],
            epoch[active],
            color='black',
            s=10,
            label='Active samples'
        )

    # Slow wave as colored segment
    if slow_wave_idx is not None and len(slow_wave_idx) > 0:
        ax.plot(times[slow_wave_idx], epoch[slow_wave_idx],
                color='green', linewidth=2.5, alpha=0.5, label='Slow wave')

    # Filtered slow signal (overlay)
    if slow_wave is not None and slow_wave_idx is not None:
        ax.plot(times[slow_wave_idx],
                slow_wave,
                color='green',
                linestyle='--',
                linewidth=1.5,
                label='Filtered slow (1–4 Hz)')

        # Peak as point
        slow_wave_peak_idx = np.argmax(np.abs(slow_wave))
        ax.scatter(times[slow_wave_peak_idx + slow_wave_idx[0]],
                   slow_wave[slow_wave_peak_idx],
                   color='green', s=40, zorder=5, label='slow_amp')

    # Duration as arrow <-->
    if slow_wave_duration is not None and len(slow_wave_duration) > 0:
        start = slow_wave_duration[0]
        end = slow_wave_duration[-1]

        # Determine polarity of slow wave
        sign = np.sign(np.mean(epoch[slow_wave_duration]))

        # Threshold where duration is defined
        slow_peak_idx = np.argmax(np.abs(slow_wave))
        slow_amp = np.abs(slow_wave[slow_peak_idx])
        y_arrow = sign * (0.5 * slow_amp)

        # Arrow
        ax.annotate(
            '',
            xy=(times[end], y_arrow),
            xytext=(times[start], y_arrow),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
        )

        # Label
        ax.text((times[start] + times[end]) / 2,
                y_arrow,
                'duration',
                color='black',
                ha='center',
                va='bottom')

    # Latency as arrow <-->
    if latency is not None:
        start = spike_idx
        end = spike_idx + latency

        # Threshold where duration is defined
        slow_peak_idx = np.argmax(np.abs(slow_wave))
        slow_amp = np.abs(slow_wave[slow_peak_idx])

        # Arrow
        ax.annotate(
            '',
            xy=(times[end], slow_amp),
            xytext=(times[start], slow_amp),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
        )

        # Label
        ax.text((times[start] + times[end]) / 2,
                slow_amp,
                'latency',
                color='black',
                ha='center',
                va='bottom')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig, ax
