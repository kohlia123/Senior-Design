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
