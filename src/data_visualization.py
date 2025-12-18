from src.config import *
from src.utils.preprocessing import get_subj_epochs, get_subj_epochs_mne
from src.utils.plotting import plot_ied


subj = "01"
epochs, ied_labels, ied_onsets, = get_subj_epochs(subj, window_size=250)

total_ied = sum(
    len(epoch_onsets)
    for chan in ied_onsets.values()
    for epoch_onsets in chan
)
print(f"Total IED events: {total_ied}")

ied_epochs = [
    ep for ep in range(len(next(iter(epochs.values()))))
    if any(ied_labels[ch][ep] == 1 for ch in epochs)
]
plot_ied(epochs, ied_labels, ied_onsets, epoch_idx=ied_epochs[0])
