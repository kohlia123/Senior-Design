import random

from src.config import *   # gives you N_SUB, etc.
from src.utils.preprocessing import get_subj_epochs
from src.utils.plotting import plot_ied



WINDOW_SIZE = 250
N_PLOTS_PER_SUBJ = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def count_total_ieds(ied_onsets):
    return sum(
        len(epoch_onsets)
        for chan in ied_onsets.values()
        for epoch_onsets in chan
    )

def get_ied_epoch_indices(epochs, ied_labels):
    n_epochs = len(next(iter(epochs.values())))
    return [
        ep for ep in range(n_epochs)
        if any(ied_labels[ch][ep] == 1 for ch in epochs)
    ]

subjects = [f"{i:02d}" for i in range(1, N_SUB + 1)]

for subj in subjects:
    print(f"\n===== sub-{subj} =====")

    epochs, ied_labels, ied_onsets = get_subj_epochs(subj, window_size=WINDOW_SIZE)

    total_ied = count_total_ieds(ied_onsets)
    print(f"Total IED events: {total_ied}")

    ied_epochs = get_ied_epoch_indices(epochs, ied_labels)
    if not ied_epochs:
        print("No IED-labeled epochs found — skipping plots.")
        continue

    k = min(N_PLOTS_PER_SUBJ, len(ied_epochs))
    chosen = random.sample(ied_epochs, k)

    for j, ep_idx in enumerate(chosen, start=1):
        print(f"Plot {j}/{k}: epoch_idx={ep_idx}")
        plot_ied(epochs, ied_labels, ied_onsets, epoch_idx=ep_idx)

