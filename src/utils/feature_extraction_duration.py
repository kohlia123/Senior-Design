import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from src.utils.preprocessing import BIDS_ROOT, get_subj_epochs
from src.utils.time_duration import ied_duration_ms

def get_subj_duration_feature(subj: str, window_size: int = 250):
    epochs_dict, labels_dict, ied_onsets_dict = get_subj_epochs(subj, window_size=window_size)

    raw = read_raw_bids(
        BIDSPath(subject=subj, task="sleep", root=BIDS_ROOT, datatype="ieeg"),
        verbose=False
    )
    sfreq = float(raw.info["sfreq"])

    rows = []
    y = []

    for chan, epochs in epochs_dict.items():
        for ep_i in range(epochs.shape[0]):
            label = int(labels_dict[chan][ep_i])
            y.append(label)

            if label == 1 and len(ied_onsets_dict[chan][ep_i]) > 0:
                onset_idx = int(ied_onsets_dict[chan][ep_i][0])
                dur = ied_duration_ms(epochs[ep_i], sfreq, onset_idx)
            else:
                dur = 0.0

            rows.append({
                "subj": subj,
                "chan_name": chan,
                "epoch_id": ep_i,
                "ied_duration_ms": float(dur),
            })

    X = pd.DataFrame(rows)
    return X, np.array(y, dtype=int)