import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import mne
from mne_bids import BIDSPath, read_raw_bids
from src.feature_extraction import get_ptp_amplitude
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = PROJECT_ROOT / "ieeg_ieds_bids"


def onset_per_chan(subj):
    """
    Create a dictionary with the onsets of each channel
    """
    # load the list of channels for each annotation
    events_interp_path = (
            BIDS_ROOT
            / "derivatives"
            / f"sub-{subj}_task-sleep_events_interpretation.tsv"
    )
    tags_df = pd.read_csv(events_interp_path, sep="\t")
    result_dict = {}

    # Iterate over the DataFrame rows
    for _, row in tags_df.iterrows():
        onset = row['time_in_sec']
        chans = row['chans']
        for string in chans.split(' '):
            if string not in result_dict:
                result_dict[string] = []
            result_dict[string].append(onset)

    # in case of extra space
    result_dict.pop('', None)
    return result_dict


def extract_epochs_features(epochs, subj, sr):
    """
    Extract features from the epochs of a single channel
    """
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'ptp_amp': get_ptp_amplitude(epochs),
    }

    # Convert to dataframe
    feat = pd.DataFrame(feat)

    return feat


def get_subj_data(subj):
    """
    Get features and labels of a single subject (all channels)
    """
    window_size = 250  # ms
    raw = read_raw_bids(BIDSPath(subject=subj, task='sleep', root=BIDS_ROOT, datatype='ieeg'))
    chans_onsets = onset_per_chan(subj)

    y = []
    x = pd.DataFrame()
    # iterate over all channels that have annotations
    for chan in chans_onsets.keys():
        epochs = []
        chan_raw = raw.copy().pick([chan]).get_data().flatten()
        # normalize chan
        chan_norm = (chan_raw - chan_raw.mean()) / chan_raw.std()
        # run on all 250ms epochs excluding the last 1s
        for i in range(0, len(chan_norm) - 4 * window_size, window_size):
            epochs.append(chan_norm[i: i + window_size])

        # mark the spikes in the right index
        curr_y = [0] * len(epochs)
        for onset in chans_onsets[chan]:
            curr_y[int(onset // 0.25)] = 1

        # add epoch-level features
        curr_feat = extract_epochs_features(epochs, subj, raw.info['sfreq'])
        # add channel-level features
        chan_feat = {
            'chan_name': chan,
            'chan_ptp': np.ptp(chan_norm),
        }

        for feat in chan_feat.keys():
            curr_feat[feat] = chan_feat[feat]

        # save the epochs as column for debugging/visualization
        curr_feat['epoch'] = epochs
        x = pd.concat([x, curr_feat], axis=0)
        y.extend(curr_y)

    return x, y


def get_subj_epochs(subj, window_size=250):
    """
    Load raw iEEG data for a subject, split channels into epochs of `window_size` samples,
    and return the epochs and labels for IEDs.

    Parameters
    ----------
    subj : str
        Subject ID, e.g. '01'
    window_size : int
        Epoch size in samples

    Returns
    -------
    epochs_dict : dict
        Dictionary mapping each channel name to a list of numpy arrays (epochs)
    labels_dict : dict
        Dictionary mapping each channel name to a list of labels (0 = no IED, 1 = IED)
    """
    # Load raw data
    raw = read_raw_bids(
        BIDSPath(subject=subj, task='sleep', root=BIDS_ROOT, datatype='ieeg'),
        verbose=False)

    # Use onset_per_chan to get dictionary of channel → onset times
    chans_onsets = onset_per_chan(subj)

    epochs_dict = {}
    labels_dict = {}
    ied_onsets_dict = {}

    sfreq = raw.info['sfreq']

    for chan, onsets in chans_onsets.items():
        # Extract channel data
        chan_idx = raw.ch_names.index(chan)
        chan_data = raw.get_data(picks=chan_idx).flatten()

        # Normalize channel
        chan_norm = (chan_data - chan_data.mean()) / chan_data.std()

        # Convert IED onsets to samples
        onsets_samples = (np.array(onsets) * sfreq).astype(int)

        # Split into epochs
        epochs = []
        ied_epochs = []
        for start in range(0, len(chan_norm) - window_size + 1, window_size):
            stop = start + window_size
            epoch = chan_norm[start:stop]

            # IEDs inside this epoch (relative indices)
            epoch_onsets = onsets_samples[
                               (onsets_samples >= start) & (onsets_samples < stop)
                               ] - start

            epochs.append(epoch)
            ied_epochs.append(epoch_onsets.tolist())

        # Convert to arrays
        epochs_arr = np.stack(epochs)  # (n_epochs, window_size)
        labels_arr = np.array([len(e) > 0 for e in ied_epochs], dtype=int)

        # Store per-channel
        epochs_dict[chan] = epochs_arr
        labels_dict[chan] = labels_arr
        ied_onsets_dict[chan] = ied_epochs

    return epochs_dict, labels_dict, ied_onsets_dict


def get_subj_epochs_mne(subj, window_size=250):
    raw = read_raw_bids(
        BIDSPath(subject=subj, task='sleep', root=BIDS_ROOT, datatype='ieeg'),
        verbose=False)

    chans_onsets = onset_per_chan(subj)
    sfreq = raw.info['sfreq']
    tmin = - (window_size / 2) / sfreq
    tmax = + (window_size / 2) / sfreq

    # collect all IED times (unique)
    ied_times = sorted(set(t for times in chans_onsets.values() for t in times))

    events = np.array([
        [int(t * sfreq), 0, 1] for t in ied_times
    ])

    epochs = mne.Epochs(
        raw,
        events,
        event_id={'IED': 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    return raw, epochs
