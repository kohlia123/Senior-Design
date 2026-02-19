import numpy as np
import pandas as pd
import antropy as ant
import scipy.stats as sp_stats
import mne
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
from src.feature_extraction import feat_sharpness, ied_duration_ms, get_ptp_amplitude

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = PROJECT_ROOT / "ieeg_ieds_bids"


def onset_per_chan(subj):
    """Create a dictionary with the onsets of each channel"""
    events_interp_path = (
            BIDS_ROOT
            / "derivatives"
            / f"sub-{subj}_task-sleep_events_interpretation.tsv"
    )
    tags_df = pd.read_csv(events_interp_path, sep="\t")
    result_dict = {}

    for _, row in tags_df.iterrows():
        onset = row['time_in_sec']
        chans = row['chans']
        for string in chans.split(' '):
            if string not in result_dict:
                result_dict[string] = []
            result_dict[string].append(onset)

    result_dict.pop('', None)
    return result_dict

def extract_epochs_features(epochs, subj, sr):
    """
    Extract ONLY the 3 custom features: Sharpness, Duration, and PTP Amplitude.
    """
    epochs_np = np.array(epochs)
    
    # 1. Initialize lists for single-epoch calculations
    sharpness_vals = []
    duration_vals = []
    
    for epoch in epochs_np:
        # Sharpness calculation
        sharp = feat_sharpness(epoch, sr)
        sharpness_vals.append(sharp)
        
        # Duration calculation (midpoint used as the search anchor)
        mid_idx = len(epoch) // 2
        dur = ied_duration_ms(epoch, sr, onset_idx=mid_idx)
        duration_vals.append(dur)

    # 2. Build the final feature dataframe
    feat = pd.DataFrame({
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        # Feature 1: Amplitude
        'ptp_amp': get_ptp_amplitude(epochs_np), 
        # Feature 2: Sharpness
        'sharpness': sharpness_vals,
        # Feature 3: Duration
        'ied_duration': duration_vals
    })

    return feat

def get_subj_data(subj):
    """Get features and labels of a single subject (all channels)"""
    window_size_ms = 250  
    raw = read_raw_bids(BIDSPath(subject=subj, task='sleep', root=BIDS_ROOT, datatype='ieeg'), verbose=False)
    chans_onsets = onset_per_chan(subj)
    sfreq = raw.info['sfreq']

    y = []
    x = pd.DataFrame()
    
    for chan in chans_onsets.keys():
        epochs = []
        # Get raw data for the specific channel
        chan_raw = raw.copy().pick([chan]).get_data().flatten()
        
        # Standardize the channel signal (Z-score)
        chan_norm = (chan_raw - chan_raw.mean()) / chan_raw.std()
        
        # Calculate window size in samples based on sfreq
        window_samples = int((window_size_ms / 1000.0) * sfreq)
        
        # Split into 250ms chunks
        for i in range(0, len(chan_norm) - window_samples, window_samples):
            epochs.append(chan_norm[i: i + window_samples])

        # Labeling: 1 if an IED onset falls within this 250ms window
        curr_y = [0] * len(epochs)
        window_sec = window_size_ms / 1000.0
        for onset in chans_onsets[chan]:
            idx = int(onset // window_sec)
            if idx < len(curr_y):
                curr_y[idx] = 1

        # Extract only our custom features
        curr_feat = extract_epochs_features(epochs, subj, sfreq)
        
        # Add metadata for tracking/debugging
        curr_feat['chan_name'] = chan
        curr_feat['epoch'] = epochs # Keeping this for build_dataset to drop later
        
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
