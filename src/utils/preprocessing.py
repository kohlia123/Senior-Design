import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import mne
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
from src.feature_extraction import (
    feat_sharpness, 
    ied_duration_ms, 
    get_ptp_amplitude, 
    feat_slow_afterwave,
    feat_background_disruption
)

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
    epochs_np = np.array(epochs)
    feats = []
    
    # Pre-define column names for consistent DataFrame shape
    slow_wave_cols = ["slow_amplitude", "spike_amplitude", "amplitude_ratio", "latency_ms", "duration_ms"]
    bg_cols = ["background_rms", "background_std", "background_line_length", 
               "background_delta_power", "background_alpha_power", "event_rms_ratio_bg"]

    for i, epoch in enumerate(epochs_np):
        # Anchor all searches at the midpoint (500ms mark)
        mid_idx = len(epoch) // 2
        
        # Base metadata and Morphology
        row = {
            'subj': subj,
            'epoch_id': i,
            'ptp_amp': np.ptp(epoch),
            'sharpness': feat_sharpness(epoch, sr),
            'ied_duration': ied_duration_ms(epoch, sr, onset_idx=mid_idx)
        }
        
        # Extract Slow After-wave (returns bool, dict)
        _, slow_feats = feat_slow_afterwave(epoch, sr, spike_index=mid_idx)
        if slow_feats:
            row.update(slow_feats)
        else:
            row.update({k: 0.0 for k in slow_wave_cols})

        # Extract Background Context (returns dict)
        bg_feats = feat_background_disruption(epoch, sr, spike_index=mid_idx)
        if bg_feats:
            row.update(bg_feats)
        else:
            row.update({k: 0.0 for k in bg_cols})
            
        feats.append(row)

    # Convert the list of dictionaries into a single DataFrame
    final_feats = pd.DataFrame(feats)
    cols_to_fix = [c for c in final_feats.columns if c not in ['subj', 'epoch_id']]
    final_feats[cols_to_fix] = final_feats[cols_to_fix].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    return final_feats

def get_subj_data(subj):
    window_size_ms = 1200  
    stride_ms = 250       
    
    # Load raw data - verbose=False prevents terminal flooding
    raw = read_raw_bids(
        BIDSPath(subject=subj, task='sleep', root=BIDS_ROOT, datatype='ieeg'), 
        verbose=False
    )
    chans_onsets = onset_per_chan(subj)
    sfreq = raw.info['sfreq']

    y = []
    x = pd.DataFrame()
    
    window_samples = int((window_size_ms / 1000.0) * sfreq)
    stride_samples = int((stride_ms / 1000.0) * sfreq)

    for chan in chans_onsets.keys():
        epochs = []
        
        # 1. Extract and normalize
        chan_raw = raw.copy().pick([chan]).get_data().flatten()
        chan_norm = (chan_raw - chan_raw.mean()) / chan_raw.std()
        
        # 2. Slice the normalized data
        # Move by 250ms stride, but grab 1000ms windows
        for i in range(0, len(chan_norm) - window_samples, stride_samples):
            epochs.append(chan_norm[i : i + window_samples])

        # 3. Labeling Logic (Overlapping)
        curr_y = []
        for i in range(len(epochs)):
            start_sec = (i * stride_samples) / sfreq
            end_sec = start_sec + (window_size_ms / 1000.0)
            
            # Check if any annotation falls within this 1s window
            is_spike = any(start_sec <= onset < end_sec for onset in chans_onsets[chan])
            curr_y.append(1 if is_spike else 0)

        # 4. Extract Features
        curr_feat = extract_epochs_features(epochs, subj, sfreq)
        
        # 5. Metadata
        curr_feat['chan_name'] = chan
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
