import mne

# plt.switch_backend('TkAgg') # needed in Pycharm for interactive view 

# Load data
raw = mne.io.read_raw_edf("ieeg_ieds_bids/sub-openieegDetroit002/ieeg/sub-openieegDetroit002_task-sleep_ieeg.edf", preload=True)

# Plot signals interactively
raw.plot(
 duration=10,  # time resolution (seconds)
 n_channels=10,  # number of channels shown
 scalings=dict(eeg=100e-5), # amplitude resolution (microV)
 block=True  # needed in Pycharm for interactive view  
)