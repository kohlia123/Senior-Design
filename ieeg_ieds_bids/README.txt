# Interictal epileptiform discharge annotations in sleep iEEG Data

## Dataset Overview
This dataset comprises multichannel intracranial EEG (iEEG) recordings from 25 epilepsy patients during overnight sleep, collected at two medical centers. 
The recordings include 852 annotated interictal epileptiform discharges, primarily from the medial temporal lobe, identified by expert neurologists. 
The data is formatted according to the BIDS (Brain Imaging Data Structure) standard for iEEG recordings.

## Dataset Structure
- participants.tsv: Contains demographic and clinical information for each participant, including:
  - participant_id: Unique identifier for each participant.
  - age: Age at the time of the study (in years).
  - sex: Biological sex (M/F).
  - SOZ: Seizure onset zone.
  - TimeFromSleepOnset: Time from sleep onset (in minutes).
  - SleepScoring: Sleep stages scored according to AASM criteria.
  
- sub-\<subject_id\>/: Contains the iEEG recordings and metadata for each participant.
  - sub-\<subject_id\>_task-sleep_ieeg.edf: The raw iEEG data in EDF format.
  - sub-\<subject_id\>_task-sleep_events.tsv: Event annotations, such as expert-determined IED (interictal epileptiform discharges) timings.
  - sub-\<subject_id\>_electrodes.tsv: Electrode names and MNI coordinates (for select subjects).
  - sub-\<subject_id\>_coordsystem.json: Describes the coordinate system used for electrode localization.

- derivatives/: Contains processed files, such as:
  - sub-\<subject_id\>_task-sleep_events_interpretation.tsv: Interpretation of events for each participant.
  - channels.tsv: Information on channel names


## License and Data Use
The dataset is shared under the CC-BY-NC license. Users are free to use the data for non-commercial purposes with appropriate attribution.

## Citation
If you use this dataset in your research, please cite the following publication:
Falach R, Geva-Sagiv M, Eliashiv D, Goldstein L, Budin O, Gurevitch G, Morris G, Strauss I, Globerson A, Fahoum F, Fried I, Nir Y. Annotated interictal discharges in intracranial EEG sleep data and related machine learning detection scheme. Sci Data. 2024 Dec 18;11(1):1354. doi: 10.1038/s41597-024-04187-y.