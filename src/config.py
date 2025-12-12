"""
Configuration file.

This file defines constants and parameters used in the preprocessing and analysis of iEEG data.
Update the file paths as needed.

"""

import numpy as np
from pathlib import Path
from src.utils.dataset_utils import setup_dataset


FILE_ID = "51412736"   # Figshare file ID for ieeg_ieds_bids_final.zip

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # assumes this config.py is inside src/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Download data
setup_dataset(FILE_ID, DATA_DIR)
