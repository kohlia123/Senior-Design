import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
from nilearn import datasets
from pathlib import Path
import matplotlib.pyplot as plt


def load_gii_surface(file):
    """Load GIFTI surface file and convert to PyVista mesh."""
    gii = nib.load(file)
    coords = gii.darrays[0].data
    faces = gii.darrays[1].data
    # Convert faces to PyVista format
    n_faces = faces.shape[0]
    faces_pv = np.hstack([np.full((n_faces, 1), 3), faces]).astype(np.int32).ravel()
    mesh = pv.PolyData(coords, faces_pv)
    return mesh


def load_subject_electrodes(data_dir, subject_id):
    """Load electrode coordinates for a specific subject."""
    # Updated path to include 'ieeg' subfolder
    electrode_file = Path(data_dir) / f"sub-{subject_id}" / "ieeg" / f"sub-{subject_id}_electrodes.tsv"
    
    if electrode_file.exists():
        df = pd.read_csv(electrode_file, sep='\t')
        
        # Print columns for first subject to help debug
        if subject_id == '01':
            print(f"Columns in electrode file: {list(df.columns)}")
            print(f"First row:\n{df.head(1)}")
        
        # Extract x, y, z coordinates (assuming columns are named 'x', 'y', 'z' or similar)
        # Adjust column names based on your actual data
        if all(col in df.columns for col in ['x', 'y', 'z']):
            coords = df[['x', 'y', 'z']].values
        elif all(col in df.columns for col in ['X', 'Y', 'Z']):
            coords = df[['X', 'Y', 'Z']].values
        else:
            # Try to find MNI coordinate columns
            coord_cols = [col for col in df.columns if any(x in col.lower() for x in ['mni', 'coord'])]
            if len(coord_cols) >= 3:
                coords = df[coord_cols[:3]].values
            else:
                print(f"Warning: Could not find coordinate columns for {subject_id}")
                print(f"Available columns: {list(df.columns)}")
                return None, None
        
        electrode_names = df['name'].values if 'name' in df.columns else None
        return coords, electrode_names
    else:
        print(f"Electrode file not found for {subject_id}: {electrode_file}")
        return None, None


def load_all_subjects_electrodes(data_dir):
    """Load electrodes for all subjects."""
    # Read participants file to get all subject IDs
    participants_file = Path(data_dir) / "participants.tsv"
    
    if participants_file.exists():
        participants = pd.read_csv(participants_file, sep='\t')
        subject_ids = participants['participant_id'].str.replace('sub-', '').values
    else:
        # If participants.tsv doesn't exist, scan directories
        subject_dirs = [d for d in Path(data_dir).glob("sub-*") if d.is_dir()]
        subject_ids = [d.name.replace('sub-', '') for d in subject_dirs]
    
    all_electrodes = []
    subject_labels = []
    
    for subject_id in subject_ids:
        coords, names = load_subject_electrodes(data_dir, subject_id)
        if coords is not None and len(coords) > 0:
            all_electrodes.append(coords)
            subject_labels.extend([subject_id] * len(coords))
            print(f"Loaded {len(coords)} electrodes for subject {subject_id}")
    
    if all_electrodes:
        all_electrodes = np.vstack(all_electrodes)
        return all_electrodes, subject_labels, subject_ids
    else:
        print("No electrode data found!")
        return None, None, None


def plot_electrodes_on_brain(all_electrodes, subject_labels=None, 
                             electrode_size=2.0, opacity=0.5, 
                             color_by_subject=True):
    """Plot all electrodes on the fsaverage brain template."""
    
    # Load fsaverage surface (GIFTI files)
    print("Loading fsaverage brain template...")
    fsaverage = datasets.fetch_surf_fsaverage()
    lh_file = fsaverage['pial_left']
    rh_file = fsaverage['pial_right']
    
    # Convert surfaces to PyVista meshes
    lh_mesh = load_gii_surface(lh_file)
    rh_mesh = load_gii_surface(rh_file)
    
    # Create a plotter
    plotter = pv.Plotter(window_size=[1200, 800])
    
    # Add brain hemispheres
    plotter.add_mesh(lh_mesh, color='lightgrey', opacity=opacity, 
                     smooth_shading=True)
    plotter.add_mesh(rh_mesh, color='lightgrey', opacity=opacity, 
                     smooth_shading=True)
    
    # Add electrodes
    if color_by_subject and subject_labels is not None:
        # Create color map for different subjects
        unique_subjects = list(set(subject_labels))
        n_subjects = len(unique_subjects)
        colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))
        
        subject_color_map = {subj: colors[i] for i, subj in enumerate(unique_subjects)}
        
        for i, (x, y, z) in enumerate(all_electrodes):
            color = subject_color_map[subject_labels[i]][:3]  # RGB only
            sphere = pv.Sphere(radius=electrode_size, center=(x, y, z))
            plotter.add_mesh(sphere, color=color, opacity=0.8)
    else:
        # Single color for all electrodes
        for x, y, z in all_electrodes:
            sphere = pv.Sphere(radius=electrode_size, center=(x, y, z))
            plotter.add_mesh(sphere, color='red', opacity=0.8)
    
    # Set camera position and add text
    plotter.camera_position = 'xy'
    plotter.add_text(f"Total electrodes: {len(all_electrodes)}", 
                     position='upper_left', font_size=12)
    
    # Add axes
    plotter.add_axes()
    
    print(f"Displaying {len(all_electrodes)} electrodes from {len(set(subject_labels)) if subject_labels else 'unknown'} subjects")
    plotter.show()


def explore_dataset_structure(data_dir):
    """Explore and print dataset structure to help debug."""
    data_path = Path(data_dir)
    
    print(f"Checking directory: {data_path}")
    print(f"Directory exists: {data_path.exists()}")
    
    if not data_path.exists():
        print("ERROR: Directory does not exist!")
        return
    
    print("\n=== Directory Contents ===")
    for item in sorted(data_path.iterdir()):
        print(f"  {item.name}")
    
    print("\n=== Subject Directories ===")
    subject_dirs = sorted([d for d in data_path.glob("sub-*") if d.is_dir()])
    print(f"Found {len(subject_dirs)} subject directories")
    
    if subject_dirs:
        # Check first subject directory
        first_subj = subject_dirs[0]
        print(f"\n=== Contents of {first_subj.name} ===")
        for item in sorted(first_subj.iterdir()):
            print(f"  {item.name}")
        
        # Look for electrode files
        print("\n=== Looking for electrode files ===")
        electrode_files = list(data_path.rglob("*electrode*.tsv"))
        if electrode_files:
            print(f"Found {len(electrode_files)} electrode file(s):")
            for f in electrode_files:
                print(f"  {f.relative_to(data_path)}")
                # Show first few lines
                print(f"    First few lines:")
                df = pd.read_csv(f, sep='\t', nrows=3)
                print(f"    Columns: {list(df.columns)}")
        else:
            print("No electrode files found with pattern '*electrode*.tsv'")
        
        # Check for coordinate files in different locations
        print("\n=== Checking for any .tsv files with coordinates ===")
        all_tsv = list(data_path.rglob("*.tsv"))
        print(f"Total .tsv files found: {len(all_tsv)}")
        for tsv in all_tsv[:10]:  # Show first 10
            print(f"  {tsv.relative_to(data_path)}")


# Main execution
if __name__ == "__main__":
    # Set your data directory path
    data_dir = "/Users/amritakohli/Downloads/ieeg_ieds_bids"
    
    # First, explore the dataset structure
    print("=== EXPLORING DATASET STRUCTURE ===\n")
    explore_dataset_structure(data_dir)
    
    print("\n\n=== ATTEMPTING TO LOAD ELECTRODE DATA ===\n")
    
    # Load all electrodes
    print("Loading electrode data for all subjects...")
    all_electrodes, subject_labels, subject_ids = load_all_subjects_electrodes(data_dir)
    
    if all_electrodes is not None:
        print(f"\nTotal subjects: {len(set(subject_labels))}")
        print(f"Total electrodes: {len(all_electrodes)}")
        print(f"Coordinate range - X: [{all_electrodes[:, 0].min():.1f}, {all_electrodes[:, 0].max():.1f}]")
        print(f"                   Y: [{all_electrodes[:, 1].min():.1f}, {all_electrodes[:, 1].max():.1f}]")
        print(f"                   Z: [{all_electrodes[:, 2].min():.1f}, {all_electrodes[:, 2].max():.1f}]")
        
        # Plot electrodes
        plot_electrodes_on_brain(all_electrodes, subject_labels, 
                                electrode_size=2.0, opacity=0.5, 
                                color_by_subject=True)
    else:
        print("\nFailed to load electrode data.")
        print("Please check the output above to understand the dataset structure.")