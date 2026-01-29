import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import plotting, datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_all_subjects_electrodes(data_dir):
    """Load electrodes for all subjects."""
    participants_file = Path(data_dir) / "participants.tsv"
    
    if participants_file.exists():
        participants = pd.read_csv(participants_file, sep='\t')
        subject_ids = participants['participant_id'].str.replace('sub-', '').values
    else:
        subject_dirs = [d for d in Path(data_dir).glob("sub-*") if d.is_dir()]
        subject_ids = [d.name.replace('sub-', '') for d in subject_dirs]
    
    all_electrodes = []
    subject_labels = []
    
    for subject_id in subject_ids:
        electrode_file = Path(data_dir) / f"sub-{subject_id}" / "ieeg" / f"sub-{subject_id}_electrodes.tsv"
        
        if electrode_file.exists():
            df = pd.read_csv(electrode_file, sep='\t')
            
            if all(col in df.columns for col in ['x', 'y', 'z']):
                coords = df[['x', 'y', 'z']].values
            elif all(col in df.columns for col in ['X', 'Y', 'Z']):
                coords = df[['X', 'Y', 'Z']].values
            else:
                coord_cols = [col for col in df.columns if any(x in col.lower() for x in ['mni', 'coord'])]
                if len(coord_cols) >= 3:
                    coords = df[coord_cols[:3]].values
                else:
                    continue
            
            all_electrodes.append(coords)
            subject_labels.extend([subject_id] * len(coords))
            print(f"Loaded {len(coords)} electrodes for subject {subject_id}")
    
    if all_electrodes:
        all_electrodes = np.vstack(all_electrodes)
        return all_electrodes, subject_labels
    else:
        return None, None


def create_comprehensive_plot(all_electrodes, subject_labels, output_file='electrodes_plot.png'):
    """
    Create a comprehensive multi-view plot of electrodes.
    """
    # Prepare colors
    unique_subjects = list(set(subject_labels))
    n_subjects = len(unique_subjects)
    colors_array = plt.cm.tab20(np.linspace(0, 1, n_subjects))
    subject_color_map = {subj: colors_array[i] for i, subj in enumerate(unique_subjects)}
    subject_to_value = {subj: i for i, subj in enumerate(unique_subjects)}
    node_values = np.array([subject_to_value[label] for label in subject_labels])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Ortho view (top left)
    ax1 = plt.subplot(2, 3, 1)
    display1 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=15,
        node_cmap='tab20',
        display_mode='ortho',
        colorbar=False,
        title='Orthogonal View',
        figure=fig,
        axes=ax1
    )
    
    # 2. Left hemisphere (top middle)
    ax2 = plt.subplot(2, 3, 2)
    display2 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=15,
        node_cmap='tab20',
        display_mode='lzr',
        colorbar=False,
        title='Left Hemisphere',
        figure=fig,
        axes=ax2
    )
    
    # 3. Right hemisphere (top right)
    ax3 = plt.subplot(2, 3, 3)
    display3 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=15,
        node_cmap='tab20',
        display_mode='r',
        colorbar=False,
        title='Right Hemisphere',
        figure=fig,
        axes=ax3
    )
    
    # 4. Axial slices (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    display4 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=15,
        node_cmap='tab20',
        display_mode='z',
        colorbar=False,
        title='Axial View',
        figure=fig,
        axes=ax4
    )
    
    # 5. Sagittal slices (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    display5 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=15,
        node_cmap='tab20',
        display_mode='x',
        colorbar=False,
        title='Sagittal View',
        figure=fig,
        axes=ax5
    )
    
    # 6. Coronal slices (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    display6 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=15,
        node_cmap='tab20',
        display_mode='y',
        colorbar=False,
        title='Coronal View',
        figure=fig,
        axes=ax6
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {output_file}")


def plot_single_subject_on_slices(data_dir, subject_id, output_file=None):
    """
    Plot a single subject's electrodes on anatomical slices.
    """
    electrode_file = Path(data_dir) / f"sub-{subject_id}" / "ieeg" / f"sub-{subject_id}_electrodes.tsv"
    
    if not electrode_file.exists():
        print(f"Electrode file not found: {electrode_file}")
        return
    
    df = pd.read_csv(electrode_file, sep='\t')
    
    if all(col in df.columns for col in ['x', 'y', 'z']):
        coords = df[['x', 'y', 'z']].values
    elif all(col in df.columns for col in ['X', 'Y', 'Z']):
        coords = df[['X', 'Y', 'Z']].values
    else:
        print("Could not find coordinate columns")
        return
    
    # Load MNI template
    mni_img = datasets.load_mni152_template()
    
    # Use electrode centroid for cut coordinates
    cut_coords = coords.mean(axis=0).tolist()
    
    # Create plot
    display = plotting.plot_anat(
        mni_img,
        cut_coords=cut_coords,
        display_mode='ortho',
        title=f'Subject {subject_id} iEEG Electrodes (N={len(coords)})',
        dim=-0.5  # Dim the background
    )
    
    # Add electrodes
    display.add_markers(
        coords.tolist(),
        marker_color='red',
        marker_size=120
    )
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_mosaic_view(all_electrodes, subject_labels):
    """
    Create a glass brain view showing all electrodes.
    """
    unique_subjects = list(set(subject_labels))
    subject_to_value = {subj: i for i, subj in enumerate(unique_subjects)}
    node_values = np.array([subject_to_value[label] for label in subject_labels])
    
    # Create a mosaic-style view using glass brain
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Axial view
    display1 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=20,
        node_cmap='tab20',
        display_mode='z',
        colorbar=True,
        title='Axial View',
        axes=axes[0, 0]
    )
    
    # Sagittal view
    display2 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=20,
        node_cmap='tab20',
        display_mode='x',
        colorbar=False,
        title='Sagittal View',
        axes=axes[0, 1]
    )
    
    # Coronal view
    display3 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=20,
        node_cmap='tab20',
        display_mode='y',
        colorbar=False,
        title='Coronal View',
        axes=axes[1, 0]
    )
    
    # 3D-like orthogonal view
    display4 = plotting.plot_markers(
        node_values=node_values,
        node_coords=all_electrodes,
        node_size=20,
        node_cmap='tab20',
        display_mode='ortho',
        colorbar=False,
        title='Orthogonal View',
        axes=axes[1, 1]
    )
    
    plt.tight_layout()
    plt.show()
    
    return display1


if __name__ == "__main__":
    data_dir = "/Users/amritakohli/Downloads/ieeg_ieds_bids"
    
    print("Loading electrode data...")
    all_electrodes, subject_labels = load_all_subjects_electrodes(data_dir)
    
    if all_electrodes is not None:
        print(f"\nTotal subjects: {len(set(subject_labels))}")
        print(f"Total electrodes: {len(all_electrodes)}")
        
        # Create comprehensive plot
        create_comprehensive_plot(all_electrodes, subject_labels)
        
        # Create mosaic view
        plot_mosaic_view(all_electrodes, subject_labels)
        
        # Plot first subject individually (example)
        first_subject = list(set(subject_labels))[0]
        print(f"\nPlotting single subject: {first_subject}")
        plot_single_subject_on_slices(data_dir, first_subject, 
                                      f'subject_{first_subject}_electrodes.png')
    else:
        print("Failed to load electrode data.")