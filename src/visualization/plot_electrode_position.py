import numpy as np
import nibabel as nib
import pyvista as pv
from nilearn import datasets


def load_gii_surface(file):
    gii = nib.load(file)
    coords = gii.darrays[0].data
    faces = gii.darrays[1].data
    # Convert faces to PyVista format
    n_faces = faces.shape[0]
    faces_pv = np.hstack([np.full((n_faces,1),3), faces]).astype(np.int32).ravel()
    mesh = pv.PolyData(coords, faces_pv)
    return mesh


# Example electrode coordinates
electrodes = np.array([
    [30, -20, 50],
    [32, -22, 52],
    [28, -18, 48],
])

# Load fsaverage surface (GIFTI files)
fsaverage = datasets.fetch_surf_fsaverage()
lh_file = fsaverage['pial_left']  # .gii
rh_file = fsaverage['pial_right']  # .gii

# Convert surfaces to PyVista meshes
lh_mesh = load_gii_surface(lh_file)
rh_mesh = load_gii_surface(rh_file)

# Create a plotter
plotter = pv.Plotter()
plotter.add_mesh(lh_mesh, color='lightgrey', opacity=0.5)
plotter.add_mesh(rh_mesh, color='lightgrey', opacity=0.5)

# Add electrodes
for x, y, z in electrodes:
    plotter.add_mesh(pv.Sphere(radius=1.5, center=(x, y, z)), color='red')

plotter.show()
