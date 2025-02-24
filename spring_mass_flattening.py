"""
Surface flattening tool for converting 3D models to 2D patterns.
"""

import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

from flattening.algorithms import surface_flattening_spring_mass


# Configuration
STL_FILE = 'files/Partial_Oblong_Cylinder_Shell.stl'
ENABLE_ENERGY_RELEASE_IN_FLATTEN = False
ENABLE_ENERGY_RELEASE_PHASE = True


def main():
    """Run the surface flattening algorithm and display the results."""
    # Load mesh from file
    mesh = trimesh.load(STL_FILE)
    
    # Perform flattening
    flattened_vertices_2d = surface_flattening_spring_mass(
        mesh,
        enable_energy_release_in_flatten=ENABLE_ENERGY_RELEASE_IN_FLATTEN,
        enable_energy_release_phase=ENABLE_ENERGY_RELEASE_PHASE
    )
    
    # Create figure with two subplots - one for 3D, one for 2D
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                      triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=0.7)
    ax3d.set_title("Original 3D Surface")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    
    # 2D plot
    ax2d = fig.add_subplot(122)
    ax2d.set_aspect("equal")  # Ensure aspect ratio is 1:1
    
    # Create PolyCollection for faces in 2D
    face_verts_2d = flattened_vertices_2d[mesh.faces]
    poly_collection = PolyCollection(
        face_verts_2d, facecolors="skyblue", edgecolors="black", linewidths=0.5
    )
    ax2d.add_collection(poly_collection)
    
    # Set plot limits to encompass the flattened mesh
    min_coords = min(flattened_vertices_2d[:, 0]), min(flattened_vertices_2d[:, 1])
    max_coords = max(flattened_vertices_2d[:, 0]), max(flattened_vertices_2d[:, 1])
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    padding_x = range_x * 0.1  # 10% padding
    padding_y = range_y * 0.1
    
    ax2d.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
    ax2d.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)
    ax2d.set_title("Flattened Surface")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
