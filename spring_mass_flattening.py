"""
Surface flattening tool for converting 3D models to 2D patterns.
"""

import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

from flattening.algorithms import surface_flattening_spring_mass


# Configuration
STL_FILE = 'files/Partial_Oblong_Cylinder_Shell'
USE_PRECOMPUTED = True
ENABLE_ENERGY_RELEASE_IN_FLATTEN = False
ENABLE_ENERGY_RELEASE_PHASE = True


def main():
    """Run the surface flattening algorithm and display the results."""
    # Load mesh from file
    mesh = trimesh.load(STL_FILE + ".stl")

    area_density = None
    if USE_PRECOMPUTED and os.path.exists(STL_FILE + "_areadensity.npy"):
        area_density = np.load(STL_FILE + "_areadensity.npy")

    vertices_2d_initial = None
    if USE_PRECOMPUTED and os.path.exists(STL_FILE + "_init2d.npy"):
        vertices_2d_initial = np.load(STL_FILE + "_init2d.npy")
    
    # Perform flattening
    flattened_vertices_2d, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = surface_flattening_spring_mass(
        mesh,
        enable_energy_release_in_flatten=ENABLE_ENERGY_RELEASE_IN_FLATTEN,
        enable_energy_release_phase=ENABLE_ENERGY_RELEASE_PHASE,
        area_density=area_density,
        vertices_2d_initial=vertices_2d_initial
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


    if area_errors:
        fig2 = plt.figure(figsize=(18, 5))

        # Plot 1: Area and shape errors (unchanged)
        axL = fig2.add_subplot(131)
        axL.plot(range(len(area_errors)), area_errors, label="area error")
        axL.plot(range(len(shape_errors)), shape_errors, label="shape error")
        axL.legend()
        axL.set_title("Errors over Energy Reduction")

        # Plot 2: Energy on its own
        axM = fig2.add_subplot(132)
        axM.plot(range(len(energies)), energies, label="energy", color='green')
        axM.legend()
        axM.set_title("Energy over Energy Reduction")

        # Plot 3: Forces and penalty displacement together
        axR = fig2.add_subplot(133)
        # axR.plot(range(len(max_forces)), max_forces, label="max force", color='red')
        axR.plot(range(len(max_displacements)), max_displacements, label="max displacement", color='green')
        axR.plot(range(len(max_penalty_displacements)), max_penalty_displacements, label="max penalty displacement", color='blue')
        axR.legend()
        axR.set_title("Displacements over Energy Reduction")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
