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
STL_FILE = 'Partial_Oblong_Cylinder_Shell_Coarse'
USE_PRECOMPUTED = False

ENABLE_ENERGY_RELEASE_IN_FLATTEN = True
ENERGY_RELEASE_ITERATIONS = 10  # Number of iterations between applying energy release in the initial flattening

ENABLE_ENERGY_RELEASE_PHASE = True

ENERGY_RELEASE_TIMESTEP = 0.01
ENERGY_RELEASE_PENALTY_COEFFICIENT = 1.0
PERMISSIBLE_ENERGY_VARIATION = 0.0005


def main():
    """Run the surface flattening algorithm and display the results."""
    # Load mesh from file
    mesh = trimesh.load("files/" + STL_FILE + ".stl")

    area_density = None
    if USE_PRECOMPUTED and os.path.exists("files/" + STL_FILE + "_areadensity.npy"):
        area_density = np.load("files/" + STL_FILE + "_areadensity.npy")

    vertices_2d_initial = None
    if USE_PRECOMPUTED and os.path.exists("files/" + STL_FILE + "_init2d.npy"):
        vertices_2d_initial = np.load("files/" + STL_FILE + "_init2d.npy")
    
    # Perform flattening
    flattened_vertices_2d, flattened_vertices_2d_initial, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = surface_flattening_spring_mass(
        mesh,
        enable_energy_release_in_flatten=ENABLE_ENERGY_RELEASE_IN_FLATTEN,
        enable_energy_release_phase=ENABLE_ENERGY_RELEASE_PHASE,
        energy_release_iterations=ENERGY_RELEASE_ITERATIONS,
        area_density=area_density,
        vertices_2d_initial=vertices_2d_initial,
        dt=ENERGY_RELEASE_TIMESTEP,
        permissible_energy_variation=PERMISSIBLE_ENERGY_VARIATION,
        penalty_coefficient=ENERGY_RELEASE_PENALTY_COEFFICIENT,
        object_name=STL_FILE,
    )
    
    # Create a single figure with 2x3 grid layout
    fig = plt.figure(figsize=(18, 10))

    # 3D plot (top-left)
    ax3d = fig.add_subplot(231, projection='3d')
    ax3d.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=0.7)
    ax3d.set_title("Original 3D Surface")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # 2D plot for initial (top-middle)
    ax2d_initial = fig.add_subplot(232)
    ax2d_initial.set_aspect("equal")  
    face_verts_2d_initial = flattened_vertices_2d_initial[mesh.faces]
    poly_collection_initial = PolyCollection(
        face_verts_2d_initial, facecolors="skyblue", edgecolors="black", linewidths=0.5
    )
    ax2d_initial.add_collection(poly_collection_initial)

    # Set plot limits for initial 2D plot
    min_coords = min(flattened_vertices_2d_initial[:, 0]), min(flattened_vertices_2d_initial[:, 1])
    max_coords = max(flattened_vertices_2d_initial[:, 0]), max(flattened_vertices_2d_initial[:, 1])
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    padding_x = range_x * 0.1
    padding_y = range_y * 0.1
    ax2d_initial.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
    ax2d_initial.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)
    ax2d_initial.set_title(f"Initial Surface (ER steps = {ENERGY_RELEASE_ITERATIONS})")

    # 2D plot for final (top-right)
    ax2d_final = fig.add_subplot(233)
    ax2d_final.set_aspect("equal")
    face_verts_2d = flattened_vertices_2d[mesh.faces]
    poly_collection_final = PolyCollection(
        face_verts_2d, facecolors="skyblue", edgecolors="black", linewidths=0.5
    )
    ax2d_final.add_collection(poly_collection_final)

    # Set plot limits for final 2D plot
    min_coords = min(flattened_vertices_2d[:, 0]), min(flattened_vertices_2d[:, 1])
    max_coords = max(flattened_vertices_2d[:, 0]), max(flattened_vertices_2d[:, 1])
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    padding_x = range_x * 0.1
    padding_y = range_y * 0.1
    ax2d_final.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
    ax2d_final.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)
    ax2d_final.set_title(f"Final Surface (dt = {ENERGY_RELEASE_TIMESTEP})")

    # Bottom row plots only if area_errors exist
    if area_errors:
        # Plot 1: Area and shape errors (bottom-left)
        axL = fig.add_subplot(234)
        axL.plot(range(len(area_errors)), area_errors, label="area error")
        axL.plot(range(len(shape_errors)), shape_errors, label="shape error")
        axL.legend()
        axL.set_title(f"Errors over ER (last: AE = {area_errors[-1]:.4f}, SE = {shape_errors[-1]:.4f})")

        # Plot 2: Energy on its own (bottom-middle)
        axM = fig.add_subplot(235)
        axM.plot(range(len(energies)), energies, label="energy", color='green')
        axM.legend()
        axM.set_title(f"Energy over ER (last: {energies[-1]:.4f})")

        # Plot 3: Forces and penalty displacement (bottom-right)
        axR = fig.add_subplot(236)
        # axR.plot(range(len(max_forces)), max_forces, label="max force", color='red')
        axR.plot(range(len(max_displacements)), max_displacements, label="max displacement", color='green')
        axR.plot(range(len(max_penalty_displacements)), max_penalty_displacements, label="max penalty displacement", color='blue')
        axR.legend()
        axR.set_title("Displacements over ER")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
