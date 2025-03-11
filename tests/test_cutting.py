import os
import sys
from matplotlib.collections import PolyCollection

# Add the parent directory to the Python path so we can import the flattening module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cutting.initial_cut import find_cutting_path, make_cut

from flattening.algorithms import surface_flattening_spring_mass

ENABLE_ENERGY_RELEASE_IN_FLATTEN = True
ENERGY_RELEASE_ITERATIONS = 10  # Number of iterations between applying energy release in the initial flattening

ENABLE_ENERGY_RELEASE_PHASE = True

ENERGY_RELEASE_TIMESTEP = 0.01
ENERGY_RELEASE_PENALTY_COEFFICIENT = 1.0
PERMISSIBLE_ENERGY_VARIATION = 0.0005

import trimesh
import numpy as np

if __name__ == "__main__":
    # Run tests directly
    mesh = trimesh.load_mesh("files/Mug_Shell.stl")

    # print(mesh.faces)
    cutting_path = find_cutting_path(mesh)
    print(cutting_path)

    # # Use matplotlib to plot the mesh and cutting path
    import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the mesh
    # ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
    #                 triangles=mesh.faces, color='blue', alpha=0.7)

    # for seg in cutting_path:
    #     ax.plot(mesh.vertices[seg, 0], mesh.vertices[seg, 1], mesh.vertices[seg, 2], color='red')

    # # # Plot the cutting path
    # # cutting_path_vertices = mesh.vertices[cutting_path]
    # # ax.plot(cutting_path_vertices[:, 0], cutting_path_vertices[:, 1], cutting_path_vertices[:, 2], 
    # #         color='red', linewidth=2)
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Mesh with Cutting Path')
    # plt.tight_layout()
    # plt.show()

    mesh_cut, final_cuts = make_cut(mesh, cutting_path)
    
    # Visualize the cut mesh, also outline the first two triangles in red

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(mesh_cut.vertices[:, 0], mesh_cut.vertices[:, 1], mesh_cut.vertices[:, 2],
                    triangles=mesh_cut.faces, color='blue', alpha=0.7)
    
    # Plot the final cuts
    for cut in final_cuts:
        ax.plot(mesh_cut.vertices[cut, 0], mesh_cut.vertices[cut, 1], mesh_cut.vertices[cut, 2], color='red')

    
    plt.show()


    # Flatten the cut mesh
    flattened_vertices_2d, flattened_vertices_2d_initial, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = surface_flattening_spring_mass(
        mesh_cut,
        enable_energy_release_in_flatten=ENABLE_ENERGY_RELEASE_IN_FLATTEN,
        enable_energy_release_phase=ENABLE_ENERGY_RELEASE_PHASE,
        energy_release_iterations=ENERGY_RELEASE_ITERATIONS,
        dt=ENERGY_RELEASE_TIMESTEP,
        permissible_energy_variation=PERMISSIBLE_ENERGY_VARIATION,
        penalty_coefficient=ENERGY_RELEASE_PENALTY_COEFFICIENT,
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
    face_verts_2d_initial = flattened_vertices_2d_initial[mesh_cut.faces]
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
    face_verts_2d = flattened_vertices_2d[mesh_cut.faces]
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

    # Use matplotlib to plot the flattened mesh, which is 

    # # Use matplotlib to plot the cut mesh

    # fig = plt.figure(figsize=(10, 10))

    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the mesh
    # ax.plot_trisurf(mesh_cut.vertices[:, 0], mesh_cut.vertices[:, 1], mesh_cut.vertices[:, 2],
    #                 triangles=mesh_cut.faces, color='blue', alpha=0.7)
    
    # plt.show()
