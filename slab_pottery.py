import argparse
import os
import sys
import trimesh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from deduplication import interactive_deduplicate
from segmenting import segment_mesh_face_normals
from flattening.algorithms import surface_flattening_spring_mass
from cutting.initial_cut import find_cutting_path, make_cut
from beveled_edges import extract_mesh_regions_with_thickness_and_bevel_angles, visualize_mesh_thickness, print_thickness_statistics
from export import generate_svg


ENABLE_ENERGY_RELEASE_IN_FLATTEN = True
ENERGY_RELEASE_ITERATIONS = 10  # Number of iterations between applying energy release in the initial flattening

ENABLE_ENERGY_RELEASE_PHASE = True

ENERGY_RELEASE_TIMESTEP = 0.01
ENERGY_RELEASE_PENALTY_COEFFICIENT = 1.0
PERMISSIBLE_ENERGY_VARIATION = 0.0005

VERBOSE = True

def load_stl_file_cli():
    parser = argparse.ArgumentParser(description='Load an STL file into a trimesh object.')
    parser.add_argument('filename', type=str, help='Path to the STL file to load')
    
    args = parser.parse_args()
    stl_filename = args.filename
    filepath = f'files/{stl_filename}'
    
    if not filepath.lower().endswith('.stl'):
        print(f"Error: The file must have a .stl extension. Got: {filepath}")
        sys.exit(1)
    
    if not os.path.isfile(filepath):
        print(f"Error: The file {filepath} does not exist.")
        sys.exit(1)
    
    try:
        mesh = trimesh.load(filepath)
        return mesh, stl_filename
    
    except Exception as e:
        print(f"Error loading the STL file: {e}")
        sys.exit(1)

def main():
    # Load the STL file by passing in a filepath argument
    mesh, _ = load_stl_file_cli()
    regions = segment_mesh_face_normals(mesh, angle_threshold=30)

    # Perform interactive deduplication to allow the user to select faces/regions made up of the same slab
    _, _, dedup_region_selections = interactive_deduplicate(mesh, regions)

    # Extract mesh with thickness information
    region_meshes, region_bevel_angles = extract_mesh_regions_with_thickness_and_bevel_angles(mesh, dedup_region_selections['region_pairs'], regions)

    if VERBOSE:
        for _, region_mesh in region_meshes.items():
            print_thickness_statistics(region_mesh)
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            visualize_mesh_thickness(
                region_mesh, 
                ax=ax, 
                alpha=1.0, 
                show_edges=True,
                cmap='viridis',
                title="Mesh Surface Thickness Visualization"
            )
            plt.tight_layout()
            plt.show()
    
    # Detect join edges and calculate ideal beveled edges for the newly generated shell mesh
    # beveled_edges = detect_beveled_edges(dedup_mesh, dedup_region_faces)
    
    # Perform flattening on all region meshes
    for idx, region_mesh in region_meshes.items():
        cutting_path = find_cutting_path(region_mesh)

        mesh_cut, _ = make_cut(region_mesh, cutting_path)
    
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
            axR.plot(range(len(max_displacements)), max_displacements, label="max displacement", color='green')
            axR.plot(range(len(max_penalty_displacements)), max_penalty_displacements, label="max penalty displacement", color='blue')
            axR.legend()
            axR.set_title("Displacements over ER")

        plt.tight_layout()
        plt.show()

        generate_svg(flattened_vertices_2d, mesh_cut.faces, f'{idx}_cut.svg', region_bevel_angles[idx])


if __name__ == "__main__":
    main()
