"""
Surface flattening tool for converting 3D models to 2D patterns.
"""

import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from matplotlib import cm
import networkx as nx

from flattening.algorithms import (
  initial_flattening,
  calculate_rho,
  calculate_rest_lengths,
  precompute_all_opposite_edges,
  gen_elastic_deformation_energy_distribution,
  update_mesh_with_path
)

from flattening.physics import calculate_node_energies 

# Configuration
STL_FILE = 'Partial_Open_Bulb_Coarse'
USE_PRECOMPUTED = True

ENABLE_ENERGY_RELEASE_IN_FLATTEN = True
ENERGY_RELEASE_ITERATIONS = 10  # Number of iterations between applying energy release in the initial flattening

ENABLE_ENERGY_RELEASE_PHASE = True

ENERGY_RELEASE_TIMESTEP = 0.01
ENERGY_RELEASE_PENALTY_COEFFICIENT = 1.0
PERMISSIBLE_ENERGY_VARIATION = 0.0005
ENERGY_CHANGE_MIN = 0.00001


def find_cut_lines(energy_sample_graph, energies, cut_line_num):
  curr_node = np.argmax(energies) # start at node with highest energy
  energy_decreasing = True
  path = list()
  path.append(curr_node)
  while (energy_decreasing):
    neighbors = np.array(list(energy_sample_graph.neighbors(curr_node)))
    if (len(neighbors) > 0) :
      delta_energies = energies[curr_node] - energies[neighbors]
      min_energy_change = np.argmin(delta_energies)
      if (delta_energies[min_energy_change] >= ENERGY_CHANGE_MIN) :
        curr_node = neighbors[min_energy_change]
        path.append(curr_node)
      else:
        energy_decreasing = False
    else :
      energy_decreasing = False
  return path


def main():
  # Load mesh from file
  mesh = trimesh.load("files/" + STL_FILE + ".stl")

  area_density = None
  if USE_PRECOMPUTED and os.path.exists("files/" + STL_FILE + "_areadensity.npy"):
      area_density = np.load("files/" + STL_FILE + "_areadensity.npy")
  else: 
    area_density = calculate_rho(mesh)
    np.save(Path(__file__).parent / "files" / (STL_FILE + "_areadensity.npy"), area_density)
  
  # Precompute values that don't change during the process
  rest_lengths = calculate_rest_lengths(mesh.vertices, mesh.edges_unique)
  all_opposite_edges = precompute_all_opposite_edges(mesh.vertices, mesh.faces)

  flattened_vertices_2d_initial = None
  if USE_PRECOMPUTED and os.path.exists("files/" + STL_FILE + "_init2d.npy"):
    print("Skipping initial flattening, using provided vertex positions.")
    flattened_vertices_2d_initial = np.load("files/" + STL_FILE + "_init2d.npy")
  else:
    # 2. Initial Flattening (Triangle Flattening - Constrained)
    flattened_vertices_2d_initial = initial_flattening(
      mesh=mesh,
      spring_constant=0.5,
      area_density=area_density,
      dt=0.001,
      permissible_area_error = 0.01,
      permissible_shape_error = 0.01,
      permissible_energy_variation= 0.0005,
      penalty_coefficient= 1.0,
      enable_energy_release=True,
      energy_release_iterations=1,
      rest_lengths=rest_lengths,
      all_opposite_edges=all_opposite_edges,
    )
    np.save(Path(__file__).parent / "files" / (STL_FILE + "_init2d.npy"), flattened_vertices_2d_initial)

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


  plt.tight_layout()
  plt.show()

  # now caculate energy
  node_energies = calculate_node_energies(flattened_vertices_2d_initial, mesh.edges_unique, rest_lengths,spring_constant=0.5)
  print(node_energies[0:3])
  # generate energy distribution map
  interpolated_energies, energy_sample_graph = gen_elastic_deformation_energy_distribution(mesh.faces, node_energies, mesh)
  flattened_mesh_centerpoints = np.mean(flattened_vertices_2d_initial[mesh.faces], axis=1)
  fig = plt.figure(figsize=(18, 10))
  # join energy sets
  all_energies = np.concatenate([node_energies, interpolated_energies])
  # Normalize energy values
  energy_norm = (all_energies - np.min(all_energies)) / (np.max(all_energies) - np.min(all_energies))
  unflattened_centers = mesh.triangles_center
  all_sampled_positions = np.concatenate([flattened_vertices_2d_initial, flattened_mesh_centerpoints])
  all_sampled_positions3d = np.concatenate([mesh.vertices, unflattened_centers])

  # Convert to colors using a colormap (e.g., 'jet' or 'viridis')
  colormap = cm.jet(energy_norm)  # RGBA colors for each vertex
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_trisurf(flattened_vertices_2d_initial[:, 0], flattened_vertices_2d_initial[:, 1], all_energies, triangles=mesh.faces, cmap=plt.cm.jet, linewidth=0.2, antialiased=True, alpha=0.8, shade=True, facecolors=plt.cm.jet(energy_norm))
  cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
  cbar.set_label('Energy Value')
  plt.show()

  path = find_cut_lines(energy_sample_graph, all_energies, 1)
  updated_mesh_to_cut = update_mesh_with_path(mesh, path, energy_sample_graph, all_sampled_positions3d)
 # 3D plot (top-left)
  fig = plt.figure(figsize=(18, 10))
  ax3d = fig.add_subplot(231, projection='3d')
  ax3d.plot_trisurf(updated_mesh_to_cut.vertices[:, 0], updated_mesh_to_cut.vertices[:, 1], updated_mesh_to_cut.vertices[:, 2],
                  triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=0.7)
  
  path_vertices = updated_mesh_to_cut.vertices[path]
  print(path_vertices)
  ax3d.plot(
    path_vertices[:, 0], path_vertices[:, 1], path_vertices[:, 2], 
    color='red', linewidth=2, marker='o', markersize=5, label="Cutting Path"
  )
  ax3d.set_title("new 3D Surface")
  ax3d.set_xlabel("X")
  ax3d.set_ylabel("Y")
  ax3d.set_zlabel("Z")
  plt.show()
if __name__ == "__main__":
  main()
