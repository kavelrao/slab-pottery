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
import heapq

from flattening.algorithms import (
  initial_flattening,
  calculate_rho,
  calculate_rest_lengths,
  precompute_all_opposite_edges,
  gen_elastic_deformation_energy_distribution,
  update_mesh_with_path
)

from flattening.physics import calculate_node_energies 

from cutting.initial_cut import (make_cut, count_boundary_loops)
from scipy.interpolate import griddata

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
ENERGY_CUT_STOP = 1e-4

def cost(curr_node, next_node, node_energies):
  energy_diff = node_energies[curr_node] - node_energies[next_node]
  if energy_diff <= 0: # energy increase
    return 0.05
  else:
    return energy_diff # energy decrease(neg. cost)

def heuristic(curr_node, next_node, vertices2d):
  return np.linalg.norm(vertices2d[curr_node] - vertices2d[next_node])

def astar_energy(energy_graph, vertices2d, node_energies):
  max_energy_node_idx = np.argmax(node_energies)
  print(f'start point: {max_energy_node_idx} has energy {node_energies[max_energy_node_idx]}')
  # visited will store the best cost to get to a node
  visited = {max_energy_node_idx: 0}

  # open_set entry: f-score, cost, heuristic, path
  open_set = [(0, 0, 0, [max_energy_node_idx])]
  heapq.heapify(open_set)

  while len(open_set) != 0:
    fscore, path_cost, h, path = heapq.heappop(open_set)
    curr_node = path[len(path)-1] # last node 
    print(f'considering path: {path}')

    # check for goal reached:
    if (node_energies[curr_node] <= ENERGY_CUT_STOP):
      print(f'node {curr_node} with energy {node_energies[curr_node]} recognized as stopping point')
      print(f'slow energy decrease path: {path}')
      return path
    
    # check all neighbors to the current node
    for neighbor in energy_graph.neighbors(curr_node):
      # calculate total path cost to this neighbors
      neighbor_cost = path_cost + cost(curr_node, neighbor, node_energies)
      # if the neighbor hasn't been seen or the cost to get to the neighbor decreased,  update best path
      if neighbor not in visited or neighbor_cost < visited[neighbor]:
          print(f'better path to node {neighbor} found with total cost: {neighbor_cost} ')
          visited[neighbor] = neighbor_cost
          h  = heuristic(curr_node, neighbor, vertices2d)
          priority = neighbor_cost + h
          new_path = path.copy()
          new_path.append(neighbor)
          print(new_path)
          heapq.heappush(open_set, (priority, neighbor_cost, h, new_path))
        

def compute_energy_gradients(energy_sample_graph, energies, curr_node, positions2d):
  neighbors_to_gradients = []
  neighbors = energy_sample_graph.neighbors(curr_node)
  # caclculate gradients in neighbor dir and associuate with each neighbor
  for neighbor in neighbors:
    # calculate dist
    dist = np.linalg.norm(positions2d[neighbor] - positions2d[curr_node])
    gradient = (energies[neighbor] - energies[curr_node]) / dist
    neighbors_to_gradients.append((neighbor, gradient))
  return neighbors_to_gradients


    
def grow_crest_line(energy_sample_graph, energies, positions2d, direction, boundaries):
  boundary_vertices = np.unique(boundaries.flatten())
  print(boundary_vertices)
  max_energy_point_idx = np.argmax(energies)
  max_energy = energies[max_energy_point_idx]
  visited_points = set([max_energy_point_idx])
  curr_node = max_energy_point_idx
  iters = 0
  crest_line = []



  if (direction == 1):
    crest_line.append(max_energy_point_idx)
  while(iters < 15):
    iters +=1 

    # calculate gradients in dir of neighbors
    neighbors_to_gradients = compute_energy_gradients(energy_sample_graph, energies, curr_node, positions2d)

      # Filter for points in the appropriate direction (for growing in both directions)
    # if direction == 1 and len(crest_line) > 0:
    #   # Forward direction: prefer points that continue the current direction
    #   curr_coor = positions2d[curr_node]
    #   last_crest_coor = positions2d[crest_line[-1]]
    #   prev_direction = np.array(curr_coor[0] - last_crest_coor[0], curr_coor[1] - last_crest_coor[1])
    #   if np.linalg.norm(prev_direction) > 0:
    #     prev_direction = prev_direction / np.linalg.norm(prev_direction)
    #     gradients = [(n, g - 0.5 * np.dot(positions2d[n]- curr_coor, prev_direction)) for n, g in gradients]
          

    #sort the neighbors to gradients
    neighbors_to_gradients = sorted(neighbors_to_gradients, key=lambda x: x[1], reverse=True)

    # Choose the neighbor with the slowest descent (or fastest ascent)
    # but avoid already visited points
    next_point = None
    for candidate, _ in neighbors_to_gradients:
        if candidate not in visited_points:
            next_point = candidate
            break
            
    if next_point is None:
        break
        
    visited_points.add(next_point)
    crest_line.append(next_point)
    curr_node = next_point
    
    # Optional: Stop if energy drops too much (e.g., 50% of max energy)
    if energies[curr_node] < 0.5 * max_energy or curr_node in boundary_vertices:
        break
        
  return crest_line

# def calculate_isolines(energy_map: Dict[Tuple[float, float], float], num_levels: int = 10) -> Dict[float, List[Tuple[float, float]]]:
#   """
#   Calculate isolines from the energy distribution map.
  
#   Parameters:
#       energy_map: Dictionary mapping coordinates to energy values
#       num_levels: Number of isoline levels to compute
  
#   Returns:
#       Dictionary mapping energy levels to lists of points on that isoline
#   """
#   min_energy = min(energy_map.values())
#   max_energy = max(energy_map.values())
  
#   # Generate evenly spaced energy levels
#   levels = np.linspace(min_energy, max_energy, num_levels)
  
#   # Initialize isolines
#   isolines = {level: [] for level in levels}
  
#   # Simple approach: assign each point to the closest energy level
#   for point, energy in energy_map.items():
#     closest_level = min(levels, key=lambda l: abs(l - energy))
#     isolines[closest_level].append(point)
  
#   return isolines


def find_cut_lines(energy_sample_graph, energies, cut_line_num):
  curr_node = np.argmax(energies) # start at node with highest energy
  energy_decreasing = True
  path = list()
  path.append(curr_node)
  print(f'added node {curr_node}')
  iteration = 0
  while (energy_decreasing):
    iteration += 1
    print(iteration)
    neighbors = np.array(list(energy_sample_graph.neighbors(curr_node)))
    print(neighbors)
    if (len(neighbors) > 0) :
      delta_energies = energies[curr_node] - energies[neighbors]
      print(f'delta energies: {delta_energies}')
      min_energy_change = np.argmin(delta_energies)
      print(min_energy_change)
      if (np.abs(delta_energies[min_energy_change]) >= ENERGY_CHANGE_MIN) :
        curr_node = neighbors[min_energy_change]
        path.append(curr_node)
        print(f'adding {curr_node}')
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
  # join energy sets
  all_energies = np.concatenate([node_energies, interpolated_energies])
  print('lowest energies')
  print(sorted(all_energies, reverse=True)[:30])
  print('highest energies:')
  print(sorted(all_energies)[:30])
  # Normalize energy values
  energy_norm = (all_energies - np.min(all_energies)) / (np.max(all_energies) - np.min(all_energies))
  unflattened_centers = mesh.triangles_center
  all_sampled_positions = np.concatenate([flattened_vertices_2d_initial, flattened_mesh_centerpoints])
  all_sampled_positions3d = np.concatenate([mesh.vertices, unflattened_centers])

  # # Create a fine grid for interpolation
  # x_min, x_max = np.min(flattened_vertices_2d_initial[:, 0]), np.max(flattened_vertices_2d_initial[:, 0])
  # y_min, y_max = np.min(flattened_vertices_2d_initial[:, 1]), np.max(flattened_vertices_2d_initial[:, 1])
  # resolution = 400  # Higher for finer granularity
  # xi = np.linspace(x_min, x_max, resolution)
  # yi = np.linspace(y_min, y_max, resolution)
  # Xi, Yi = np.meshgrid(xi, yi)

  # grid_energy = griddata(all_sampled_positions, energy_norm, (Xi, Yi), method='linear')
  # Convert to colors using a colormap (e.g., 'jet' or 'viridis')
  #colormap = cm.jet(energy_norm)  # RGBA colors for each vertex
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111)
  ax.set_aspect("equal")  
  # Plot the interpolated energy field as an image
  # im = ax.pcolormesh(xi, yi, grid_energy, cmap=plt.cm.jet, shading='auto')
  face_verts_2d_initial = flattened_vertices_2d_initial[mesh.faces]
  poly_collection2 = PolyCollection(face_verts_2d_initial, facecolors="none", edgecolors="black", linewidths=0.5)
  ax.add_collection(poly_collection2)

  # Scatter plot of finer points with energy-based coloring
  sc = plt.scatter(all_sampled_positions[:, 0], all_sampled_positions[:, 1], c=all_energies, 
                 cmap=plt.cm.jet, s=40, edgecolors='k', linewidth=1.5)
  

  # Set plot limits for initial 2D plot
  min_coords = min(flattened_vertices_2d_initial[:, 0]), min(flattened_vertices_2d_initial[:, 1])
  max_coords = max(flattened_vertices_2d_initial[:, 0]), max(flattened_vertices_2d_initial[:, 1])
  range_x = max_coords[0] - min_coords[0]
  range_y = max_coords[1] - min_coords[1]
  padding_x = range_x * 0.1
  padding_y = range_y * 0.1
  ax.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
  ax.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)
  ax.set_title(f"Initial Surface (ER steps = {ENERGY_RELEASE_ITERATIONS}) with Energies")
  plt.tight_layout()
  plt.show()


  # # Optionally, add the positions used for interpolation as small dots
  # ax.scatter(all_sampled_positions[:, 0], all_sampled_positions[:, 1], s=5, color='black', alpha=0.3)
  # plt.tight_layout()
  # plt.show()

  # triang = plt.tri.Triangulation(flattened_vertices_2d_initial[:, 0], 
  #                                   flattened_vertices_2d_initial[:, 1],
  #                                   triangles=mesh.faces)
  # tcm = ax.tripcolor(triang, node_energies, cmap=plt.cm.jet, shading='gouraud')
  # Convert to colors using a colormap (e.g., 'jet' or 'viridis')
  #colormap = cm.jet(energy_norm)  # RGBA colors for each vertex
   # Convert to colors using a colormap (e.g., 'jet' or 'viridis')
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_trisurf(all_sampled_positions[:, 0], all_sampled_positions[:, 1], all_energies, triangles=mesh.faces, cmap=plt.cm.jet, linewidth=0.2, antialiased=True, alpha=0.8, shade=True, facecolors=plt.cm.jet(energy_norm))
  cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
  cbar.set_label('Energy Value')
  plt.show()

  path = astar_energy(energy_graph=energy_sample_graph, vertices2d=all_sampled_positions, node_energies=all_energies)
  print(path)

  # num_loops, boundary_loops = count_boundary_loops(mesh, np.arange(len(mesh.faces)))
  # print(boundary_loops)
  # forward_path = grow_crest_line(energy_sample_graph, all_energies, all_sampled_positions, 1, boundary_loops)
  # print(forward_path)
  #backward_path = grow_crest_line(energy_sample_graph, all_energies, all_sampled_positions, -1)
 # path = list(reversed(backward_path)) + forward_path
  #print(f'cocatenated path: {path}')
#   path = forward_path
#   path = find_cut_lines(energy_sample_graph, all_energies, 1)
#   print(path)
#   print(f'previous mesh vertex list length: {len(mesh.vertices)}')
  updated_mesh_to_cut, path = update_mesh_with_path(mesh, path, energy_sample_graph, all_sampled_positions3d)
#   print(f'new mesh vertex list length: {len(updated_mesh_to_cut.vertices)}')
#   print(len(path))
#   print(f'updated path indexing: {path}')

 # 3D plot (top-left)
  fig = plt.figure(figsize=(18, 10))
  ax3d = fig.add_subplot(111, projection='3d')
  ax3d.plot_trisurf(updated_mesh_to_cut.vertices[:, 0], updated_mesh_to_cut.vertices[:, 1], updated_mesh_to_cut.vertices[:, 2],
                  triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=0.7)
  
  path_vertices = updated_mesh_to_cut.vertices[path]
  print(path_vertices)

  #Plot the cutting path as connected line segments
  for i in range(len(path_vertices) - 1):
      ax3d.plot(
          [path_vertices[i, 0], path_vertices[i + 1, 0]],  # X-coordinates
          [path_vertices[i, 1], path_vertices[i + 1, 1]],  # Y-coordinates
          [path_vertices[i, 2], path_vertices[i + 1, 2]],  # Z-coordinates
          color='red', linewidth=2, zorder=10
      )
  ax3d.plot(
    path_vertices[:, 0], path_vertices[:, 1], path_vertices[:, 2], 
    color='red', linewidth=2, marker='o', markersize=5, label="Cutting Path", zorder=10
  )
  ax3d.set_title("new 3D Surface")
  ax3d.set_xlabel("X")
  ax3d.set_ylabel("Y")
  ax3d.set_zlabel("Z")
  plt.show()

#   ax = fig.add_subplot(231)
#   ax.set_aspect("equal")  
#   # Plot the interpolated energy field as an image
#   # im = ax.pcolormesh(xi, yi, grid_energy, cmap=plt.cm.jet, shading='auto')
#   face_verts_2d_initial = flattened_vertices_2d_initial[mesh.faces]
#   poly_collection2 = PolyCollection(face_verts_2d_initial, facecolors="none", edgecolors="black", linewidths=0.5)
#   ax.add_collection(poly_collection2)
#   path_coor= all_sampled_positions[path]
#   plt.plot(path_coor[:, 0], path_coor[:, 1], 'r-', linewidth=2.5, label="Crest Line")
#   plt.show()

  # 2d plot

  
if __name__ == "__main__":
  main()
