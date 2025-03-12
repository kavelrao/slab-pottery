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
from heapdict import heapdict

from flattening.algorithms import (
  initial_flattening,
  calculate_rho,
  calculate_rest_lengths,
  precompute_all_opposite_edges,
  gen_elastic_deformation_energy_distribution,
  update_mesh_with_path,
  surface_flattening_spring_mass
)

from flattening.physics import calculate_node_energies 

from cutting.initial_cut import (make_cut, count_boundary_loops, find_cutting_path)
from scipy.interpolate import griddata
from plotting.mesh_plotting import plot_flat_mesh_with_node_energies

# Configuration
STL_FILE = 'Partial_Oblong_Cylinder_Shell_Coarse'#'Partial_Open_Bulb_Coarse'
USE_PRECOMPUTED = True

ENABLE_ENERGY_RELEASE_IN_FLATTEN = True
ENERGY_RELEASE_ITERATIONS = 10  # Number of iterations between applying energy release in the initial flattening

ENABLE_ENERGY_RELEASE_PHASE = True

ENERGY_RELEASE_TIMESTEP = 0.01
ENERGY_RELEASE_PENALTY_COEFFICIENT = 1.0
PERMISSIBLE_ENERGY_VARIATION = 0.0005
ENERGY_CHANGE_MIN = 0.00001
ENERGY_CUT_STOP = 1e-1 /2

def calculate_gradient_at_node(node, graph, vertices3d, node_energies):
    """
    Calculate an approximate energy gradient at a node using its neighbors
    """
    # Initialize gradient vector
    gradient = np.zeros(3)
    total_weight = 0
    
    # For each neighbor, calculate the direction and energy difference
    for neighbor in graph.neighbors(node):
        # Vector pointing from current node to neighbor
        direction = vertices3d[neighbor] - vertices3d[node]
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Normalize the direction
            direction = direction / distance
            
            # Energy difference (negative means energy decreases in this direction)
            energy_diff = node_energies[neighbor] - node_energies[node]
            
            # Weight by inverse distance (closer neighbors have stronger influence)
            weight = 1.0 / distance
            
            # Accumulate weighted contribution to gradient
            gradient += direction * energy_diff * weight
            total_weight += weight
    
    # Average the gradient
    if total_weight > 0:
        gradient = gradient / total_weight
        
    return gradient

# def cost(curr_node, next_node, start_node, node_energies, local_dist, dist_so_far):
#     # Get energy values
#     curr_energy = node_energies[curr_node]
#     next_energy = node_energies[next_node]
#     start_energy = node_energies[start_node]
    
#     # Local energy gradient (between current and next)
#     local_diff = curr_energy - next_energy
    
#     # Global energy gradient (from start to next)
#     global_diff = start_energy - next_energy
    
#     # If we're going uphill locally, strongly penalize
#     if global_diff <= 0:
#         return 1000 - local_diff  # Increased penalty for uphill movement
    
#     # Add distance to next node
#     total_dist = dist_so_far + local_dist
    
#     # Calculate local and global gradients
#     local_gradient = local_diff / local_dist
#     global_gradient = global_diff / total_dist
    
#     # Blend local and global gradients (favor smoother paths)
#     # Adjust weights to control importance of local vs global gradient
#     blended_cost = global_diff#+ global_gradient * 0.6
    
#     return blended_cost

# def cost(start_node, next_node, curr_node, node_energies, dist, local_dist):
#   energy_diff_local = node_energies[curr_node] - node_energies[next_node]
#   energy_diff_global = node_energies[start_node] - node_energies[next_node]
#   if (energy_diff_local <=0):
#     return -1 * energy_diff_local * 10
#   else:
#     return energy_diff_global #/ dist #+  energy_diff_local# energy decrease(neg. cost)
def cost(start_node, next_node, node_energies, dist):
  energy_diff = node_energies[start_node] - node_energies[next_node]
  if energy_diff <= 0: # energy increase
    return 1000 - energy_diff
  else:
    return energy_diff / dist# energy decrease(neg. cost)


def heuristic(curr_node, next_node, vertices3d):
  return np.linalg.norm(vertices3d[curr_node] - vertices3d[next_node])


#BEST ONE
def astar_energy(energy_graph, vertices3d, node_energies):
  node_energies =  (node_energies - np.min(node_energies)) / (np.max(node_energies) - np.min(node_energies))
  max_energy_node_idx = np.argmax(node_energies)
  print(f'start point: {max_energy_node_idx} has energy {node_energies[max_energy_node_idx]}')
  # visited will store the best cost to get to a node
  visited = {max_energy_node_idx: 0}
  # open_set entry: f-score, cost, heuristic, dist path
  open_set = [(0, 0, 0, 0, [max_energy_node_idx])]
  heapq.heapify(open_set)
  while len(open_set) != 0:
    fscore, path_cost, h, dist_so_far, path = heapq.heappop(open_set)
    curr_node = path[len(path)-1] # last node
    print(f'considering path: {path}')
    # check for goal reached:
    if (node_energies[curr_node] <= ENERGY_CUT_STOP):
      print(f'node {curr_node} with energy {node_energies[curr_node]} recognized as stopping point')
      print(f'slow energy decrease path: {path}')
      return path
   
    # check all neighbors to the current node
    for neighbor in energy_graph.neighbors(curr_node):
      # calculate total path cost to this neighbor
      start_node = path[0]
      dist = dist_so_far + np.abs(np.linalg.norm(vertices3d[neighbor] - vertices3d[start_node]))
      neighbor_cost = path_cost + cost(path[0], neighbor, node_energies, dist)
      # if the neighbor hasn't been seen or the cost to get to the neighbor decreased,  update best path
      if neighbor not in visited or neighbor_cost < visited[neighbor]:
          print(f'better path to node {neighbor} found with total cost: {neighbor_cost} ')
          visited[neighbor] = neighbor_cost
          h  = heuristic(curr_node, neighbor, vertices3d)
          priority = neighbor_cost + h
          new_path = path.copy()
          new_path.append(neighbor)
          print(new_path)
          heapq.heappush(open_set, (priority, neighbor_cost, h, dist, new_path))
       

# def astar_energy(energy_graph, vertices3d, node_energies):
#   node_energies =  (node_energies - np.min(node_energies)) / (np.max(node_energies) - np.min(node_energies))
#   max_energy_node_idx = np.argmax(node_energies)
#   print(f'start point: {max_energy_node_idx} has energy {node_energies[max_energy_node_idx]}')
#   # visited will store the best cost to get to a node
#   visited = {max_energy_node_idx: 0}

#   # open_set entry: f-score, cost, heuristic, dist path
#   open_set = [(0, 0, 0, 0, [max_energy_node_idx])]
#   heapq.heapify(open_set)

#   while len(open_set) != 0:
#     fscore, path_cost, h, path_dist, path = heapq.heappop(open_set)
#     curr_node = path[len(path)-1] # last node 
#     print(f'considering path: {path}')

#     # check for goal reached:
#     if (node_energies[curr_node] <= ENERGY_CUT_STOP):
#       print(f'node {curr_node} with energy {node_energies[curr_node]} recognized as stopping point')
#       print(f'slow energy decrease path: {path}')
#       return path
    
#     # check all neighbors to the current node
#     for neighbor in energy_graph.neighbors(curr_node):
#       # calculate total path cost to this neighbor
#       # start_node = path[0]
#       # dist = dist_so_far + np.abs(np.linalg.norm(vertices3d[neighbor] - vertices3d[start_node]))
#       # neighbor_cost = path_cost + cost(path[0], neighbor, node_energies, dist)
#       local_dist = np.linalg.norm(vertices3d[neighbor] - vertices3d[curr_node])
#       neighbor_cost = path_cost + cost(curr_node, neighbor, path[0], node_energies, local_dist, path_dist)
#       # if the neighbor hasn't been seen or the cost to get to the neighbor decreased,  update best path
#       if neighbor not in visited or neighbor_cost < visited[neighbor]:
#           print(f'better path to node {neighbor} found with total cost: {neighbor_cost}')
#           visited[neighbor] = neighbor_cost
#           h  = heuristic(curr_node, neighbor, vertices3d)
#           priority = neighbor_cost + h
#           new_path = path.copy()
#           new_path.append(neighbor)
#           print(new_path)
#           heapq.heappush(open_set, (priority, neighbor_cost, h, path_dist + local_dist, new_path))
        

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

# def astar_energy(energy_graph, vertices3d, node_energies):
#   node_energies =  (node_energies - np.min(node_energies)) / (np.max(node_energies) - np.min(node_energies))
#   max_energy_node_idx = np.argmax(node_energies)
#   print(f'start point: {max_energy_node_idx} has energy {node_energies[max_energy_node_idx]}')
#   # visited will store the best cost to get to a node
#   visited = {max_energy_node_idx: 0}
#   # open_set entry: f-score, cost, heuristic, dist path
#   open_set = [(0, 0, 0, 0, [max_energy_node_idx])]
#   heapq.heapify(open_set)
#   while len(open_set) != 0:
#     fscore, path_cost, h, dist_so_far, path = heapq.heappop(open_set)
#     curr_node = path[len(path)-1] # last node
#     print(f'considering path: {path}')
#     # check for goal reached:
#     if (node_energies[curr_node] <= ENERGY_CUT_STOP):
#       print(f'node {curr_node} with energy {node_energies[curr_node]} recognized as stopping point')
#       print(f'slow energy decrease path: {path}')
#       return path
   
#     # check all neighbors to the current node
#     for neighbor in energy_graph.neighbors(curr_node):
#       # calculate total path cost to this neighbor
#       start_node = path[0]
#       local_dist = np.linalg.norm(vertices3d[neighbor] - vertices3d[curr_node])
#       dist = dist_so_far + local_dist
#       neighbor_cost = path_cost + cost(path[0], neighbor, curr_node, node_energies, dist, local_dist)
#       # if the neighbor hasn't been seen or the cost to get to the neighbor decreased,  update best path
#       if neighbor not in visited or neighbor_cost < visited[neighbor]:
#           print(f'better path to node {neighbor} found with total cost: {neighbor_cost} ')
#           visited[neighbor] = neighbor_cost
#           h  = heuristic(curr_node, neighbor, vertices3d)
#           priority = neighbor_cost + h
#           new_path = path.copy()
#           new_path.append(neighbor)
#           print(new_path)
#           heapq.heappush(open_set, (priority, neighbor_cost, h, dist, new_path))

# def astar_energy(energy_graph, vertices3d, node_energies):
#   node_energies =  (node_energies - np.min(node_energies)) / (np.max(node_energies) - np.min(node_energies))
#   max_energy_node_idx = np.argmax(node_energies)
#   print(f'start point: {max_energy_node_idx} has energy {node_energies[max_energy_node_idx]}')
#   # visited will store the best cost to get to a node
#   visited = {max_energy_node_idx: 0}
#   # open_set entry: f-score, cost, heuristic, dist path
#   open_set = heapdict()
#   open_set[max_energy_node_idx] = (0, 0, 0, 0, [max_energy_node_idx])
#   #heapq.heapify(open_set)
#   while len(open_set) != 0:
#     curr_node, (fscore, path_cost, h, dist_so_far, path) = open_set.popitem()
#     #curr_node = path[len(path)-1] # last node
#     print(f'considering path: {path}')
#     # check for goal reached:
#     if (node_energies[curr_node] <= ENERGY_CUT_STOP):
#       print(f'node {curr_node} with energy {node_energies[curr_node]} recognized as stopping point')
#       print(f'slow energy decrease path: {path}')
#       return path
   
#     # check all neighbors to the current node
#     for neighbor in energy_graph.neighbors(curr_node):
#       # calculate total path cost to this neighbor
#       start_node = path[0]
#       local_dist = np.linalg.norm(vertices3d[neighbor] - vertices3d[curr_node])
#       dist = dist_so_far + local_dist
#       neighbor_cost = cost(path[0], neighbor, curr_node, node_energies, dist, local_dist)
#       # if the neighbor hasn't been seen or the cost to get to the neighbor decreased,  update best path
#       if neighbor not in visited or neighbor_cost < visited[neighbor]:
#           print(f'better path to node {neighbor} found with total cost: {neighbor_cost} ')
#           visited[neighbor] = neighbor_cost
#           h  = heuristic(curr_node, neighbor, vertices3d)
#           priority = neighbor_cost + h
#           new_path = path.copy()
#           new_path.append(neighbor)
#           print(new_path)
#           open_set[neighbor] = (priority, neighbor_cost, h, dist, new_path)

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

def plot_energy_for_mesh():
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

  # now caculate energy
  node_energies = calculate_node_energies(flattened_vertices_2d_initial, mesh.edges_unique, rest_lengths,spring_constant=0.5)

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

  plot_flat_mesh_with_node_energies(mesh, flattened_vertices_2d_initial, all_sampled_positions, all_energies)



def main():
  #plot_energy_for_mesh()
  # Load mesh from file
  mesh = trimesh.load("files/" + STL_FILE + ".stl")
  
  # cutting_path = find_cutting_path(mesh)
  # mesh, final_cuts = make_cut(mesh, cutting_path)
  # # Visualize the cut mesh, also outline the first two triangles in red

  # fig = plt.figure(figsize=(10, 10))

  # ax = fig.add_subplot(111, projection='3d')

  # # Plot the mesh
  # ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
  #                 triangles=mesh.faces, color='blue', alpha=0.7)
  
  # # Plot the final cuts
  # for cut in final_cuts:
  #     ax.plot(mesh.vertices[cut, 0], mesh.vertices[cut, 1], mesh.vertices[cut, 2], color='red')
  
  # plt.show()

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

  # PLOT INITIAL FLATTEN WITH ENERGIES
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111)
  ax.set_aspect("equal")  
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

  # PLOT 3D MESH WITH ENERGIES
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_trisurf(all_sampled_positions[:, 0], all_sampled_positions[:, 1], all_energies, triangles=mesh.faces, cmap=plt.cm.jet, linewidth=0.2, antialiased=True, alpha=0.8, shade=True, facecolors=plt.cm.jet(energy_norm))
  cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
  cbar.set_label('Energy Value')
  plt.show()

  # FIND SLOWEST DESCENT PATH
  path = astar_energy(energy_graph=energy_sample_graph, vertices3d=all_sampled_positions3d, node_energies=all_energies)
  print(path)
  # update the mesh to include interpolated vertices from path if needed
  print(f'len of faces in original mesh: {len(mesh.faces)}')
  updated_mesh_to_cut, path = update_mesh_with_path(mesh, path, energy_sample_graph, all_sampled_positions3d)
  print(f'updated_mesh now has {len(updated_mesh_to_cut.vertices)} vertices and {len(updated_mesh_to_cut.faces)} faces')
  print(f'old mesh had {len(mesh.vertices)} vertices and {len(mesh.faces)} faces')

    # Print face adjacency
  print("Face adjacency:\n", updated_mesh_to_cut.face_adjacency)

  # Check if adjacency is empty
  if updated_mesh_to_cut.face_adjacency.shape[0] == 0:
    print("Warning: No face adjacency found!")

  adjacency = updated_mesh_to_cut.face_adjacency
  # Get connected components
  components = trimesh.graph.connected_components(
    edges=adjacency,  # Face adjacency edges
    min_len=1  # Ignore small components if needed
  )

  # Print the number of components
  print(f"Number of connected components: {len(components)}")

  # Print the first component's face indices
  print("First component face indices:", sorted(components[0]))
  print("Second component face indices:", components[1])
  print(updated_mesh_to_cut.faces[components[1]])
  # PLOT MESH WITH CUT LINES
 # 3D plot (top-left)
  fig = plt.figure(figsize=(18, 10))
  ax3d = fig.add_subplot(111, projection='3d')
  ax3d.plot_trisurf(updated_mesh_to_cut.vertices[:, 0], updated_mesh_to_cut.vertices[:, 1], updated_mesh_to_cut.vertices[:, 2],
                  triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=0.1,zorder=1)
  
  path_vertices = updated_mesh_to_cut.vertices[path]
  print(path_vertices)

  #Plot the cutting path as connected line segments
  for i in range(len(path_vertices) - 1):
      ax3d.plot(
          [path_vertices[i, 0], path_vertices[i + 1, 0]],  # X-coordinates
          [path_vertices[i, 1], path_vertices[i + 1, 1]],  # Y-coordinates
          [path_vertices[i, 2], path_vertices[i + 1, 2]],  # Z-coordinates
          color='orange', linewidth=1, zorder=9
      )
  # ax3d.plot(
  #   path_vertices[:, 0], path_vertices[:, 1], path_vertices[:, 2], 
  #   color='orange', linewidth=2, marker='o', markersize=5, label="Cutting Path", zorder=9
  # )

  # PLOT UNCONNECTED FACES
  faces_vertices = updated_mesh_to_cut.faces[components[1]].flatten()
  print(f'disconnected faces (idx) {faces_vertices}')
  print(f'disconnected faces: {updated_mesh_to_cut.faces[faces_vertices]}')

  #Plot the unnconnected components edges
  # for f in updated_mesh_to_cut.faces[components[1]]:
  #    face_vertices = updated_mesh_to_cut.vertices[f]
  #    ax3d.plot(
  #         [updated_mesh_to_cut.vertices[f[:, 0]], f:[i + 1, 0]],  # X-coordinates
  #         [path_vertices[i, 1], path_vertices[i + 1, 1]],  # Y-coordinates
  #         [path_vertices[i, 2], path_vertices[i + 1, 2]],  # Z-coordinates
  #         color='orange', linewidth=1, zorder=9
  #     )
  #ax3d.scatter(updated_mesh_to_cut.vertices[faces_vertices, 0], updated_mesh_to_cut.vertices[faces_vertices, 1],  updated_mesh_to_cut.vertices[faces_vertices, 2], color='red', zorder=10) 

  #  # Scatter plot of finer points with energy-based coloring
  print(len(all_sampled_positions3d))
  print(len(all_energies))
  # sc = ax3d.scatter(all_sampled_positions3d[:, 0], all_sampled_positions3d[:, 1], all_sampled_positions3d[:, 2], c=all_energies, 
  #                cmap=plt.cm.jet, s=40, edgecolors='k', linewidth=1.5)

  ax3d.set_title("new 3D Surface")
  ax3d.set_xlabel("X")
  ax3d.set_ylabel("Y")
  ax3d.set_zlabel("Z")
  plt.show()

  # Cut the mesh
  edge_path = []
  for i in range(len(path) - 1):
     edge_path.append((path[i], path[i+1]))
  print(f'path of edges: {edge_path}')
  mesh_cut, final_cuts = make_cut(updated_mesh_to_cut, edge_path)
  print(f'path length: {len(edge_path)}')
  print(f'final cuts length: {len(final_cuts)}')
  print(final_cuts)
  print(len(mesh_cut.edges))
  print(len(mesh.edges))
  print(len(mesh_cut.faces))
  print(len(mesh.faces))
  print(len(mesh_cut.vertices))

  # PLOT THE CUT MESH
  # 3D plot (top-left)
  fig = plt.figure(figsize=(18, 10))
  ax3d = fig.add_subplot(111, projection='3d')
  ax3d.plot_trisurf(mesh_cut.vertices[:, 0], mesh_cut.vertices[:, 1], mesh_cut.vertices[:, 2],
                  triangles=mesh_cut.faces, cmap='viridis', edgecolor='black', alpha=0.7)
   #Plot the cutting path as connected line segments
  #Plot the final cuts
  for cut in final_cuts:
      ax3d.plot(mesh_cut.vertices[cut, 0], mesh_cut.vertices[cut, 1], mesh_cut.vertices[cut, 2], color='red', zorder=10)
  ax3d.set_title("cut mesh")
  plt.show()
  
  # Perform flattening on new, cut mesh
  if USE_PRECOMPUTED and os.path.exists("files/" + STL_FILE + "_additionalcuts.npy"):
    flattened_vertices_2d_cut = np.load("files/" + STL_FILE + "_additionalcuts.npy")
  else:
    flattened_vertices_2d_cut, flattened_vertices_2d_initial, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = surface_flattening_spring_mass(
        mesh_cut,
        enable_energy_release_in_flatten=True,
        enable_energy_release_phase=False,
        energy_release_iterations=50,
        area_density=None,
        vertices_2d_initial=None,
        dt=ENERGY_RELEASE_TIMESTEP,
        permissible_energy_variation=PERMISSIBLE_ENERGY_VARIATION,
        penalty_coefficient=ENERGY_RELEASE_PENALTY_COEFFICIENT,
        object_name=STL_FILE,
    )
    np.save(Path(__file__).parent / "files" / (STL_FILE + "_additionalcuts.npy"), flattened_vertices_2d_initial)

  # calculate updated energy
  rest_lengths = calculate_rest_lengths(mesh_cut.vertices, mesh_cut.edges_unique)
  node_energies = calculate_node_energies(flattened_vertices_2d_cut, mesh_cut.edges_unique, rest_lengths,spring_constant=0.5)
  # plot new mesh!
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111)
  ax.set_aspect("equal")  
  # Plot the interpolated energy field as an image
  # im = ax.pcolormesh(xi, yi, grid_energy, cmap=plt.cm.jet, shading='auto')
  face_verts_2d = flattened_vertices_2d_cut[mesh_cut.faces]
  poly_collection3 = PolyCollection(face_verts_2d, facecolors="none", edgecolors="black", linewidths=0.5)
  ax.add_collection3(poly_collection3)

  # Scatter plot of finer points with energy-based coloring
  ax.scatter(flattened_vertices_2d_cut[:, 0], flattened_vertices_2d_cut[:, 1], c=all_energies, 
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
  ax.set_title(f"Final surface after additional cut (ER steps = {ENERGY_RELEASE_ITERATIONS}) with Energies")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
