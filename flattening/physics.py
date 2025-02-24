"""
Physics simulation utilities for spring-mass system.
"""

import numpy as np
from numpy.typing import NDArray
import trimesh
from tqdm import trange

from .geometry import (
    calculate_vertex_areas,
    point_to_segment_distance_2d,
    point_to_segment_distance_3d,
    get_opposite_edges
)


def calculate_rest_lengths(vertices_3d: NDArray[np.float64], edges: NDArray[np.int64]) -> dict:
    """Calculate the rest lengths for all edges in the mesh."""
    rest_lengths = {}
    for edge in edges:
        v1_idx, v2_idx = edge
        rest_lengths[(v1_idx, v2_idx)] = np.linalg.norm(
            vertices_3d[v1_idx] - vertices_3d[v2_idx]
        )
        rest_lengths[(v2_idx, v1_idx)] = rest_lengths[(v1_idx, v2_idx)]  # Bidirectional
    
    return rest_lengths


def calculate_masses(vertices_3d: NDArray[np.float64], faces: NDArray[np.int64], area_density: float) -> NDArray[np.float64]:
    """Calculate the masses for all vertices in the mesh."""
    vertex_areas = calculate_vertex_areas(vertices_3d, faces)
    return vertex_areas * area_density


def calculate_penalty_displacements(vertices_3d, faces, vertices_2d, penalty_coefficient=1.0):
    """
    Calculate penalty displacements for each vertex to prevent overlap.
    """
    num_vertices = len(vertices_3d)
    penalty_displacements = np.zeros_like(vertices_2d)
    
    for vertex_index in range(num_vertices):
        opposite_edges_indices = get_opposite_edges(vertex_index, faces)
        p_i_2d = vertices_2d[vertex_index]
        q_i_3d = vertices_3d[vertex_index]
        penalty_displacement_vertex = np.zeros(2)
        
        for opp_edge_idx_pair in opposite_edges_indices:
            v_j_index, v_k_index = opp_edge_idx_pair
            
            p_j_2d = vertices_2d[v_j_index]
            p_k_2d = vertices_2d[v_k_index]
            q_j_3d = vertices_3d[v_j_index]
            q_k_3d = vertices_3d[v_k_index]
            
            h_j, n_hat_2d = point_to_segment_distance_2d(p_i_2d, p_j_2d, p_k_2d)
            h_star_j, _ = point_to_segment_distance_3d(q_i_3d, q_j_3d, q_k_3d)
            
            if h_j <= h_star_j:
                c_penalty = 1.0
            else:
                c_penalty = 0.0
            
            if c_penalty > 0:
                n_hat_norm = np.linalg.norm(n_hat_2d)
                if n_hat_norm > 1e-9:
                    n_hat_2d = n_hat_2d / n_hat_norm
                else:
                    print("Warning: n_hat_norm was close to zero, defaulting additional penalty displacement to 0")
                    n_hat_2d = np.array([0.0, 0.0])
                
                penalty_displacement_magnitude = penalty_coefficient * c_penalty * abs(h_j - h_star_j)
                penalty_displacement_vertex += penalty_displacement_magnitude * n_hat_2d
        
        penalty_displacements[vertex_index] = -penalty_displacement_vertex
    
    return penalty_displacements


def calculate_forces(vertices_2d: NDArray[np.float64], edges: NDArray[np.int64], 
                     rest_lengths: dict, spring_constant: float) -> NDArray[np.float64]:
    """Calculate the forces for all vertices based on the spring-mass model."""
    forces = np.zeros_like(vertices_2d)
    
    for edge in edges:
        v1_idx, v2_idx = edge
        p1_2d = vertices_2d[v1_idx]
        p2_2d = vertices_2d[v2_idx]
        current_length_2d = np.linalg.norm(p2_2d - p1_2d)
        rest_length = rest_lengths[(v1_idx, v2_idx)]
        
        if current_length_2d < 1e-9:
            raise ValueError("current_length_2d is nearly zero in force calculation")
        
        direction = (p2_2d - p1_2d) / current_length_2d  # Unit vector
        force_magnitude = spring_constant * (current_length_2d - rest_length)
        force_vector = force_magnitude * direction
        
        forces[v1_idx] += -force_vector  # Equal and opposite forces
        forces[v2_idx] += force_vector
    
    return forces


def calculate_energy(vertices_2d: NDArray[np.float64], edges: NDArray[np.int64], 
                     rest_lengths: dict, spring_constant: float) -> float:
    """Calculate the total energy in the spring-mass system."""
    energy = 0.0
    for edge in edges:
        v1_idx, v2_idx = edge
        p1_2d = vertices_2d[v1_idx]
        p2_2d = vertices_2d[v2_idx]
        current_length_2d = np.linalg.norm(p2_2d - p1_2d)
        rest_length = rest_lengths[(v1_idx, v2_idx)]
        energy += 0.5 * spring_constant * (current_length_2d - rest_length) ** 2
    
    # Double the energy to match the paper's formula which sums over vertices, not edges
    return energy * 2


def calculate_rho(mesh: trimesh.Trimesh) -> float:
    """
    Calculate the optimal area density (rho) value for mass calculation.
    
    This ensures the minimum mass is normalized to 1.0, as described in the paper.
    """
    rho_current = 1.0  # Initial guess for rho
    tolerance_rho = 1e-6
    max_iterations_rho = 100
    rho_values_history = []
    rho_relaxation_factor = 0.1
    
    for rho_iteration in trange(max_iterations_rho, desc="Calculating rho"):
        vertex_areas = calculate_vertex_areas(mesh.vertices, mesh.faces)
        masses_rho_iter = vertex_areas * rho_current
        
        min_m = np.min(masses_rho_iter)
        if min_m < 1e-9:
            rho_new = rho_current
        else:
            rho_new = 1.0 / min_m
        
        # Relaxation update rule
        rho_current = (1 - rho_relaxation_factor) * rho_current + rho_relaxation_factor * rho_new
        rho_values_history.append(rho_current)
        
        relative_change_rho = abs((rho_new - rho_current) / rho_current) if rho_current != 0 else rho_new
        if relative_change_rho < tolerance_rho:
            print(f"Rho converged in {rho_iteration + 1} iterations, rho = {rho_current:.6f}")
            break
    
    else:  # Loop completed without break (max iterations reached)
        print(
            f"Warning: Rho iteration reached max iterations ({max_iterations_rho}), "
            f"convergence not guaranteed, using rho = {rho_current:.6f}."
        )
        print("Rho values history:", rho_values_history)
    
    return rho_current