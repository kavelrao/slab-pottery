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
    get_opposite_edges,
    point_to_segment_distance_2d_batch,
    point_to_segment_distance_3d_batch,
    precompute_all_opposite_edges,
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


def calculate_penalty_displacements_vectorized(vertices_3d, faces, vertices_2d, penalty_coefficient=1.0, all_opposite_edges=None):
    """
    Vectorized version of penalty displacement calculation for better performance.
    
    Args:
        vertices_3d: 3D vertex positions
        faces: Face indices
        vertices_2d: 2D vertex positions
        penalty_coefficient: Coefficient to control penalty strength
        all_opposite_edges: Optional precomputed opposite edges for all vertices
        
    Returns:
        numpy.ndarray: Penalty displacement vectors for each vertex
    """
    
    num_vertices = len(vertices_3d)
    penalty_displacements = np.zeros_like(vertices_2d)
    
    # Precompute all opposite edges if not provided (major optimization)
    if all_opposite_edges is None:
        all_opposite_edges = precompute_all_opposite_edges(vertices_3d, faces)
    
    # For each vertex, collect all its opposite edges and process them in batches
    for vertex_index in range(num_vertices):
        opposite_edges_indices = all_opposite_edges[vertex_index]
        
        if not opposite_edges_indices:  # Skip if no opposite edges
            continue
        
        # Extract edge endpoints
        edge_endpoints = np.array(opposite_edges_indices)
        v_j_indices = edge_endpoints[:, 0]
        v_k_indices = edge_endpoints[:, 1]
        
        # Extract points needed for distance calculations
        p_i_2d = vertices_2d[vertex_index]
        q_i_3d = vertices_3d[vertex_index]
        
        # Create batches for point-to-segment distance calculations
        p_i_2d_batch = np.tile(p_i_2d, (len(edge_endpoints), 1))  # Repeat the vertex for each edge
        q_i_3d_batch = np.tile(q_i_3d, (len(edge_endpoints), 1))
        
        # Get 2D segment endpoints
        p_j_2d_batch = vertices_2d[v_j_indices]
        p_k_2d_batch = vertices_2d[v_k_indices]
        
        # Get 3D segment endpoints
        q_j_3d_batch = vertices_3d[v_j_indices]
        q_k_3d_batch = vertices_3d[v_k_indices]
        
        # Calculate distances and vectors using batch functions
        h_j_batch, n_hat_2d_batch = point_to_segment_distance_2d_batch(
            p_i_2d_batch, p_j_2d_batch, p_k_2d_batch
        )
        h_star_j_batch, _ = point_to_segment_distance_3d_batch(
            q_i_3d_batch, q_j_3d_batch, q_k_3d_batch
        )
        
        # Determine which edges generate penalties
        penalty_mask = h_j_batch <= h_star_j_batch
        
        if not np.any(penalty_mask):
            continue  # No penalties for this vertex
        
        # Process only edges that generate penalties
        penalty_indices = np.where(penalty_mask)[0]
        n_hat_2d_penalty = n_hat_2d_batch[penalty_indices]
        
        # Normalize direction vectors
        n_hat_norms = np.linalg.norm(n_hat_2d_penalty, axis=1)
        valid_norm_mask = n_hat_norms > 1e-9
        
        if not np.any(valid_norm_mask):
            continue  # No valid norms
        
        # Only process valid norms
        valid_indices = penalty_indices[valid_norm_mask]
        valid_n_hat_2d = n_hat_2d_penalty[valid_norm_mask]
        valid_n_hat_norms = n_hat_norms[valid_norm_mask]
        
        # Normalize
        valid_n_hat_2d = valid_n_hat_2d / valid_n_hat_norms[:, np.newaxis]
        
        # Calculate penalty magnitudes
        penalty_magnitudes = penalty_coefficient * np.abs(
            h_j_batch[valid_indices] - h_star_j_batch[valid_indices]
        )
        
        # Calculate penalty vectors
        penalty_vectors = penalty_magnitudes[:, np.newaxis] * valid_n_hat_2d
        
        # Sum penalty vectors for this vertex
        penalty_displacement_vertex = np.sum(penalty_vectors, axis=0)
        
        # Apply penalty displacement
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


def calculate_forces_vectorized(vertices_2d: NDArray[np.float64], edges: NDArray[np.int64], 
                             rest_lengths: dict, spring_constant: float) -> NDArray[np.float64]:
    """
    Vectorized implementation of force calculation for better performance.
    
    Args:
        vertices_2d: 2D vertex positions
        edges: Edge indices
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for force calculation
        
    Returns:
        numpy.ndarray: Force vectors for each vertex
    """
    num_vertices = len(vertices_2d)
    forces = np.zeros((num_vertices, 2), dtype=np.float64)
    
    # Extract edge endpoints
    v1_indices = edges[:, 0]
    v2_indices = edges[:, 1]
    
    # Get positions of edge endpoints
    p1_positions = vertices_2d[v1_indices]
    p2_positions = vertices_2d[v2_indices]
    
    # Calculate edge vectors and current lengths
    edge_vectors = p2_positions - p1_positions
    current_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Check for near-zero lengths
    mask = current_lengths > 1e-9
    if not np.all(mask):
        raise ValueError("Some edges have near-zero lengths in vectorized force calculation")
    
    # Get rest lengths for edges (convert tuple keys to indices)
    rest_length_values = np.array([rest_lengths[(v1, v2)] for v1, v2 in edges])
    
    # Calculate force magnitudes
    force_magnitudes = spring_constant * (current_lengths - rest_length_values)
    
    # Normalize edge vectors to get directions
    directions = edge_vectors / current_lengths[:, np.newaxis]
    
    # Calculate force vectors
    force_vectors = force_magnitudes[:, np.newaxis] * directions
    
    # Apply forces to vertices using numpy's add.at for accumulation
    np.add.at(forces, v1_indices, -force_vectors)
    np.add.at(forces, v2_indices, force_vectors)
    
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


def calculate_energy_vectorized(vertices_2d: NDArray[np.float64], edges: NDArray[np.int64], 
                             rest_lengths: dict, spring_constant: float) -> float:
    """
    Vectorized implementation of energy calculation for better performance.
    
    Args:
        vertices_2d: 2D vertex positions
        edges: Edge indices
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for energy calculation
        
    Returns:
        float: Total energy in the spring-mass system
    """
    # Extract edge endpoints
    v1_indices = edges[:, 0]
    v2_indices = edges[:, 1]
    
    # Get positions of edge endpoints
    p1_positions = vertices_2d[v1_indices]
    p2_positions = vertices_2d[v2_indices]
    
    # Calculate current lengths
    current_lengths = np.linalg.norm(p2_positions - p1_positions, axis=1)
    
    # Get rest lengths for edges (convert tuple keys to indices)
    rest_length_values = np.array([rest_lengths[(v1, v2)] for v1, v2 in edges])
    
    # Calculate energy components for each edge
    energy_components = 0.5 * spring_constant * (current_lengths - rest_length_values) ** 2
    
    # Sum all energy components and double as per the paper
    total_energy = 2.0 * np.sum(energy_components)
    
    return total_energy


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