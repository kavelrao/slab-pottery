"""
Core algorithms for mesh flattening.
"""

import numpy as np
from numpy.typing import NDArray
import trimesh
from tqdm import tqdm, trange

from .geometry import (
    build_face_adjacency,
    get_mesh_subset,
    place_initial_triangle
)
from .metrics import (
    calculate_area_error,
    calculate_shape_error
)
from .physics import (
    calculate_energy,
    calculate_forces,
    calculate_masses,
    calculate_penalty_displacements,
    calculate_rest_lengths,
    calculate_rho
)


def initial_flattening(
    mesh: trimesh.Trimesh,
    spring_constant: float,
    area_density: float,
    dt: float,
    permissible_area_error: float,
    permissible_shape_error: float,
    permissible_energy_variation: float,
):
    """
    Perform initial triangle flattening to get a starting 2D layout.
    Implements constrained triangle flattening method from the paper.
    """
    vertices_3d = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    vertices_2d = np.zeros((len(vertices_3d), 2))
    flattened_faces_indices = set()
    flattened_vertices_indices = set()
    
    # Start with the first face and place it in 2D
    first_face_idx = 0
    f_indices = faces[first_face_idx]
    
    # Place first triangle
    initial_positions = place_initial_triangle(vertices_3d, f_indices)
    vertices_2d[f_indices[0]] = initial_positions[0]
    vertices_2d[f_indices[1]] = initial_positions[1]
    vertices_2d[f_indices[2]] = initial_positions[2]
    
    # Create adjacency list for faces based on shared edges
    face_adjacency = build_face_adjacency(faces)
    
    # BFS queue starting with first face
    flatten_neighbors_queue = [first_face_idx]
    flattened_faces_indices.add(first_face_idx)
    flattened_vertices_indices.update(set(f_indices))
    
    pbar = tqdm(total=len(faces) - 1, desc="Initial flattening")
    while flatten_neighbors_queue:
        current_face_idx = flatten_neighbors_queue.pop(0)
        
        for adjacent_face_idx in face_adjacency[current_face_idx]:
            if adjacent_face_idx not in flattened_faces_indices:
                face_indices = faces[adjacent_face_idx]
                prev_face_indices = faces[current_face_idx]
                
                # Find shared edge vertices and unflattened vertex
                shared_vertices = set(face_indices).intersection(set(prev_face_indices))
                assert len(shared_vertices) == 2, (set(face_indices), flattened_vertices_indices)
                
                unflattened_vert_idx = list(set(face_indices) - shared_vertices)[0]
                shared_edge_verts = list(shared_vertices)
                
                # Position the new vertex using the shared edge
                p1_2d = vertices_2d[shared_edge_verts[0]]
                p2_2d = vertices_2d[shared_edge_verts[1]]
                p1_3d = vertices_3d[shared_edge_verts[0]]
                p2_3d = vertices_3d[shared_edge_verts[1]]
                p3_3d = vertices_3d[unflattened_vert_idx]
                
                l12_2d = np.linalg.norm(p2_2d - p1_2d)
                l13 = np.linalg.norm(p3_3d - p1_3d)
                l23 = np.linalg.norm(p3_3d - p2_3d)
                
                if l13 < 1e-9 or l12_2d < 1e-9:
                    print(f"Warning: Very small edge lengths detected: l13={l13}, l12={l12_2d}")
                
                # Calculate angle using cosine law
                cos_theta = (l13**2 + l12_2d**2 - l23**2) / (2 * l13 * l12_2d)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                
                # Calculate direction vector of existing edge
                edge_dir = (p2_2d - p1_2d) / l12_2d
                
                # Calculate normal of triangle being added in 3D
                v1_3d = p2_3d - p1_3d
                v2_3d = p3_3d - p1_3d
                normal_new = np.cross(v1_3d, v2_3d)
                normal_new = normal_new / np.linalg.norm(normal_new)
                
                # Calculate normal of the reference triangle (already flattened)
                ref_face = None
                for f in faces:
                    if (shared_edge_verts[0] in f and 
                        shared_edge_verts[1] in f and 
                        unflattened_vert_idx not in f):
                        ref_face = f
                        break
                assert ref_face is not None, "Couldn't find an adjacent already flattened face"
                
                # Calculate normal of reference triangle
                ref_verts = vertices_3d[ref_face]
                v1_ref = ref_verts[1] - ref_verts[0]
                v2_ref = ref_verts[2] - ref_verts[0]
                normal_ref = np.cross(v1_ref, v2_ref)
                normal_ref = normal_ref / np.linalg.norm(normal_ref)
                
                # Calculate dot product between normals to determine orientation
                dot_product = np.dot(normal_new, normal_ref)
                
                # Rotation based on orientation
                if dot_product > 0:  # If normals point in same direction, rotate counterclockwise
                    rot_matrix = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                    ])
                else:  # Rotate clockwise
                    rot_matrix = np.array([
                        [np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]
                    ])
                
                p3_dir = rot_matrix @ edge_dir
                p3_2d = p1_2d + l13 * p3_dir
                
                # Update the vertex position
                vertices_2d[unflattened_vert_idx] = p3_2d
                
                # Update BFS bookkeeping
                flatten_neighbors_queue.append(adjacent_face_idx)
                flattened_faces_indices.add(adjacent_face_idx)
                flattened_vertices_indices.update(set(face_indices))
                
                # Get the spring mass system for the subset of already flattened vertices
                flattened_vertices_subset = np.array(list(flattened_vertices_indices))
                flattened_vertices_2d = vertices_2d[flattened_vertices_subset]
                
                flattened_vertices_3d, flattened_edges, flattened_faces = get_mesh_subset(
                    vertices_3d, mesh.edges_unique, mesh.faces, flattened_vertices_subset
                )
                
                # Apply energy release with N=50 steps (currently disabled)
                if False:  # ENABLE_ENERGY_RELEASE:
                    vertices_2d[flattened_vertices_subset] = energy_release(
                        flattened_vertices_3d,
                        flattened_edges,
                        flattened_faces,
                        flattened_vertices_2d,
                        spring_constant,
                        area_density,
                        dt,
                        50,  # Iteration steps number N=50 as per paper
                        permissible_area_error,
                        permissible_shape_error,
                        permissible_energy_variation,
                        verbose=True
                    )
                pbar.update(1)
    
    pbar.close()
    assert len(flattened_faces_indices) == len(faces)
    return vertices_2d


def energy_release(
    vertices_3d: NDArray[np.float64],
    edges: NDArray[np.float64],
    faces: NDArray[np.float64],
    vertices_2d_initial: NDArray[np.float64],
    spring_constant: float,
    area_density: float,
    dt: float,
    max_iterations: int,
    permissible_area_error: float,
    permissible_shape_error: float,
    permissible_energy_variation: float,
    verbose: bool = False
):
    """
    Perform energy release phase using spring-mass model and Euler's method.
    """
    vertices_3d = vertices_3d.copy()
    faces = faces.copy()
    edges = edges.copy()
    vertices_2d = vertices_2d_initial.copy()
    
    # Calculate initial edge lengths and masses
    rest_lengths = calculate_rest_lengths(vertices_3d, edges)
    masses = calculate_masses(vertices_3d, faces, area_density)
    
    velocities = np.zeros_like(vertices_2d)
    accelerations = np.zeros_like(vertices_2d)
    
    # Calculate initial energy
    prev_energy = calculate_energy(vertices_2d, edges, rest_lengths, spring_constant)
    
    # Determine progress bar type based on iteration count
    range_func = range if max_iterations < 50 else lambda x: trange(x, desc="Energy Release")
    
    for iteration in range_func(max_iterations):
        # Calculate forces based on spring-mass model
        forces = calculate_forces(vertices_2d, edges, rest_lengths, spring_constant)
        
        # Euler's method integration
        accelerations = forces / masses[:, np.newaxis]  # a = F/m
        velocities += accelerations * dt
        vertices_2d += velocities * dt + 0.5 * accelerations * dt**2
        
        # Calculate penalty displacements and apply them
        penalty_displacements = calculate_penalty_displacements(vertices_3d, faces, vertices_2d)
        vertices_2d += penalty_displacements
        
        # Calculate energy for monitoring convergence
        current_energy = calculate_energy(vertices_2d, edges, rest_lengths, spring_constant)
        
        # Calculate error metrics
        area_error = calculate_area_error(vertices_3d, vertices_2d, faces)
        shape_error = calculate_shape_error(vertices_3d, vertices_2d, edges)
        energy_variation_percentage = (
            abs((current_energy - prev_energy) / prev_energy)
            if prev_energy != float("inf")
            else 1.0
        )
        
        # Check termination conditions
        if ((area_error < permissible_area_error and shape_error < permissible_shape_error) or 
            energy_variation_percentage < permissible_energy_variation):
            if verbose:
                print(f"Termination at iteration: {iteration}, Area Error: {area_error:.4f}, "
                      f"Shape Error: {shape_error:.4f}, Energy Variation: {energy_variation_percentage:.4f}, "
                      f"Energy: {current_energy}")
            break
        
        if iteration > max_iterations - 2:  # Max iterations reached
            if verbose:
                print(f"Max iteration termination at iteration: {iteration}, Area Error: {area_error:.4f}, "
                      f"Shape Error: {shape_error:.4f}, Energy Variation: {energy_variation_percentage:.4f}, "
                      f"Energy: {current_energy}")
            break
        
        prev_energy = current_energy
    
    return vertices_2d


def surface_flattening_spring_mass(
    mesh: trimesh.Trimesh,
    spring_constant: float = 0.5,
    dt: float = 0.01,
    max_iterations: int = 1000,
    permissible_area_error: float = 0.01,
    permissible_shape_error: float = 0.01,
    permissible_energy_variation: float = 0.0005,
    enable_energy_release: bool = True,
):
    """
    Implement a spring-mass surface flattening algorithm based on the paper
    "SURFACE FLATTENING BASED ON ENERGY MODEL".

    Args:
        mesh (trimesh.Mesh): Input 3D mesh to be flattened.
        spring_constant (float): Spring constant 'C' in the energy model.
        dt (float): Time step for Euler's method.
        max_iterations (int): Maximum iterations for energy release phase.
        permissible_area_error (float): Permissible relative area difference for termination.
        permissible_shape_error (float): Permissible relative edge length difference for termination.
        permissible_energy_variation (float): Permissible percentage variation of energy for termination.
        enable_energy_release (bool): Whether to run the energy release phase.

    Returns:
        numpy.ndarray: 2D vertex positions of the flattened mesh.
    """
    # 1. Calculate optimal area density (rho)
    area_density = calculate_rho(mesh)
    
    # 2. Initial Flattening (Triangle Flattening - Constrained)
    vertices_2d_initial = initial_flattening(
        mesh,
        spring_constant,
        area_density,
        dt,
        permissible_area_error,
        permissible_shape_error,
        permissible_energy_variation,
    )
    vertices_2d = vertices_2d_initial.copy()
    
    # 3. Planar Mesh Deformation (Energy Release) - if enabled
    if enable_energy_release:
        vertices_2d = energy_release(
            mesh.vertices,
            mesh.edges_unique,
            mesh.faces,
            vertices_2d,
            spring_constant,
            area_density,
            dt,
            max_iterations,
            permissible_area_error,
            permissible_shape_error,
            permissible_energy_variation,
            verbose=True
        )
    
    return vertices_2d