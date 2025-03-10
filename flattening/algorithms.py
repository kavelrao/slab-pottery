"""
Core algorithms for mesh flattening.
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import trimesh
from tqdm import tqdm, trange
from itertools import combinations
import networkx as nx

from .geometry import (
    build_face_adjacency,
    get_mesh_subset,
    place_initial_triangle
)
from .metrics import (
    calculate_area_error,
    calculate_shape_error,
    calculate_area_error_vectorized,
    calculate_shape_error_vectorized
)
from .physics import (
    calculate_energy_vectorized,
    calculate_forces_vectorized,
    calculate_masses,
    calculate_penalty_displacements,
    calculate_penalty_displacements_vectorized,
    calculate_rest_lengths,
    calculate_rho
)
from .geometry import precompute_all_opposite_edges


def initial_flattening(
    mesh: trimesh.Trimesh,
    spring_constant: float,
    area_density: float,
    dt: float,
    permissible_area_error: float,
    permissible_shape_error: float,
    permissible_energy_variation: float,
    penalty_coefficient: float,
    enable_energy_release: bool,
    energy_release_iterations: int,
    rest_lengths: dict = None,
    all_opposite_edges: list = None,
):
    """
    Perform initial triangle flattening to get a starting 2D layout.
    Implements constrained triangle flattening method from the paper.
    
    Args:
        mesh: Input 3D mesh
        spring_constant: Spring constant for force calculation
        area_density: Area density for mass calculation
        dt: Time step for integration
        permissible_area_error: Error threshold for area
        permissible_shape_error: Error threshold for shape
        permissible_energy_variation: Error threshold for energy variation
        penalty_coefficient: The coefficient of the penalty function displacement for energy release
        enable_energy_release: Whether to apply energy release during the initial flattening
        energy_release_iterations: Number of iterations between each energy release step
        rest_lengths: Optional precomputed rest lengths dictionary
        all_opposite_edges: Optional precomputed opposite edges for all vertices
        
    Returns:
        NDArray[np.float64]: Initial 2D vertex positions
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
                
                # Get mesh subset with precomputed values
                subset_result = get_mesh_subset(
                    vertices_3d, mesh.edges_unique, mesh.faces, 
                    flattened_vertices_subset, rest_lengths, all_opposite_edges
                )
                flattened_vertices_3d, flattened_edges, flattened_faces, flattened_rest_lengths, flattened_opposite_edges = subset_result

                # Apply energy release with N=50 steps
                if enable_energy_release and len(flattened_faces_indices) % energy_release_iterations == 0:
                    vertices_2d[flattened_vertices_subset], _, _, _, _, _, _ = energy_release(
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
                        penalty_coefficient,
                        rest_lengths=flattened_rest_lengths,
                        all_opposite_edges=flattened_opposite_edges,
                        verbose=False
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
    penalty_coefficient: float,
    rest_lengths: dict = None,
    all_opposite_edges: list = None,
    verbose: bool = False
):
    """
    Perform energy release phase using spring-mass model and Euler's method.
    
    Args:
        vertices_3d: 3D vertex positions
        edges: Edge indices
        faces: Face indices
        vertices_2d_initial: Initial 2D vertex positions
        spring_constant: Spring constant for force calculation
        area_density: Area density for mass calculation
        dt: Time step for integration
        max_iterations: Maximum number of iterations
        permissible_area_error: Error threshold for area
        permissible_shape_error: Error threshold for shape
        permissible_energy_variation: Error threshold for energy variation
        penalty_coefficient: The coefficient of the penalty function displacement for energy release
        rest_lengths: Optional precomputed rest lengths dictionary
        all_opposite_edges: Optional precomputed opposite edges for all vertices
        verbose: Whether to print progress information
        
    Returns:
        NDArray[np.float64]: Optimized 2D vertex positions, bunch of lists for debugging
    """
    vertices_3d = vertices_3d.copy()
    faces = faces.copy()
    edges = edges.copy()
    vertices_2d = vertices_2d_initial.copy()
    
    # Calculate or use provided rest lengths
    if rest_lengths is None:
        rest_lengths = calculate_rest_lengths(vertices_3d, edges)
    
    # Calculate masses
    masses = calculate_masses(vertices_3d, faces, area_density)
    
    velocities = np.zeros_like(vertices_2d)
    accelerations = np.zeros_like(vertices_2d)
    
    # Calculate initial energy
    prev_energy = calculate_energy_vectorized(vertices_2d, edges, rest_lengths, spring_constant)
    
    # Use provided opposite edges or compute them if not provided
    if all_opposite_edges is None:
        all_opposite_edges = precompute_all_opposite_edges(vertices_3d, faces)
    
    # Determine whether to use progress bar based on iteration count
    range_func = range if max_iterations < 100 else lambda x: trange(x, desc="Energy Release")

    # Track errors and energy
    area_errors = [calculate_area_error_vectorized(vertices_3d, vertices_2d, faces)]
    shape_errors = [calculate_shape_error_vectorized(vertices_3d, vertices_2d, edges)]
    max_forces = [np.linalg.norm(calculate_forces_vectorized(vertices_2d, edges, rest_lengths, spring_constant), axis=1).max()]
    energies = [prev_energy]
    max_displacements = [0]
    max_penalty_displacements = [0]
    
    for iteration in range_func(max_iterations):
        # Calculate forces based on spring-mass model
        forces = calculate_forces_vectorized(vertices_2d, edges, rest_lengths, spring_constant)
        
        # Euler's method integration
        accelerations = forces / masses[:, np.newaxis]  # a = F/m
        velocities += accelerations * dt
        displacements = velocities * dt + 0.5 * accelerations * dt**2
        vertices_2d += displacements
        
        # Calculate penalty displacements and apply them - using vectorized version
        penalty_displacements = dt * calculate_penalty_displacements_vectorized(
            vertices_3d, faces, vertices_2d, all_opposite_edges=all_opposite_edges, penalty_coefficient=penalty_coefficient
        )
        vertices_2d += penalty_displacements
        
        # Calculate energy for monitoring convergence
        current_energy = calculate_energy_vectorized(vertices_2d, edges, rest_lengths, spring_constant)
        
        # Calculate error metrics - using vectorized versions
        area_error = calculate_area_error_vectorized(vertices_3d, vertices_2d, faces)
        shape_error = calculate_shape_error_vectorized(vertices_3d, vertices_2d, edges)
        energy_variation_percentage = (
            abs((current_energy - prev_energy) / prev_energy)
            if prev_energy != float("inf")
            else 1.0
        )

        area_errors.append(area_error)
        shape_errors.append(shape_error)
        max_forces.append(np.linalg.norm(forces, axis=1).max())
        energies.append(current_energy)
        max_displacements.append(np.linalg.norm(displacements, axis=1).max())
        max_penalty_displacements.append(np.linalg.norm(penalty_displacements, axis=1).max())
        
        # Check termination conditions
        # TODO: Maybe early exit if the energy starts increasing?
        if ((area_error < permissible_area_error or shape_error < permissible_shape_error) and
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
    
    return vertices_2d, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements

def surface_flattening_spring_mass(
    mesh: trimesh.Trimesh,
    spring_constant: float = 0.5,
    dt: float = 0.001,
    max_iterations: int = 1000,
    permissible_area_error: float = 0.01,
    permissible_shape_error: float = 0.01,
    permissible_energy_variation: float = 0.0005,
    penalty_coefficient: float = 1.0,
    enable_energy_release_in_flatten: bool = True,
    energy_release_iterations: int = 1,
    enable_energy_release_phase: bool = True,
    area_density: np.float64 | None = None,
    vertices_2d_initial: NDArray[np.float64] | None = None,
    object_name: str | None = None,
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
        penalty_coefficient: The coefficient of the penalty function displacement for energy release
        enable_energy_release_in_flatten: Whether to apply energy release during the initial flattening
        energy_release_iterations: Number of iterations between each energy release step in initial flattening
        area_density (float): If provided, skips the area density calculation step and uses this value.
        vertices_2d_initial (NDArray): If provided, skips the initial flattening step and uses this array for the 2d vertices.

    Returns:
        numpy.ndarray: 2D vertex positions of the flattened mesh after energy release
        numpy.ndarray: 2D vertex positions of the initial flattening mesh, without energy release
        bunch of lists for debugging.
    """
    # 1. Calculate optimal area density (rho)
    if area_density is None:
        area_density = calculate_rho(mesh)
        if object_name:
            np.save(Path(__file__).parent.parent / "files" / (object_name + "_areadensity.npy"), area_density)
    else:
        print(f"Skipping area density computation, using saved value {area_density}.")
    
    # Precompute values that don't change during the process
    rest_lengths = calculate_rest_lengths(mesh.vertices, mesh.edges_unique)
    all_opposite_edges = precompute_all_opposite_edges(mesh.vertices, mesh.faces)

    # 2. Initial Flattening (Triangle Flattening - Constrained)
    if vertices_2d_initial is None:
        vertices_2d_initial = initial_flattening(
            mesh,
            spring_constant,
            area_density,
            dt,
            permissible_area_error,
            permissible_shape_error,
            permissible_energy_variation,
            penalty_coefficient=penalty_coefficient,
            enable_energy_release=enable_energy_release_in_flatten,
            energy_release_iterations=energy_release_iterations,
            rest_lengths=rest_lengths,
            all_opposite_edges=all_opposite_edges,
        )
        if object_name:
            np.save(Path(__file__).parent.parent / "files" / (object_name + "_init2d.npy"), vertices_2d_initial)
    else:
        print("Skipping initial flattening, using provided vertex positions.")
    vertices_2d = vertices_2d_initial.copy()
    
    # 3. Planar Mesh Deformation (Energy Release) - if enabled
    area_errors, shape_errors, energies = [], [], []
    if enable_energy_release_phase:
        vertices_2d, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = energy_release(
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
            penalty_coefficient=penalty_coefficient,
            rest_lengths=rest_lengths,
            all_opposite_edges=all_opposite_edges,
            verbose=True
        )
    
    return vertices_2d, vertices_2d_initial, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements

def gen_elastic_deformation_energy_distribution(faces, node_energies, mesh):
    num_nodes = len(mesh.vertices)
    energy_graph = mesh.vertex_adjacency_graph.copy()
    interpolated_energies = np.zeros(len(faces), dtype=np.float64)
    for i, (v0, v1, v2) in enumerate(faces):
        # add energy gradients for current face edges
        # for edge in list(combinations([v0, v1, v2])):
        #     n0, n1 = edge
        #     if not energy_graph.has_edge(n0, n1):
        #         energy_graph[n0][n1]['diff'] = node_energies[n1] - node_energies[n0]
        # interpolate face vertex energies for new node energy
        interpolated_energies[i] = np.sum([node_energies[v0], node_energies[v1], node_energies[v2]]) / 3
        interpolated_node_num = num_nodes+i

        #add energy gradients for the new edges from dividing the face
        for n0, n1 in [(interpolated_node_num, v0), (interpolated_node_num, v1), (interpolated_node_num, v2)]:
            energy_graph.add_edge(n0, n1)
        
    return interpolated_energies, energy_graph

def update_mesh_with_path(mesh, path, graph, all_sampled_positions):
    graph_cut = graph.copy()
    faces_cut = mesh.faces.copy()
    faces_cut_sorted = np.sort(faces_cut, axis=1)
    vertices_cut = mesh.vertices.copy()
    new_vertices = []
    new_faces = []
    new_node_num = len(mesh.vertices)
    updated_path = []
    # check each node to see if it needs to be added to mesh
    for node in path:
        if node >= len(mesh.vertices):
            # get all the neighbors (they will always be the edges of an existing triangle)
            neighbors = np.sort(np.array(list(graph.neighbors(node)))) 
             # find the corresponding face:
            face_to_cut_idx = np.where((faces_cut_sorted == neighbors).all(axis=1))[0]
            cut_face = mesh.faces[face_to_cut_idx].flatten()
            new_vertices.append(all_sampled_positions[node]) # add the middle vertex
            # subdivide into 3 faces, each face has 2 original vertices and other is the new node
            f1 = cut_face.copy()
            f1[0] = new_node_num 
            f2 = cut_face.copy()
            f2[1] = new_node_num
            f3 = cut_face.copy()
            f3[2] = new_node_num
            # replace the old face
            faces_cut[face_to_cut_idx] = f1
            # add other two faces as new face
            new_faces.extend([f2, f3])
            # update path to use new node indexing
            updated_path.append(new_node_num)
            print(f'adding node {new_node_num}')
            new_node_num += 1
            print(f'face 1: {f1}\nface2: {f2}\nface 3: {f3}')
            og_vertices = mesh.vertices[cut_face]
            print(f'original vertices: {og_vertices}')
            print(f'new vertex: {all_sampled_positions[node]}')
            print(f'does python believe this new node is already in the triangles?: {(og_vertices == all_sampled_positions[node]).all(axis=1).any()}')
        else:
            # node index stays the same
            updated_path.append(node)
    # create mesh with new vertices and face
    print(f'new vertices list {new_vertices} is {len(new_vertices)} long')
    print(mesh.vertices)
    unique_vertices = np.unique(mesh.vertices, axis=0)
    print(f"Unique vertices of original : {len(unique_vertices)}")
    vertices_cut = np.vstack((mesh.vertices, np.array(new_vertices)))
    # Get unique vertices and their first occurrences
    unique_vertices, indices, inverse_indices, counts = np.unique(
        vertices_cut, axis=0, return_index=True, return_inverse=True, return_counts=True
    )

    # Find duplicates (where counts > 1)
    duplicate_indices = np.where(counts > 1)[0]

    # Print duplicate vertex positions and their indices in the original array
    for dup in duplicate_indices:
        original_indices = np.where(inverse_indices == dup)[0]
        print(f"Vertex {unique_vertices[dup]} appears at indices {original_indices}")
    print(f"Unique vertices of new : {len(unique_vertices)}")
    print(len(vertices_cut))
    faces_cut = np.vstack((mesh.faces, np.array(new_faces)))
    new_mesh = trimesh.Trimesh(vertices_cut, faces_cut)
    print(f'new mesh num of vertices: {len(new_mesh.vertices)}')
    return new_mesh, updated_path

            

            


            

