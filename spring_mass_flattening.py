import trimesh
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm, trange


STL_FILE = 'files/Partial_Cylinder_Shell.stl'
ENABLE_ENERGY_RELEASE = True


# ===== Geometry Utilities =====

def calculate_face_area(face_verts: NDArray[np.float64]) -> float:
    """Calculate the area of a 3D triangle."""
    p1_3d, p2_3d, p3_3d = face_verts
    v1 = p2_3d - p1_3d
    v2 = p3_3d - p1_3d
    return 0.5 * np.linalg.norm(np.cross(v1, v2))


def point_to_segment_distance_2d(point_p, segment_p1, segment_p2):
    """
    Calculate the shortest distance and vector from a point to a line segment in 2D.
    
    Returns:
        tuple: (shortest distance, vector from point_p to closest point on segment)
    """
    l2 = np.sum((segment_p1 - segment_p2)**2)
    if l2 == 0.0:
        distance = np.linalg.norm(point_p - segment_p1)
        vector_to_segment = segment_p1 - point_p
        return distance, vector_to_segment
    
    t = max(0, min(1, np.dot(point_p - segment_p1, segment_p2 - segment_p1) / l2))
    projection = segment_p1 + t * (segment_p2 - segment_p1)
    distance = np.linalg.norm(point_p - projection)
    vector_to_segment = projection - point_p
    return distance, vector_to_segment


def point_to_segment_distance_3d(point_q, segment_q1, segment_q2):
    """
    Calculate the shortest distance and vector from a point to a line segment in 3D.
    
    Returns:
        tuple: (shortest distance, vector from point_q to closest point on segment)
    """
    l2 = np.sum((segment_q1 - segment_q2)**2)
    if l2 == 0.0:
        distance = np.linalg.norm(point_q - segment_q1)
        vector_to_segment = segment_q1 - point_q
        return distance, vector_to_segment
    
    t = max(0, min(1, np.dot(point_q - segment_q1, segment_q2 - segment_q1) / l2))
    projection = segment_q1 + t * (segment_q2 - segment_q1)
    distance = np.linalg.norm(point_q - projection)
    vector_to_segment = projection - point_q
    return distance, vector_to_segment


def get_opposite_edges(vertex_index, faces):
    """
    Get the "opposite edges" for a given vertex index based on the faces it belongs to.
    Opposite edges are edges of the faces that do *not* include the vertex.
    """
    opposite_edges = []
    for face in faces:
        if vertex_index in face:
            face_verts = list(face)
            face_verts.remove(vertex_index)
            opposite_edges.append(tuple(sorted(face_verts)))
    return list(set(opposite_edges))


def get_mesh_subset(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    faces: NDArray[np.int64],
    vertex_indices_subset: NDArray[np.int64]
):
    """
    Extract a subset of vertices from a mesh and return re-indexed vertices, edges, and faces.
    """
    subset_vertices = vertices[vertex_indices_subset]
    
    original_to_subset_index_map = {
        original_index: new_index
        for new_index, original_index in enumerate(vertex_indices_subset)
    }
    
    subset_faces = []
    for face in faces:
        if all(v_idx in vertex_indices_subset for v_idx in face):
            reindexed_face = [original_to_subset_index_map[v_idx] for v_idx in face]
            subset_faces.append(reindexed_face)
    subset_faces = np.array(subset_faces)
    
    subset_edges = []
    for edge in edges:
        if all(v_idx in vertex_indices_subset for v_idx in edge):
            reindexed_edge = [original_to_subset_index_map[v_idx] for v_idx in edge]
            subset_edges.append(reindexed_edge)
    subset_edges = np.array(subset_edges)
    
    return subset_vertices, subset_edges, subset_faces


def calculate_vertex_areas(vertices: NDArray[np.float64], faces: NDArray[np.int64]) -> NDArray[np.float64]:
    """Calculate the area associated with each vertex (1/3 of connected face areas)."""
    num_vertices = len(vertices)
    vertex_areas = np.zeros(num_vertices)
    
    for i in range(num_vertices):
        connected_faces_indices = np.where(np.any(faces == i, axis=1))[0]
        vertex_area = 0.0
        for face_index in connected_faces_indices:
            face_3d_verts = vertices[faces[face_index]]
            face_area = calculate_face_area(face_3d_verts)
            vertex_area += face_area
        vertex_areas[i] = vertex_area / 3.0  # Divide by 3 for even distribution
    
    return vertex_areas


def build_face_adjacency(faces: NDArray[np.int64]) -> list:
    """Build adjacency list for faces based on shared edges."""
    face_adjacency = [[] for _ in range(len(faces))]
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            shared_vertices = set(faces[i]).intersection(set(faces[j]))
            if len(shared_vertices) == 2:  # They share an edge
                face_adjacency[i].append(j)
                face_adjacency[j].append(i)
    return face_adjacency


# ===== Error Metrics =====

def calculate_area_error(vertices_3d, vertices_2d, faces):
    """
    Calculate the Area Accuracy (Es) as described in the paper.
    """
    total_area_diff = 0.0
    total_original_area = 0.0

    for face_indices in faces:
        face_3d_verts = vertices_3d[face_indices]
        face_original_area = calculate_face_area(face_3d_verts)
        
        face_2d_verts = vertices_2d[face_indices]
        p1_2d, p2_2d, p3_2d = face_2d_verts
        face_flattened_area = 0.5 * abs(np.cross(p2_2d - p1_2d, p3_2d - p1_2d))
        
        area_diff = abs(face_original_area - face_flattened_area)
        total_area_diff += area_diff
        total_original_area += face_original_area
    
    if total_original_area > 0:
        relative_area_error = total_area_diff / total_original_area
    else:
        print("Warning: total area is 0")
        relative_area_error = 0.0
    
    return relative_area_error


def calculate_shape_error(vertices_3d, vertices_2d, edges):
    """
    Calculate the Shape Accuracy (Ec) as described in the paper.
    """
    total_length_diff = 0.0
    total_original_length = 0.0
    
    for edge_indices in edges:
        v1_idx, v2_idx = edge_indices
        
        p1_3d = vertices_3d[v1_idx]
        p2_3d = vertices_3d[v2_idx]
        original_length = np.linalg.norm(p2_3d - p1_3d)
        
        p1_2d = vertices_2d[v1_idx]
        p2_2d = vertices_2d[v2_idx]
        flattened_length = np.linalg.norm(p2_2d - p1_2d)
        
        length_diff = abs(original_length - flattened_length)
        total_length_diff += length_diff
        total_original_length += original_length
    
    if total_original_length > 0:
        relative_shape_error = total_length_diff / total_original_length
    else:
        print("Warning: total edge length is 0")
        relative_shape_error = 0.0
    
    return relative_shape_error


# ===== Physics Simulation =====

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


# ===== Rho Calculation =====

def calculate_rho(mesh: trimesh.Trimesh) -> float:
    """
    Calculate the optimal area density (rho) value for mass calculation.
    
    This ensures the minimum mass is normalized to 1.0, as described in the paper.
    """
    rho_current = 1.0  # Initial guess for rho
    tolerance_rho = 1e-6
    max_iterations_rho = 100
    rho_values_history = []
    num_vertices = len(mesh.vertices)
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


# ===== Initial Flattening =====

def place_initial_triangle(vertices_3d: NDArray[np.float64], face_indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """Place the first triangle in 2D space."""
    p1_3d = vertices_3d[face_indices[0]]
    p2_3d = vertices_3d[face_indices[1]]
    p3_3d = vertices_3d[face_indices[2]]
    
    l12 = np.linalg.norm(p2_3d - p1_3d)
    l13 = np.linalg.norm(p3_3d - p1_3d)
    l23 = np.linalg.norm(p3_3d - p2_3d)
    
    p1_2d = np.array([0.0, 0.0])
    p2_2d = np.array([l12, 0.0])
    
    # Place third vertex using cosine law
    cos_theta = (l13**2 + l12**2 - l23**2) / (2 * l13 * l12)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    p3_2d = p1_2d + l13 * np.array([np.cos(theta), np.sin(theta)])
    
    return np.array([p1_2d, p2_2d, p3_2d])


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
    if ENABLE_ENERGY_RELEASE:
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


if __name__ == "__main__":
    # Load mesh from file
    mesh = trimesh.load(STL_FILE)
    
    # Perform flattening
    flattened_vertices_2d = surface_flattening_spring_mass(mesh)
    
    # Visualization (requires matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with two subplots - one for 3D, one for 2D
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                      triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=0.7)
    ax3d.set_title("Original 3D Surface")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    
    # 2D plot
    ax2d = fig.add_subplot(122)
    ax2d.set_aspect("equal")  # Ensure aspect ratio is 1:1
    
    # Create PolyCollection for faces in 2D
    face_verts_2d = flattened_vertices_2d[mesh.faces]
    poly_collection = PolyCollection(
        face_verts_2d, facecolors="skyblue", edgecolors="black", linewidths=0.5
    )
    ax2d.add_collection(poly_collection)
    
    # Set plot limits to encompass the flattened mesh
    min_coords = np.min(flattened_vertices_2d, axis=0)
    max_coords = np.max(flattened_vertices_2d, axis=0)
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    padding_x = range_x * 0.1  # 10% padding
    padding_y = range_y * 0.1
    
    ax2d.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
    ax2d.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)
    ax2d.set_title("Flattened Surface")
    
    plt.tight_layout()
    plt.show()
