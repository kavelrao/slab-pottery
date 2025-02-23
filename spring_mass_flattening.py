import trimesh
import numpy as np
from numpy.typing import NDArray


ENABLE_ENERGY_RELEASE = True


def surface_flattening_spring_mass(
    mesh: trimesh.Trimesh,
    spring_constant: float=0.5,
    dt: float=0.01,
    max_iterations: int=1000,
    permissible_area_error: float=0.01,
    permissible_shape_error: float=0.01,
    permissible_energy_variation: float=0.0005,
):
    """
    Implements a spring-mass surface flattening algorithm based on the paper
    "SURFACE FLATTENING BASED ON ENERGY MODEL".

    Args:
        mesh (trimesh.Mesh): Input 3D mesh to be flattened.
        spring_constant (float): Spring constant 'C' in the energy model.
        area_density (float): Area density 'rho' in the mass calculation.
        dt (float): Time step for Euler's method.
        max_iterations (int): Maximum iterations for energy release phase.
        permissible_area_error (float): Permissible relative area difference for termination.
        permissible_shape_error (float): Permissible relative edge length difference for termination.
        permissible_energy_variation (float): Permissible percentage variation of energy for termination.

    Returns:
        numpy.ndarray: 2D vertex positions of the flattened mesh.
    """

    # --- Iterative rho calculation with RELAXATION ---
    rho_current = 1.0  # Initial guess for rho
    tolerance_rho = 1e-6
    max_iterations_rho = 100
    rho_values_history = []  # To track rho values for debugging
    num_vertices = len(mesh.vertices)
    rho_relaxation_factor = 0.1

    for rho_iteration in range(max_iterations_rho):
        masses_rho_iter = np.zeros(num_vertices)  # Masses for this rho iteration

        for i in range(num_vertices):
            connected_faces_indices = np.where(np.any(mesh.faces == i, axis=1))[0]
            vertex_area = 0.0
            for face_index in connected_faces_indices:
                face_3d_verts = mesh.vertices[mesh.faces[face_index]]
                p1_3d, p2_3d, p3_3d = face_3d_verts
                v1 = p2_3d - p1_3d
                v2 = p3_3d - p1_3d
                face_area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                vertex_area += face_area
            masses_rho_iter[i] = vertex_area * (rho_current / 3)

        min_m = np.min(masses_rho_iter)
        if min_m < 1e-9:
            rho_new = rho_current
        else:
            rho_new = 1.0 / min_m

        # --- RELAXATION UPDATE RULE ---
        rho_current = (1 - rho_relaxation_factor) * rho_current + rho_relaxation_factor * rho_new

        rho_values_history.append(rho_current) # Store rho value

        relative_change_rho = abs((rho_new - rho_current) / rho_current) if rho_current != 0 else rho_new
        if relative_change_rho < tolerance_rho:
            print(f"Rho converged in {rho_iteration + 1} iterations, rho = {rho_current:.6f}")
            break
        # rho_current = rho_new  # REMOVED - replaced by relaxed update

    else:  # else block runs if loop completes without break (max_iterations reached)
        print(
            f"Warning: Rho iteration reached max iterations ({max_iterations_rho}), convergence not guaranteed, using rho = {rho_current:.6f}."
        )
        print("Rho values history:", rho_values_history) # Print history for debugging

    area_density = rho_current # Use the converged rho as area_density for masses

    # 1. Initial Flattening (Triangle Flattening - Constrained, with Energy Release)
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

    if not ENABLE_ENERGY_RELEASE:
        return vertices_2d

    # 2. Planar Mesh Deformation (Energy Release)
    vertices_2d_flattened = energy_release(
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

    return vertices_2d_flattened


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
    Performs initial triangle flattening to get a starting 2D layout.
    Implements constrained triangle flattening and *includes Energy Release*
    with 50 iterations after each triangle is added, as per the paper's InitFlatten algorithm.
    """
    vertices_3d = mesh.vertices.copy()
    faces = mesh.faces.copy()

    vertices_2d = np.zeros((len(vertices_3d), 2))
    flattened_faces_indices: set[int] = set()  # Keep track of flattened faces
    flattened_vertices_indices: set[int] = set()

    # Start with the first face and arbitrarily place its vertices
    first_face_idx = 0
    f_indices = faces[first_face_idx]
    p1_3d = vertices_3d[f_indices[0]]
    p2_3d = vertices_3d[f_indices[1]]
    p3_3d = vertices_3d[f_indices[2]]

    l12 = np.linalg.norm(p2_3d - p1_3d)
    l13 = np.linalg.norm(p3_3d - p1_3d)
    l23 = np.linalg.norm(p3_3d - p2_3d)

    p1_2d = np.array([0.0, 0.0])
    p2_2d = np.array([l12, 0.0])

    # Calculate p3_2d using circle intersection
    x = (l12**2 - l23**2 + l13**2) / (2 * l12)
    y_sq = l13**2 - x**2
    y = np.sqrt(max(0, y_sq))
    if y_sq < 0:
        print(f"Warning: Imaginary y in circle intersection, projecting p3 onto edge p1-p2 for face {first_face_idx}")
    p3_2d = np.array([x, y])

    vertices_2d[f_indices[0]] = p1_2d
    vertices_2d[f_indices[1]] = p2_2d
    vertices_2d[f_indices[2]] = p3_2d

    # Create adjacency list for faces based on shared edges
    face_adjacency = [[] for _ in range(len(faces))]
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            # Check if faces share an edge
            shared_vertices = set(faces[i]).intersection(set(faces[j]))
            if len(shared_vertices) == 2:  # They share an edge
                face_adjacency[i].append(j)
                face_adjacency[j].append(i)

    # BFS queue starting with first face
    flatten_neighbors_queue = [first_face_idx]  # Start with first face
    flattened_faces_indices.add(first_face_idx)
    flattened_vertices_indices.update(set(f_indices))

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


                p1_2d_constrained = vertices_2d[shared_edge_verts[0]]
                p2_2d_constrained = vertices_2d[shared_edge_verts[1]]
                p1_3d_orig = vertices_3d[shared_edge_verts[0]]
                p2_3d_orig = vertices_3d[shared_edge_verts[1]]
                p3_3d_orig = vertices_3d[unflattened_vert_idx]

                l13 = np.linalg.norm(p3_3d_orig - p1_3d_orig)
                l23 = np.linalg.norm(p3_3d_orig - p2_3d_orig)

                # Place p3_2d using circle intersection relative to constrained points
                l12_2d = np.linalg.norm(
                    p2_2d_constrained - p1_2d_constrained
                )  # Current 2D edge length

                p1_2d_temp = p1_2d_constrained
                p2_2d_temp = p2_2d_constrained

                l12_2d_temp = np.linalg.norm(p2_2d_temp - p1_2d_temp)
                x = (l12_2d_temp**2 - l23**2 + l13**2) / (2 * l12_2d_temp)
                y_sq = l13**2 - x**2
                y = np.sqrt(max(0, y_sq))  # Ensure positive sqrt
                if y_sq < 0:
                    print(f"Warning: Imaginary y in circle intersection, projecting p3 onto edge p1-p2 for face {adjacent_face_idx}")

                # Determine sign of y (up or down). Heuristic to avoid fold-over
                p3_2d_option1 = p1_2d_temp + np.array([x, y])
                p3_2d_option2 = p1_2d_temp + np.array([x, -y])

                # Heuristic to pick option 1 or 2: compare cross product sign
                v_indices_in_common = np.intersect1d(face_indices, prev_face_indices)
                assert len(v_indices_in_common) == 2
                v_ref1_idx = v_indices_in_common[0]
                v_ref2_idx = v_indices_in_common[1]
                vec_ref_2d = vertices_2d[v_ref2_idx] - vertices_2d[v_ref1_idx]

                diff_indices = np.setdiff1d(
                    face_indices, v_indices_in_common
                )  # Calculate set difference

                assert len(diff_indices) == 1
                index_to_use = diff_indices[0]
                # Validate index - although this should now be less likely to be needed if setdiff1d works as expected in normal cases.
                assert index_to_use in face_indices
                reference_cross_prod = np.cross(
                    vec_ref_2d,
                    vertices_2d[index_to_use] - vertices_2d[v_ref1_idx],
                )

                # Heuristic to pick option 1 or 2: compare cross product sign
                vec1 = p2_2d_temp - p1_2d_temp
                vec2_opt1 = p3_2d_option1 - p1_2d_temp
                vec2_opt2 = p3_2d_option2 - p1_2d_temp

                cross_prod_opt1 = np.cross(vec1, vec2_opt1)
                cross_prod_opt2 = np.cross(vec1, vec2_opt2)

                if reference_cross_prod >= 0:  # Reference face is CCW, try to maintain CCW
                    if cross_prod_opt1 >= 0:
                        p3_2d = p3_2d_option1
                    else:
                        p3_2d = p3_2d_option2
                else:  # Reference face is CW, try to maintain CW
                    if cross_prod_opt1 < 0:
                        p3_2d = p3_2d_option1
                    else:
                        p3_2d = p3_2d_option2

                vertices_2d[unflattened_vert_idx] = p3_2d

                # Update BFS bookkeeping
                flatten_neighbors_queue.append(adjacent_face_idx)
                flattened_faces_indices.add(adjacent_face_idx)
                flattened_vertices_indices.update(set(face_indices))

                # Get the spring mass system for the subset of already flattened vertices
                flattened_vertices_2d = vertices_2d[np.array(list(flattened_vertices_indices))]
                flattened_vertices_3d, flattened_edges, flattened_faces = get_mesh_subset(vertices_3d, mesh.edges_unique, mesh.faces, np.array(list(flattened_vertices_indices)))

                # Call Energy release with N=50 steps
                if ENABLE_ENERGY_RELEASE:
                    vertices_2d[np.array(list(flattened_vertices_indices))] = energy_release(
                        flattened_vertices_3d,
                        flattened_edges,
                        flattened_faces,
                        flattened_vertices_2d,
                        spring_constant,
                        area_density,
                        dt,
                        50, # Iteration steps number N=50 as per paper
                        permissible_area_error,
                        permissible_shape_error,
                        permissible_energy_variation,
                        verbose=True
                    )

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
    Performs energy release phase using spring-mass model and Euler's method.
    """
    vertices_3d = vertices_3d.copy()
    faces = faces.copy()
    edges = edges.copy()
    num_vertices = len(vertices_3d)
    vertices_2d = vertices_2d_initial.copy()

    # Calculate initial edge lengths in 3D (rest lengths)
    rest_lengths = {}
    for edge in edges:
        v1_idx, v2_idx = edge
        rest_lengths[(v1_idx, v2_idx)] = np.linalg.norm(
            vertices_3d[v1_idx] - vertices_3d[v2_idx]
        )
        rest_lengths[(v2_idx, v1_idx)] = rest_lengths[(v1_idx, v2_idx)]  # Bidirectional

    # Calculate masses based on connected triangle areas
    masses = np.zeros(num_vertices)
    for i in range(num_vertices):
        connected_faces_indices = np.where(np.any(faces == i, axis=1))[0] # Faces connected to vertex i
        vertex_area = 0.0
        for face_index in connected_faces_indices:
            face_vertices_3d = vertices_3d[faces[face_index]]
            face_area = calculate_face_area(face_vertices_3d)
            vertex_area += face_area
        assert vertex_area != 0
        masses[i] = vertex_area * (area_density / 3.0)

    velocities = np.zeros_like(vertices_2d)
    accelerations = np.zeros_like(vertices_2d)

    prev_energy = float("inf")

    for iteration in range(max_iterations):
        forces = np.zeros_like(vertices_2d)

        # Calculate forces based on spring-mass model
        for edge in edges:
            v1_idx, v2_idx = edge
            p1_2d = vertices_2d[v1_idx]
            p2_2d = vertices_2d[v2_idx]
            current_length_2d = np.linalg.norm(p2_2d - p1_2d)
            rest_length = rest_lengths[(v1_idx, v2_idx)]

            if current_length_2d < 1e-9:
                # raise ValueError("current_length_2d is nearly zero")
                direction = (
                    (p2_2d - p1_2d)
                    if np.linalg.norm(p2_2d - p1_2d) > 1e-9
                    else np.array([1.0, 0.0])
                )
                direction = (
                    direction / np.linalg.norm(direction)
                    if np.linalg.norm(direction) > 1e-9
                    else np.array([1.0, 0.0])
                )
            else:
                direction = (p2_2d - p1_2d) / current_length_2d  # Unit vector

                force_magnitude = spring_constant * (current_length_2d - rest_length)
                force_vector = force_magnitude * direction

                forces[v1_idx] += -force_vector  # Equal and opposite forces
                forces[v2_idx] += force_vector

        # Euler's method integration
        accelerations = forces / masses[:, np.newaxis]  # a = F/m
        velocities += accelerations * dt
        vertices_2d += velocities * dt + 0.5 * accelerations * dt**2

        # --- Calculate Penalty Displacements and apply them ---
        penalty_displacements = calculate_penalty_displacements(vertices_3d, faces, vertices_2d)
        vertices_2d += penalty_displacements # Apply penalty as displacement

        # Calculate Energy (for monitoring convergence - not used for algorithm logic here)
        # TODO: I think this is undercounting because the true formula sums over P_i, where each P_i sums over its edges -- so each edge should be double counted. Does it matter?
        current_energy = 0.0
        for edge in edges:
            v1_idx, v2_idx = edge
            p1_2d = vertices_2d[v1_idx]
            p2_2d = vertices_2d[v2_idx]
            current_length_2d = np.linalg.norm(p2_2d - p1_2d)
            rest_length = rest_lengths[(v1_idx, v2_idx)]
            current_energy += (
                0.5 * spring_constant * (current_length_2d - rest_length) ** 2
            )

        area_error = calculate_area_error(vertices_3d, vertices_2d, faces)
        shape_error = calculate_shape_error(vertices_3d, vertices_2d, edges)
        energy_variation_percentage = (
            abs((current_energy - prev_energy) / prev_energy)
            if prev_energy != float("inf")
            else 1.0
        )

        # --- Termination conditions ---
        if (
            (
                area_error < permissible_area_error and
                shape_error < permissible_shape_error
            ) or energy_variation_percentage < permissible_energy_variation
           ):
            if verbose:
                print(f"Termination at iteration: {iteration}, Area Error: {area_error:.4f}, Shape Error: {shape_error:.4f}, Energy Variation: {energy_variation_percentage:.4f}")
            break
        if iteration > max_iterations - 2: # Max iterations reached
            if verbose:
                print(f"Max iteration termination at iteration: {iteration}, Area Error: {area_error:.4f}, Shape Error: {shape_error:.4f}, Energy Variation: {energy_variation_percentage:.4f}")
            break

        prev_energy = current_energy

    return vertices_2d


def calculate_face_area(face_verts: NDArray[np.float64]):
    # Area of the 3D triangle is half the magnitude of the cross product
    p1_3d, p2_3d, p3_3d = face_verts

    # Calculate two edge vectors
    v1 = p2_3d - p1_3d
    v2 = p3_3d - p1_3d

    return 0.5 * np.linalg.norm(np.cross(v1, v2))


def calculate_penalty_displacements(vertices_3d, faces, vertices_2d, penalty_coefficient = 1.0):
    """
    Calculates penalty displacements for each vertex to prevent overlap.

    Args:
        vertices_3d (numpy.ndarray): The 3D vertex positions of the original mesh.
        faces (numpy.ndarray): The face indices array.
        vertices_2d (numpy.ndarray): The current 2D vertex positions of the flattened mesh.
        penalty_coefficient (float): Coefficient to control penalty strength.

    Returns:
        numpy.ndarray: Penalty displacement vectors for each vertex (same shape as vertices_2d).
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

                penalty_displacement_magnitude = penalty_coefficient * c_penalty * abs(h_j - h_star_j) # Displacement, not force
                penalty_displacement_vertex += penalty_displacement_magnitude * n_hat_2d

        penalty_displacements[vertex_index] = -penalty_displacement_vertex

    return penalty_displacements


def calculate_area_error(vertices_3d, vertices_2d, faces):
    """
    Calculates the Area Accuracy (Es) as described in the paper.
    Calculates 3D face areas using the cross product method.

    Args:
        vertices_3d (numpy.ndarray): The 3D vertex positions of the original mesh.
        faces (numpy.ndarray): The face indices array.
        vertices_2d (numpy.ndarray): The 2D vertex positions of the flattened mesh.

    Returns:
        float: The relative area difference (Es).
    """
    total_area_diff = 0.0
    total_original_area = 0.0

    for face_indices in faces:
        # 3D area calculation (using cross product)
        face_3d_verts = vertices_3d[face_indices]

        face_original_area = calculate_face_area(face_3d_verts)


        # 2D area calculation (using triangle area formula in 2D)
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
        relative_area_error = 0.0  # Avoid division by zero if mesh has zero area

    return relative_area_error


def calculate_shape_error(vertices_3d, vertices_2d, edges):
    """
    Calculates the Shape Accuracy (Ec) as described in the paper.
    Now takes vertices_3d, edges, and vertices_2d as separate arguments.

    Args:
        vertices_3d (numpy.ndarray): The 3D vertex positions of the original mesh.
        edges (numpy.ndarray): The edge indices array (unique edges).
        vertices_2d (numpy.ndarray): The 2D vertex positions of the flattened mesh.

    Returns:
        float: The relative edge length difference (Ec).
    """
    total_length_diff = 0.0
    total_original_length = 0.0

    for edge_indices in edges:
        v1_idx, v2_idx = edge_indices

        # 3D edge length
        p1_3d = vertices_3d[v1_idx]
        p2_3d = vertices_3d[v2_idx]
        original_length = np.linalg.norm(p2_3d - p1_3d)

        # 2D edge length
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
        relative_shape_error = 0.0 # Avoid division by zero if mesh has zero total edge length

    return relative_shape_error


def point_to_segment_distance_2d(point_p, segment_p1, segment_p2):
    """
    Calculates the shortest distance and the vector from a point to a line segment in 2D.

    Args:
        point_p (numpy.ndarray): Point coordinates (2D).
        segment_p1 (numpy.ndarray): Segment start point coordinates (2D).
        segment_p2 (numpy.ndarray): Segment end point coordinates (2D).

    Returns:
        tuple: (shortest distance, vector from point_p to closest point on segment)
    """
    l2 = np.sum((segment_p1 - segment_p2)**2)
    if l2 == 0.0:
        distance = np.linalg.norm(point_p - segment_p1)
        vector_to_segment = segment_p1 - point_p # Vector from point to segment
        return distance, vector_to_segment
    t = max(0, min(1, np.dot(point_p - segment_p1, segment_p2 - segment_p1) / l2))
    projection = segment_p1 + t * (segment_p2 - segment_p1)
    distance = np.linalg.norm(point_p - projection)
    vector_to_segment = projection - point_p # Vector from point to segment
    return distance, vector_to_segment


def point_to_segment_distance_3d(point_q, segment_q1, segment_q2):
    """
    Calculates the shortest distance and the vector from a point to a line segment in 3D.

    Args:
        point_q (numpy.ndarray): Point coordinates (3D).
        segment_q1 (numpy.ndarray): Segment start point coordinates (3D).
        segment_q2 (numpy.ndarray): Segment end point coordinates (3D).

    Returns:
        tuple: (shortest distance, vector from point_q to closest point on segment)
    """
    l2 = np.sum((segment_q1 - segment_q2)**2)
    if l2 == 0.0:
        distance = np.linalg.norm(point_q - segment_q1)
        vector_to_segment = segment_q1 - point_q # Vector from point to segment
        return distance, vector_to_segment
    t = max(0, min(1, np.dot(point_q - segment_q1, segment_q2 - segment_q1) / l2))
    projection = segment_q1 + t * (segment_q2 - segment_q1)
    distance = np.linalg.norm(point_q - projection)
    vector_to_segment = projection - point_q # Vector from point to segment
    return distance, vector_to_segment


def get_opposite_edges(vertex_index, faces):
    """
    Gets the "opposite edges" for a given vertex index based on the faces it belongs to.
    Opposite edges are edges of the faces that do *not* include the vertex.

    Args:
        vertex_index (int): The index of the vertex.
        faces (numpy.ndarray): The face indices array.

    Returns:
        list: List of opposite edge vertex index pairs (tuples).
    """
    opposite_edges = []
    for face in faces:
        if vertex_index in face:
            face_verts = list(face) # Make mutable
            face_verts.remove(vertex_index) # Remove the vertex in question, remaining two are opposite edge
            opposite_edges.append(tuple(sorted(face_verts))) # Ensure consistent edge order (sorted) and tuple for hashability
    return list(set(opposite_edges)) # Use set to get unique edges and convert back to list


def get_mesh_subset(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    faces: NDArray[np.int64],
    vertex_indices_subset: NDArray[np.int64]
):
    """
    Extracts a subset of vertices from a Trimesh mesh and returns a new mesh-like
    structure with re-indexed vertices, edges, and faces.

    Args:
        original_mesh (trimesh.Mesh): The original Trimesh mesh object.
        vertex_indices_subset (list or numpy.ndarray): A list or array of vertex indices
            from the original mesh that define the subset.

    Returns:
        tuple: A tuple containing:
            - subset_vertices (numpy.ndarray): Vertex positions for the subset.
            - subset_edges (numpy.ndarray): Re-indexed edges for the subset.
            - subset_faces (numpy.ndarray): Re-indexed faces for the subset.
    """

    # 1. Get subset vertex positions
    subset_vertices = vertices[vertex_indices_subset]

    # 2. Create a mapping from original indices to new subset indices
    original_to_subset_index_map = {original_index: new_index
                                     for new_index, original_index in enumerate(vertex_indices_subset)}

    # 3. Filter and re-index faces
    subset_faces = []
    for face in faces:
        if all(v_idx in vertex_indices_subset for v_idx in face):
            reindexed_face = [original_to_subset_index_map[v_idx] for v_idx in face]
            subset_faces.append(reindexed_face)
    subset_faces = np.array(subset_faces)

    # 4. Filter and re-index edges
    subset_edges = []
    for edge in edges:
        if all(v_idx in vertex_indices_subset for v_idx in edge):
            reindexed_edge = [original_to_subset_index_map[v_idx] for v_idx in edge]
            subset_edges.append(reindexed_edge)
    subset_edges = np.array(subset_edges)

    return subset_vertices, subset_edges, subset_faces


if __name__ == "__main__":
    # mesh = trimesh.creation.cylinder(5, 10)  # Cylinder mesh
    mesh = trimesh.load('files/Partial_Cylinder_Shell.stl') # Load from file if you have one

    flattened_vertices_2d = surface_flattening_spring_mass(mesh)

    # --- Visualization (requires matplotlib) ---
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
