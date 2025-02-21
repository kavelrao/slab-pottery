import trimesh
import numpy as np
from numpy.typing import NDArray


"""
TODO: What's not implemented from the paper rn?
- Surface Cutting
- Penalty Function
"""


def surface_flattening_spring_mass(
    mesh: trimesh.Trimesh,
    spring_constant: float=0.5,
    area_density: float=1.0,
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
    Implements a simplified constrained triangle flattening (averaging positions when constrained).
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
    # TODO: Check here if there are problems, not sure about this max with 0 -- maybe should throw an error instead?
    x = (l12**2 - l23**2 + l13**2) / (2 * l12)
    y = np.sqrt(max(0, l13**2 - x**2))  # Ensure positive sqrt
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
                assert adjacent_face_idx > 0, "adjacent_face_idx was 0"

                # Process the adjacent face
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

                # TODO: see if this check is needed
                if False: #l12_2d < 1e-9:  # Prevent division by zero or near zero
                    raise ValueError("l12_2d is near zero")
                    # direction = (
                    #     (p2_2d_constrained - p1_2d_constrained)
                    #     if np.linalg.norm(p2_2d_constrained - p1_2d_constrained) > 1e-9
                    #     else np.array([1.0, 0.0])
                    # )
                    # direction = (
                    #     direction / np.linalg.norm(direction)
                    #     if np.linalg.norm(direction) > 1e-9
                    #     else np.array([1.0, 0.0])
                    # )

                    # p2_2d_temp = p1_2d_constrained + direction * l12  # Use original 3D length
                    # p1_2d_temp = p1_2d_constrained
                else:
                    p1_2d_temp = p1_2d_constrained
                    p2_2d_temp = p2_2d_constrained

                l12_2d_temp = np.linalg.norm(p2_2d_temp - p1_2d_temp)
                x = (l12_2d_temp**2 - l23**2 + l13**2) / (2 * l12_2d_temp)
                y_sq = l13**2 - x**2
                # TODO: check this max with 0, should we be doing this here?
                y = np.sqrt(max(0, y_sq))  # Ensure positive sqrt

                # Determine sign of y (up or down). Heuristic to avoid fold-over
                p3_2d_option1 = p1_2d_temp + np.array([x, y])
                p3_2d_option2 = p1_2d_temp + np.array([x, -y])

                # Heuristic to pick option 1 or 2: compare cross product sign
                v_indices_in_common = np.intersect1d(face_indices, prev_face_indices)
                if len(v_indices_in_common) >= 2:
                    v_ref1_idx = v_indices_in_common[0]
                    v_ref2_idx = v_indices_in_common[1]
                    vec_ref_2d = vertices_2d[v_ref2_idx] - vertices_2d[v_ref1_idx]

                    diff_indices = np.setdiff1d(
                        face_indices, v_indices_in_common
                    )  # Calculate set difference

                    # Check if diff_indices is empty or has more than one element (unexpected)
                    if len(diff_indices) != 1:  # Expecting exactly one unique index
                        assert False, f"Warning: Unexpected number of diff_indices: {len(diff_indices)}. Defaulting cross_prod."
                        # reference_cross_prod = 1.0
                    else:
                        index_to_use = diff_indices[0]
                        # Validate index - although this should now be less likely to be needed if setdiff1d works as expected in normal cases.
                        if index_to_use in face_indices:
                            reference_cross_prod = np.cross(
                                vec_ref_2d,
                                vertices_2d[index_to_use] - vertices_2d[v_ref1_idx],
                            )
                        else:
                            assert False, f"Critical Warning: index_to_use {index_to_use} still not in face_indices {face_indices} after diff_indices check. Defaulting cross_prod."
                            # reference_cross_prod = 1.0
                else:
                    reference_cross_prod = 1.0

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

                # Energy release with N=50 steps
                vertices_2d[np.array(list(flattened_vertices_indices))] = energy_release(
                    flattened_vertices_3d,
                    flattened_edges,
                    flattened_faces,
                    flattened_vertices_2d,
                    spring_constant,
                    area_density,
                    dt,
                    15,
                    permissible_area_error,
                    permissible_shape_error,
                    permissible_energy_variation,
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
            face_area = mesh.area_faces[face_index] # Use trimesh's area calculation
            vertex_area += face_area
        assert vertex_area != 0
        masses[i] = vertex_area * area_density

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

        # --- Termination conditions (Simplified - Area and Shape error not implemented for brevity in this example) ---
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
        p1_3d, p2_3d, p3_3d = face_3d_verts

        # Calculate two edge vectors
        v1 = p2_3d - p1_3d
        v2 = p3_3d - p1_3d

        # Area of the 3D triangle is half the magnitude of the cross product
        face_original_area = 0.5 * np.linalg.norm(np.cross(v1, v2))


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
    # Load a sample mesh (replace 'your_mesh.obj' with your mesh file)
    # You can try a simple mesh like a sphere or cube for testing
    mesh = trimesh.creation.cylinder(5, 10)  # Example mesh
    # mesh = trimesh.load('your_mesh.obj') # Load from file if you have one

    flattened_vertices_2d = surface_flattening_spring_mass(mesh)

    # --- Visualization (requires matplotlib) ---
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots()
    ax.set_aspect("equal")  # Ensure aspect ratio is 1:1

    # Create PolyCollection for faces in 2D
    face_verts_2d = flattened_vertices_2d[mesh.faces]

    poly_collection = PolyCollection(
        face_verts_2d, facecolors="skyblue", edgecolors="black", linewidths=0.5
    )
    ax.add_collection(poly_collection)

    # Set plot limits to encompass the flattened mesh
    min_coords = np.min(flattened_vertices_2d, axis=0)
    max_coords = np.max(flattened_vertices_2d, axis=0)
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    padding_x = range_x * 0.1  # 10% padding
    padding_y = range_y * 0.1

    ax.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
    ax.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)

    ax.set_title("Flattened Surface (Spring-Mass Model)")
    plt.show()
