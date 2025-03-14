"""
Geometry utilities for mesh operations and calculations.
"""

import numpy as np
from numpy.typing import NDArray


def calculate_face_area(face_verts: NDArray[np.float64]) -> float:
    """Calculate the area of a 3D triangle."""
    p1_3d, p2_3d, p3_3d = face_verts
    v1 = p2_3d - p1_3d
    v2 = p3_3d - p1_3d
    return 0.5 * np.linalg.norm(np.cross(v1, v2))


def calculate_face_area_vectorized(faces_verts: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Vectorized implementation to calculate areas for multiple triangular faces.
    
    Args:
        faces_verts: Array of shape (n_faces, 3, 3) where each element is a face
                     with 3 vertices, each vertex having 3 coordinates
        
    Returns:
        numpy.ndarray: Array of face areas with shape (n_faces,)
    """
    # Extract vertices for each face
    p1_3d = faces_verts[:, 0]  # First vertex of each face
    p2_3d = faces_verts[:, 1]  # Second vertex of each face
    p3_3d = faces_verts[:, 2]  # Third vertex of each face
    
    # Calculate edge vectors
    v1 = p2_3d - p1_3d
    v2 = p3_3d - p1_3d
    
    # Calculate cross products
    cross_products = np.cross(v1, v2)
    
    # Calculate face areas
    return 0.5 * np.linalg.norm(cross_products, axis=1)


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


def point_to_segment_distance_2d_batch(points, segment_starts, segment_ends):
    """
    Calculate the shortest distance and vectors from points to line segments in 2D,
    using vectorized operations for better performance.
    
    Args:
        points: Array of points with shape (n, 2)
        segment_starts: Array of segment start points with shape (n, 2)
        segment_ends: Array of segment end points with shape (n, 2)
        
    Returns:
        tuple: (distances, vectors_to_segments)
            - distances: Array of shortest distances with shape (n,)
            - vectors_to_segments: Array of vectors from points to closest points on segments with shape (n, 2)
    """
    # Calculate squared segment lengths
    segments = segment_ends - segment_starts
    l2 = np.sum(segments**2, axis=1)
    
    # Handle zero-length segments
    zero_length_mask = l2 < 1e-12
    
    # Initialize arrays for results
    distances = np.zeros(len(points))
    vectors_to_segments = np.zeros_like(points)
    
    if np.any(zero_length_mask):
        # Handle zero-length segments
        zero_idxs = np.where(zero_length_mask)[0]
        distances[zero_idxs] = np.linalg.norm(points[zero_idxs] - segment_starts[zero_idxs], axis=1)
        vectors_to_segments[zero_idxs] = segment_starts[zero_idxs] - points[zero_idxs]
    
    if np.any(~zero_length_mask):
        # Handle non-zero length segments
        non_zero_idxs = np.where(~zero_length_mask)[0]
        
        # Calculate t parameters
        t_values = np.zeros(len(non_zero_idxs))
        for i, idx in enumerate(non_zero_idxs):
            t_values[i] = np.dot(points[idx] - segment_starts[idx], segments[idx]) / l2[idx]
        
        # Clamp t to [0, 1]
        t_values = np.clip(t_values, 0, 1)
        
        # Calculate projections
        projections = np.zeros((len(non_zero_idxs), 2))
        for i, idx in enumerate(non_zero_idxs):
            projections[i] = segment_starts[idx] + t_values[i] * segments[idx]
        
        # Calculate distances and vectors
        distances[non_zero_idxs] = np.linalg.norm(points[non_zero_idxs] - projections, axis=1)
        vectors_to_segments[non_zero_idxs] = projections - points[non_zero_idxs]
    
    return distances, vectors_to_segments


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


def point_to_segment_distance_3d_batch(points, segment_starts, segment_ends):
    """
    Calculate the shortest distance and vectors from points to line segments in 3D,
    using vectorized operations for better performance.
    
    Args:
        points: Array of points with shape (n, 3)
        segment_starts: Array of segment start points with shape (n, 3)
        segment_ends: Array of segment end points with shape (n, 3)
        
    Returns:
        tuple: (distances, vectors_to_segments)
            - distances: Array of shortest distances with shape (n,)
            - vectors_to_segments: Array of vectors from points to closest points on segments with shape (n, 3)
    """
    # Calculate squared segment lengths
    segments = segment_ends - segment_starts
    l2 = np.sum(segments**2, axis=1)
    
    # Handle zero-length segments
    zero_length_mask = l2 < 1e-12
    
    # Initialize arrays for results
    distances = np.zeros(len(points))
    vectors_to_segments = np.zeros_like(points)
    
    if np.any(zero_length_mask):
        # Handle zero-length segments
        zero_idxs = np.where(zero_length_mask)[0]
        distances[zero_idxs] = np.linalg.norm(points[zero_idxs] - segment_starts[zero_idxs], axis=1)
        vectors_to_segments[zero_idxs] = segment_starts[zero_idxs] - points[zero_idxs]
    
    if np.any(~zero_length_mask):
        # Handle non-zero length segments
        non_zero_idxs = np.where(~zero_length_mask)[0]
        
        # Calculate t parameters
        t_values = np.zeros(len(non_zero_idxs))
        for i, idx in enumerate(non_zero_idxs):
            t_values[i] = np.dot(points[idx] - segment_starts[idx], segments[idx]) / l2[idx]
        
        # Clamp t to [0, 1]
        t_values = np.clip(t_values, 0, 1)
        
        # Calculate projections
        projections = np.zeros((len(non_zero_idxs), 3))
        for i, idx in enumerate(non_zero_idxs):
            projections[i] = segment_starts[idx] + t_values[i] * segments[idx]
        
        # Calculate distances and vectors
        distances[non_zero_idxs] = np.linalg.norm(points[non_zero_idxs] - projections, axis=1)
        vectors_to_segments[non_zero_idxs] = projections - points[non_zero_idxs]
    
    return distances, vectors_to_segments


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


def precompute_all_opposite_edges(vertices, faces):
    """
    Precompute opposite edges for all vertices at once.
    
    Args:
        vertices: Vertex positions array
        faces: Face indices array
        
    Returns:
        list: List where each item contains the opposite edges for the corresponding vertex
    """
    num_vertices = len(vertices)
    all_opposite_edges = [[] for _ in range(num_vertices)]
    
    # Process each face once to find opposite edges for all vertices
    for face in faces:
        for i, vertex_index in enumerate(face):
            # The other two vertices in this face form an opposite edge for this vertex
            other_vertices = [face[j] for j in range(3) if j != i]
            all_opposite_edges[vertex_index].append(tuple(sorted(other_vertices)))
    
    # Remove duplicates in each vertex's opposite edges list
    for i in range(num_vertices):
        all_opposite_edges[i] = list(set(all_opposite_edges[i]))
    
    return all_opposite_edges


def get_mesh_subset(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int64],
    faces: NDArray[np.int64],
    vertex_indices_subset: NDArray[np.int64],
    rest_lengths: dict = None,
    all_opposite_edges: list = None
):
    """
    Extract a subset of vertices from a mesh and return re-indexed vertices, edges, and faces.
    
    Args:
        vertices: Original vertex positions
        edges: Original edge indices
        faces: Original face indices
        vertex_indices_subset: Indices of vertices to include in the subset
        rest_lengths: Optional dictionary of rest lengths for the edges
        all_opposite_edges: Optional list of opposite edges for all vertices
        
    Returns:
        tuple: A 5-tuple containing:
            - subset_vertices: Vertex positions for the subset
            - subset_edges: Re-indexed edges for the subset
            - subset_faces: Re-indexed faces for the subset
            - subset_rest_lengths: Re-indexed rest lengths or None if not provided
            - subset_all_opposite_edges: Re-indexed opposite edges or None if not provided
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
    
    # Handle rest lengths if provided
    subset_rest_lengths = None
    if rest_lengths is not None:
        subset_rest_lengths = {}
        for edge in subset_edges:
            v1_idx, v2_idx = edge
            orig_v1_idx = vertex_indices_subset[v1_idx]
            orig_v2_idx = vertex_indices_subset[v2_idx]
            
            # Assert that rest length exists for this edge
            assert (orig_v1_idx, orig_v2_idx) in rest_lengths or (orig_v2_idx, orig_v1_idx) in rest_lengths, \
                f"Rest length not found for edge ({orig_v1_idx}, {orig_v2_idx})"
                
            # Get original rest length (prefer v1->v2 but fall back to v2->v1)
            if (orig_v1_idx, orig_v2_idx) in rest_lengths:
                length = rest_lengths[(orig_v1_idx, orig_v2_idx)]
            else:
                length = rest_lengths[(orig_v2_idx, orig_v1_idx)]
                
            subset_rest_lengths[(v1_idx, v2_idx)] = length
            subset_rest_lengths[(v2_idx, v1_idx)] = length  # Bidirectional
    
    # Handle opposite edges if provided
    subset_all_opposite_edges = None
    if all_opposite_edges is not None:
        subset_all_opposite_edges = []
        
        # For each vertex in the subset
        for new_idx, orig_idx in enumerate(vertex_indices_subset):
            # Get the original opposite edges
            orig_opposite_edges = all_opposite_edges[orig_idx]
            
            # Convert to subset indices
            subset_opposite_edges = []
            for orig_edge in orig_opposite_edges:
                # Only include edges where both vertices are in the subset
                if all(v_idx in vertex_indices_subset for v_idx in orig_edge):
                    subset_edge = tuple(sorted(
                        original_to_subset_index_map[v_idx] for v_idx in orig_edge
                    ))
                    subset_opposite_edges.append(subset_edge)
            
            subset_all_opposite_edges.append(subset_opposite_edges)
    
    return subset_vertices, subset_edges, subset_faces, subset_rest_lengths, subset_all_opposite_edges


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