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