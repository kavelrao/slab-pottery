"""
Error metrics for evaluating flattening quality.
"""

import numpy as np
from numpy.typing import NDArray

from .geometry import calculate_face_area, calculate_face_area_vectorized


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


def calculate_area_error_vectorized(vertices_3d, vertices_2d, faces):
    """
    Vectorized implementation to calculate the Area Accuracy (Es) as described in the paper.
    
    Args:
        vertices_3d: 3D vertex positions
        vertices_2d: 2D vertex positions
        faces: Face indices
        
    Returns:
        float: Relative area error
    """
    # Prepare 3D faces for area calculation
    faces_3d = vertices_3d[faces]
    # Calculate original face areas
    original_areas = calculate_face_area_vectorized(faces_3d)
    
    # Prepare 2D faces for area calculation
    faces_2d = vertices_2d[faces]
    
    # Calculate flattened face areas
    # Since we're in 2D, we can use a simpler formula for triangle area
    # A = 0.5 * |cross(v1, v2)|
    p1_2d = faces_2d[:, 0]
    p2_2d = faces_2d[:, 1]
    p3_2d = faces_2d[:, 2]
    
    v1 = p2_2d - p1_2d
    v2 = p3_2d - p1_2d
    
    # For 2D points, the cross product is a scalar (z-component only)
    cross_products_2d = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    flattened_areas = 0.5 * np.abs(cross_products_2d)
    
    # Calculate area differences
    area_diffs = np.abs(original_areas - flattened_areas)
    
    # Calculate total differences and total original area
    total_area_diff = np.sum(area_diffs)
    total_original_area = np.sum(original_areas)
    
    # Calculate relative error
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


def calculate_shape_error_vectorized(vertices_3d, vertices_2d, edges):
    """
    Vectorized implementation to calculate the Shape Accuracy (Ec) as described in the paper.
    
    Args:
        vertices_3d: 3D vertex positions
        vertices_2d: 2D vertex positions
        edges: Edge indices
        
    Returns:
        float: Relative shape error
    """
    # Extract edge endpoints
    v1_indices = edges[:, 0]
    v2_indices = edges[:, 1]
    
    # Get 3D positions of edge endpoints
    p1_3d = vertices_3d[v1_indices]
    p2_3d = vertices_3d[v2_indices]
    
    # Calculate original edge lengths
    edge_vectors_3d = p2_3d - p1_3d
    original_lengths = np.linalg.norm(edge_vectors_3d, axis=1)
    
    # Get 2D positions of edge endpoints
    p1_2d = vertices_2d[v1_indices]
    p2_2d = vertices_2d[v2_indices]
    
    # Calculate flattened edge lengths
    edge_vectors_2d = p2_2d - p1_2d
    flattened_lengths = np.linalg.norm(edge_vectors_2d, axis=1)
    
    # Calculate length differences
    length_diffs = np.abs(original_lengths - flattened_lengths)
    
    # Calculate total differences and total original length
    total_length_diff = np.sum(length_diffs)
    total_original_length = np.sum(original_lengths)
    
    # Calculate relative error
    if total_original_length > 0:
        relative_shape_error = total_length_diff / total_original_length
    else:
        print("Warning: total edge length is 0")
        relative_shape_error = 0.0
    
    return relative_shape_error