"""
Error metrics for evaluating flattening quality.
"""

import numpy as np
from numpy.typing import NDArray

from .geometry import calculate_face_area


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