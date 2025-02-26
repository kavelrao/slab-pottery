"""
Test script to verify that vectorized metric calculations produce the same results
as their non-vectorized counterparts.
"""

import os
import sys
import time
import numpy as np
import trimesh


from flattening.metrics import (
    calculate_area_error,
    calculate_area_error_vectorized,
    calculate_shape_error, 
    calculate_shape_error_vectorized
)
from flattening.geometry import (
    calculate_face_area,
    calculate_face_area_vectorized
)


def test_face_area_calculation(faces_3d):
    """
    Test that both face area calculation methods produce the same results.
    
    Args:
        faces_3d: 3D face vertices of shape (n_faces, 3, 3)
        
    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Calculate areas using both methods
    areas_regular = np.array([calculate_face_area(face) for face in faces_3d])
    areas_vectorized = calculate_face_area_vectorized(faces_3d)
    
    # Compare results
    diff = np.abs(areas_regular - areas_vectorized).max()
    
    print(f"Max face area difference: {diff}")
    
    # Check if difference is within tolerance
    tolerance = 1e-9
    return diff < tolerance


def test_area_error_calculation(vertices_3d, vertices_2d, faces):
    """
    Test that both area error calculation methods produce the same results.
    
    Args:
        vertices_3d: 3D vertex positions
        vertices_2d: 2D vertex positions
        faces: Face indices
        
    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Calculate area error using both methods
    error_regular = calculate_area_error(vertices_3d, vertices_2d, faces)
    error_vectorized = calculate_area_error_vectorized(vertices_3d, vertices_2d, faces)
    
    # Compare results
    diff = abs(error_regular - error_vectorized)
    
    print(f"Area error (regular): {error_regular}")
    print(f"Area error (vectorized): {error_vectorized}")
    print(f"Area error difference: {diff}")
    
    # Check if difference is within tolerance
    tolerance = 1e-9
    return diff < tolerance


def test_shape_error_calculation(vertices_3d, vertices_2d, edges):
    """
    Test that both shape error calculation methods produce the same results.
    
    Args:
        vertices_3d: 3D vertex positions
        vertices_2d: 2D vertex positions
        edges: Edge indices
        
    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Calculate shape error using both methods
    error_regular = calculate_shape_error(vertices_3d, vertices_2d, edges)
    error_vectorized = calculate_shape_error_vectorized(vertices_3d, vertices_2d, edges)
    
    # Compare results
    diff = abs(error_regular - error_vectorized)
    
    print(f"Shape error (regular): {error_regular}")
    print(f"Shape error (vectorized): {error_vectorized}")
    print(f"Shape error difference: {diff}")
    
    # Check if difference is within tolerance
    tolerance = 1e-9
    return diff < tolerance


def create_test_mesh():
    """Create a small test mesh for validation."""
    # Create a small mesh (e.g., a tetrahedron)
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=int)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_test_data():
    """Create test data for validation."""
    # Create a simple mesh
    mesh = create_test_mesh()
    
    # Create 2D positions for the vertices (as if they were already flattened)
    vertices_2d = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],
        [0.5, 0.289]
    ], dtype=float)
    
    # Gather the 3D faces for face area calculation
    faces_3d = mesh.vertices[mesh.faces]
    
    return mesh.vertices, vertices_2d, mesh.faces, mesh.edges_unique, faces_3d


def main():
    """
    Main test function to compare vectorized and non-vectorized metric implementations.
    """
    print("Testing vectorized metric calculations...")
    print("-" * 60)
    
    # Create test data
    vertices_3d, vertices_2d, faces, edges, faces_3d = create_test_data()
    
    # Test face area calculation
    areas_match = test_face_area_calculation(faces_3d)
    print(f"Face area calculation test {'PASSED' if areas_match else 'FAILED'}")
    
    # Test area error calculation
    area_error_match = test_area_error_calculation(vertices_3d, vertices_2d, faces)
    print(f"Area error calculation test {'PASSED' if area_error_match else 'FAILED'}")
    
    # Test shape error calculation
    shape_error_match = test_shape_error_calculation(vertices_3d, vertices_2d, edges)
    print(f"Shape error calculation test {'PASSED' if shape_error_match else 'FAILED'}")
    
    # Create larger random mesh for performance comparison
    print("\nTesting with a larger random mesh...")
    print("-" * 60)
    
    # Create a larger test mesh (e.g., random point cloud)
    num_vertices = 1000
    np.random.seed(42)  # For reproducibility
    
    # Create random 3D vertices
    vertices_3d_large = np.random.rand(num_vertices, 3)
    
    # Create a mesh from the point cloud (using trimesh's convex hull)
    mesh_large = trimesh.Trimesh(vertices=vertices_3d_large).convex_hull
    
    # Create 2D positions (random projection)
    vertices_2d_large = np.random.rand(len(mesh_large.vertices), 2)
    
    # Get the faces and edges from the mesh
    faces_large = mesh_large.faces
    edges_large = mesh_large.edges_unique
    
    # Gather the 3D faces for face area calculation
    faces_3d_large = mesh_large.vertices[faces_large]
    
    # Verify results
    areas_match_large = test_face_area_calculation(faces_3d_large)
    print(f"Face area calculation test (large mesh) {'PASSED' if areas_match_large else 'FAILED'}")
    
    area_error_match_large = test_area_error_calculation(mesh_large.vertices, vertices_2d_large, faces_large)
    print(f"Area error calculation test (large mesh) {'PASSED' if area_error_match_large else 'FAILED'}")
    
    shape_error_match_large = test_shape_error_calculation(mesh_large.vertices, vertices_2d_large, edges_large)
    print(f"Shape error calculation test (large mesh) {'PASSED' if shape_error_match_large else 'FAILED'}")
    
    print("\nPerformance comparison...")
    print("-" * 60)
    
    # Time face area calculations
    start_time = time.time()
    _ = np.array([calculate_face_area(face) for face in faces_3d_large])
    non_vectorized_face_area_time = time.time() - start_time
    print(f"Non-vectorized face area calculation time: {non_vectorized_face_area_time:.6f} seconds")
    
    start_time = time.time()
    _ = calculate_face_area_vectorized(faces_3d_large)
    vectorized_face_area_time = time.time() - start_time
    print(f"Vectorized face area calculation time: {vectorized_face_area_time:.6f} seconds")
    print(f"Speedup: {non_vectorized_face_area_time / vectorized_face_area_time:.2f}x")
    
    # Time area error calculations
    start_time = time.time()
    _ = calculate_area_error(mesh_large.vertices, vertices_2d_large, faces_large)
    non_vectorized_area_error_time = time.time() - start_time
    print(f"Non-vectorized area error calculation time: {non_vectorized_area_error_time:.6f} seconds")
    
    start_time = time.time()
    _ = calculate_area_error_vectorized(mesh_large.vertices, vertices_2d_large, faces_large)
    vectorized_area_error_time = time.time() - start_time
    print(f"Vectorized area error calculation time: {vectorized_area_error_time:.6f} seconds")
    print(f"Speedup: {non_vectorized_area_error_time / vectorized_area_error_time:.2f}x")
    
    # Time shape error calculations
    start_time = time.time()
    _ = calculate_shape_error(mesh_large.vertices, vertices_2d_large, edges_large)
    non_vectorized_shape_error_time = time.time() - start_time
    print(f"Non-vectorized shape error calculation time: {non_vectorized_shape_error_time:.6f} seconds")
    
    start_time = time.time()
    _ = calculate_shape_error_vectorized(mesh_large.vertices, vertices_2d_large, edges_large)
    vectorized_shape_error_time = time.time() - start_time
    print(f"Vectorized shape error calculation time: {vectorized_shape_error_time:.6f} seconds")
    print(f"Speedup: {non_vectorized_shape_error_time / vectorized_shape_error_time:.2f}x")


if __name__ == "__main__":
    main()