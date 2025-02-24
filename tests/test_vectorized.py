"""
Test script to verify that vectorized force and energy calculations produce the same results
as their non-vectorized counterparts.
"""

import os
import sys
import numpy as np
import trimesh

# Add the parent directory to the Python path so we can import the flattening module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flattening.physics import (
    calculate_forces,
    calculate_forces_vectorized,
    calculate_energy,
    calculate_energy_vectorized,
    calculate_rest_lengths
)


def test_forces_calculation(vertices_2d, edges, rest_lengths, spring_constant=0.5):
    """
    Test that both force calculation methods produce the same results.
    
    Args:
        vertices_2d: 2D vertex positions
        edges: Edge indices
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for force calculation
        
    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Calculate forces using both methods
    forces_regular = calculate_forces(vertices_2d, edges, rest_lengths, spring_constant)
    forces_vectorized = calculate_forces_vectorized(vertices_2d, edges, rest_lengths, spring_constant)
    
    # Compare results
    diff = np.abs(forces_regular - forces_vectorized).max()
    
    print(f"Max force difference: {diff}")
    
    # Check if difference is within tolerance
    tolerance = 1e-9
    return diff < tolerance


def test_energy_calculation(vertices_2d, edges, rest_lengths, spring_constant=0.5):
    """
    Test that both energy calculation methods produce the same results.
    
    Args:
        vertices_2d: 2D vertex positions
        edges: Edge indices
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for energy calculation
        
    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Calculate energy using both methods
    energy_regular = calculate_energy(vertices_2d, edges, rest_lengths, spring_constant)
    energy_vectorized = calculate_energy_vectorized(vertices_2d, edges, rest_lengths, spring_constant)
    
    # Compare results
    diff = abs(energy_regular - energy_vectorized)
    
    print(f"Energy (regular): {energy_regular}")
    print(f"Energy (vectorized): {energy_vectorized}")
    print(f"Energy difference: {diff}")
    
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
    
    # Get the edges from the mesh
    edges = mesh.edges_unique
    
    # Calculate rest lengths
    rest_lengths = calculate_rest_lengths(mesh.vertices, edges)
    
    return vertices_2d, edges, rest_lengths


def main():
    """
    Main test function to compare vectorized and non-vectorized implementations.
    """
    print("Testing vectorized force and energy calculations...")
    print("-" * 60)
    
    # Create test data
    vertices_2d, edges, rest_lengths = create_test_data()
    
    # Test forces calculation
    forces_match = test_forces_calculation(vertices_2d, edges, rest_lengths)
    print(f"Forces calculation test {'PASSED' if forces_match else 'FAILED'}")
    
    # Test energy calculation
    energy_match = test_energy_calculation(vertices_2d, edges, rest_lengths)
    print(f"Energy calculation test {'PASSED' if energy_match else 'FAILED'}")
    
    # Create larger random mesh for performance comparison
    print("\nTesting with a larger random mesh...")
    print("-" * 60)
    
    # Create a larger test mesh (e.g., random point cloud)
    num_vertices = 1000
    np.random.seed(42)  # For reproducibility
    
    # Create random 3D vertices
    vertices_3d = np.random.rand(num_vertices, 3)
    
    # Create a mesh from the point cloud (using trimesh's convex hull)
    mesh_large = trimesh.Trimesh(vertices=vertices_3d).convex_hull
    
    # Create 2D positions (random projection)
    vertices_2d_large = np.random.rand(len(mesh_large.vertices), 2)
    
    # Get the edges from the mesh
    edges_large = mesh_large.edges_unique
    
    # Calculate rest lengths
    rest_lengths_large = calculate_rest_lengths(mesh_large.vertices, edges_large)
    
    # Test forces calculation
    forces_match_large = test_forces_calculation(
        vertices_2d_large, edges_large, rest_lengths_large
    )
    print(f"Forces calculation test (large mesh) {'PASSED' if forces_match_large else 'FAILED'}")
    
    # Test energy calculation
    energy_match_large = test_energy_calculation(
        vertices_2d_large, edges_large, rest_lengths_large
    )
    print(f"Energy calculation test (large mesh) {'PASSED' if energy_match_large else 'FAILED'}")
    
    print("\nPerformance comparison...")
    print("-" * 60)
    
    # Time the force calculations
    import time
    
    # Time non-vectorized force calculation
    start_time = time.time()
    _ = calculate_forces(vertices_2d_large, edges_large, rest_lengths_large, 0.5)
    non_vectorized_force_time = time.time() - start_time
    print(f"Non-vectorized force calculation time: {non_vectorized_force_time:.6f} seconds")
    
    # Time vectorized force calculation
    start_time = time.time()
    _ = calculate_forces_vectorized(vertices_2d_large, edges_large, rest_lengths_large, 0.5)
    vectorized_force_time = time.time() - start_time
    print(f"Vectorized force calculation time: {vectorized_force_time:.6f} seconds")
    print(f"Speedup: {non_vectorized_force_time / vectorized_force_time:.2f}x")
    
    # Time non-vectorized energy calculation
    start_time = time.time()
    _ = calculate_energy(vertices_2d_large, edges_large, rest_lengths_large, 0.5)
    non_vectorized_energy_time = time.time() - start_time
    print(f"Non-vectorized energy calculation time: {non_vectorized_energy_time:.6f} seconds")
    
    # Time vectorized energy calculation
    start_time = time.time()
    _ = calculate_energy_vectorized(vertices_2d_large, edges_large, rest_lengths_large, 0.5)
    vectorized_energy_time = time.time() - start_time
    print(f"Vectorized energy calculation time: {vectorized_energy_time:.6f} seconds")
    print(f"Speedup: {non_vectorized_energy_time / vectorized_energy_time:.2f}x")


if __name__ == "__main__":
    main()