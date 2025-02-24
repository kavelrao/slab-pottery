"""
Test script to verify that the vectorized penalty displacement calculation produces the same results
as the original implementation, using a real mesh file.
"""

import os
import sys
import numpy as np
import trimesh
import time

# Add the parent directory to the Python path so we can import the flattening module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flattening.physics import (
    calculate_penalty_displacements,
    calculate_penalty_displacements_vectorized
)
from flattening.geometry import precompute_all_opposite_edges


def test_penalty_displacements(vertices_3d, faces, vertices_2d, penalty_coefficient=1.0):
    """
    Test that both penalty displacement calculation methods produce the same results.
    
    Args:
        vertices_3d: 3D vertex positions
        faces: Face indices
        vertices_2d: 2D vertex positions
        penalty_coefficient: Coefficient for penalty calculation
        
    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Calculate displacement using both methods
    all_opposite_edges = precompute_all_opposite_edges(vertices_3d, faces)
    
    # Original method
    displacements_regular = calculate_penalty_displacements(
        vertices_3d, faces, vertices_2d, penalty_coefficient
    )
    
    # Vectorized method with precomputed opposite edges
    displacements_vectorized = calculate_penalty_displacements_vectorized(
        vertices_3d, faces, vertices_2d, penalty_coefficient, all_opposite_edges
    )
    
    # Compare results
    diff = np.abs(displacements_regular - displacements_vectorized).max()
    
    print(f"Max penalty displacement difference: {diff}")
    
    # Check if difference is within tolerance
    tolerance = 1e-9
    return diff < tolerance


def create_random_2d_positions(vertices_3d):
    """Create random 2D positions for the given 3D vertices."""
    np.random.seed(42)  # For reproducibility
    
    # Project vertices to XY plane and add some random noise
    vertices_2d = np.column_stack([
        vertices_3d[:, 0] + np.random.rand(len(vertices_3d)) * 0.1,
        vertices_3d[:, 1] + np.random.rand(len(vertices_3d)) * 0.1
    ])
    
    return vertices_2d


def main():
    """
    Main test function to compare original and vectorized penalty displacement calculations.
    """
    print("Testing vectorized penalty displacement calculation...")
    print("-" * 60)
    
    # Load the specified mesh file
    mesh_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'files', 'Partial_Oblong_Cylinder_Shell.stl')
    
    # Check if the file exists
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return
    
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create 2D positions for testing
    vertices_2d = create_random_2d_positions(mesh.vertices)
    
    # Test penalty displacement calculation
    displacements_match = test_penalty_displacements(mesh.vertices, mesh.faces, vertices_2d)
    print(f"Penalty displacement calculation test {'PASSED' if displacements_match else 'FAILED'}")
    
    # Performance comparison
    print("\nPerformance comparison...")
    print("-" * 60)
    
    # Precompute opposite edges
    print("Precomputing opposite edges...")
    start_time = time.time()
    all_opposite_edges = precompute_all_opposite_edges(mesh.vertices, mesh.faces)
    precompute_time = time.time() - start_time
    print(f"Precomputing opposite edges time: {precompute_time:.6f} seconds")
    
    print("\nRunning comparison tests...")
    
    # Time the original penalty displacement calculation
    start_time = time.time()
    _ = calculate_penalty_displacements(
        mesh.vertices, mesh.faces, vertices_2d
    )
    regular_time = time.time() - start_time
    print(f"Original penalty displacement calculation time: {regular_time:.6f} seconds")
    
    # Time the vectorized penalty displacement calculation with precomputed opposite edges
    start_time = time.time()
    _ = calculate_penalty_displacements_vectorized(
        mesh.vertices, mesh.faces, vertices_2d, 
        all_opposite_edges=all_opposite_edges
    )
    vectorized_time = time.time() - start_time
    print(f"Vectorized penalty displacement calculation time: {vectorized_time:.6f} seconds")
    
    # Calculate speedup
    if vectorized_time > 0:
        print(f"Speedup: {regular_time / vectorized_time:.2f}x")
    else:
        print("Could not calculate speedup (division by zero)")
    
    # Include the precomputation time in the total vectorized approach time
    total_vectorized_time = precompute_time + vectorized_time
    print(f"\nTotal vectorized approach (including precomputation): {total_vectorized_time:.6f} seconds")
    if total_vectorized_time > 0:
        print(f"Overall speedup: {regular_time / total_vectorized_time:.2f}x")


if __name__ == "__main__":
    main()