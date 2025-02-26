"""
Test script to verify the flattening algorithms.
"""

import os
import sys
import numpy as np
import trimesh

# Add the parent directory to the Python path so we can import the flattening module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flattening.algorithms import (
    initial_flattening,
    energy_release,
    surface_flattening_spring_mass
)


def create_test_mesh():
    """Create a simple test mesh for validation."""
    # Create a simple tetrahedron
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


def test_initial_flattening():
    """Test the initial_flattening function."""
    # Create a test mesh
    mesh = create_test_mesh()
    
    # Parameters for initial flattening
    spring_constant = 0.5
    area_density = 1.0
    dt = 0.001
    permissible_area_error = 0.01
    permissible_shape_error = 0.01
    permissible_energy_variation = 0.0005
    penalty_coefficient = 1.0
    enable_energy_release = False
    energy_release_iterations = 1
    
    # Run initial flattening
    vertices_2d = initial_flattening(
        mesh,
        spring_constant,
        area_density,
        dt,
        permissible_area_error,
        permissible_shape_error,
        permissible_energy_variation,
        penalty_coefficient,
        enable_energy_release,
        energy_release_iterations
    )
    
    # Check that the result has the correct shape
    assert vertices_2d.shape == (len(mesh.vertices), 2)
    
    # Check that all vertices are flattened (have 2D coordinates)
    assert not np.any(np.isnan(vertices_2d))
    
    # Check that the first vertex is at the origin
    assert np.allclose(vertices_2d[0], np.array([0.0, 0.0]))
    
    # Check that the second vertex is on the x-axis
    assert np.isclose(vertices_2d[1][1], 0.0)
    
    # Test edge length preservation between 3D and 2D for the first triangle
    # Edge 0-1
    length_3d_01 = np.linalg.norm(mesh.vertices[1] - mesh.vertices[0])
    length_2d_01 = np.linalg.norm(vertices_2d[1] - vertices_2d[0])
    assert np.isclose(length_2d_01, length_3d_01)
    
    # Edge 0-2
    length_3d_02 = np.linalg.norm(mesh.vertices[2] - mesh.vertices[0])
    length_2d_02 = np.linalg.norm(vertices_2d[2] - vertices_2d[0])
    assert np.isclose(length_2d_02, length_3d_02)
    
    # Edge 1-2
    length_3d_12 = np.linalg.norm(mesh.vertices[2] - mesh.vertices[1])
    length_2d_12 = np.linalg.norm(vertices_2d[2] - vertices_2d[1])
    assert np.isclose(length_2d_12, length_3d_12)


def test_energy_release():
    """Test the energy_release function."""
    # Create a test mesh
    mesh = create_test_mesh()
    
    # Parameters for energy release
    spring_constant = 0.5
    area_density = 1.0
    dt = 0.001
    max_iterations = 10
    permissible_area_error = 0.01
    permissible_shape_error = 0.01
    permissible_energy_variation = 0.0005
    penalty_coefficient = 1.0
    
    # First, run initial flattening to get initial 2D positions
    vertices_2d_initial = initial_flattening(
        mesh,
        spring_constant,
        area_density,
        dt,
        permissible_area_error,
        permissible_shape_error,
        permissible_energy_variation,
        penalty_coefficient,
        False,  # Disable energy release in initial flattening
        1
    )
    
    # Run energy release
    vertices_2d, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = energy_release(
        mesh.vertices,
        mesh.edges_unique,
        mesh.faces,
        vertices_2d_initial,
        spring_constant,
        area_density,
        dt,
        max_iterations,
        permissible_area_error,
        permissible_shape_error,
        permissible_energy_variation,
        penalty_coefficient
    )
    
    # Check that the result has the correct shape
    assert vertices_2d.shape == (len(mesh.vertices), 2)
    
    # Check that all vertices are flattened (have 2D coordinates)
    assert not np.any(np.isnan(vertices_2d))
    
    # Verify that output lists have the expected lengths
    assert len(area_errors) == max_iterations + 1
    assert len(shape_errors) == max_iterations + 1
    assert len(max_forces) == max_iterations + 1
    assert len(energies) == max_iterations + 1
    assert len(max_displacements) == max_iterations + 1
    assert len(max_penalty_displacements) == max_iterations + 1


def test_surface_flattening_spring_mass():
    """Test the full surface_flattening_spring_mass function."""
    # Create a test mesh
    mesh = create_test_mesh()
    
    # Parameters
    spring_constant = 0.5
    dt = 0.001
    max_iterations = 10
    permissible_area_error = 0.01
    permissible_shape_error = 0.01
    permissible_energy_variation = 0.0005
    penalty_coefficient = 1.0
    
    # Run full flattening process
    vertices_2d, vertices_2d_initial, area_errors, shape_errors, max_forces, energies, max_displacements, max_penalty_displacements = surface_flattening_spring_mass(
        mesh,
        spring_constant,
        dt,
        max_iterations,
        permissible_area_error,
        permissible_shape_error,
        permissible_energy_variation,
        penalty_coefficient,
        False,  # Disable energy release in initial flattening
        1,
        True  # Enable energy release phase
    )
    
    # Check that the results have the correct shapes
    assert vertices_2d.shape == (len(mesh.vertices), 2)
    assert vertices_2d_initial.shape == (len(mesh.vertices), 2)
    
    # Check that all vertices are flattened (have 2D coordinates)
    assert not np.any(np.isnan(vertices_2d))
    assert not np.any(np.isnan(vertices_2d_initial))
    
    # Verify that output lists have the expected lengths
    assert len(area_errors) == max_iterations + 1
    assert len(shape_errors) == max_iterations + 1
    assert len(max_forces) == max_iterations + 1
    assert len(energies) == max_iterations + 1
    assert len(max_displacements) == max_iterations + 1
    assert len(max_penalty_displacements) == max_iterations + 1


if __name__ == "__main__":
    # Run tests directly
    test_initial_flattening()
    test_energy_release()
    test_surface_flattening_spring_mass()
    print("All tests passed!")