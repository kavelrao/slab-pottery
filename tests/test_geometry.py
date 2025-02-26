"""
Test script to verify the geometry functions in the flattening module.
"""

import os
import sys
import numpy as np
import pytest

# Add the parent directory to the Python path so we can import the flattening module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flattening.geometry import (
    point_to_segment_distance_2d,
    point_to_segment_distance_3d,
    place_initial_triangle,
    calculate_vertex_areas,
    build_face_adjacency
)


def test_point_to_segment_distance_2d():
    """Test the point_to_segment_distance_2d function."""
    # Test case 1: Point is closest to the middle of the segment
    point = np.array([1.0, 1.0])
    segment_start = np.array([0.0, 0.0])
    segment_end = np.array([2.0, 0.0])
    distance, vector = point_to_segment_distance_2d(point, segment_start, segment_end)
    assert np.isclose(distance, 1.0)
    assert np.allclose(vector, np.array([0.0, -1.0]))
    
    # Test case 2: Point is closest to the start of the segment
    point = np.array([-1.0, 1.0])
    segment_start = np.array([0.0, 0.0])
    segment_end = np.array([2.0, 0.0])
    distance, vector = point_to_segment_distance_2d(point, segment_start, segment_end)
    assert np.isclose(distance, np.sqrt(2))
    assert np.allclose(vector, np.array([1.0, -1.0]))
    
    # Test case 3: Point is closest to the end of the segment
    point = np.array([3.0, 1.0])
    segment_start = np.array([0.0, 0.0])
    segment_end = np.array([2.0, 0.0])
    distance, vector = point_to_segment_distance_2d(point, segment_start, segment_end)
    assert np.isclose(distance, np.sqrt(2))
    assert np.allclose(vector, np.array([-1.0, -1.0]))
    
    # Test case 4: Zero-length segment
    point = np.array([1.0, 1.0])
    segment_start = np.array([0.0, 0.0])
    segment_end = np.array([0.0, 0.0])
    distance, vector = point_to_segment_distance_2d(point, segment_start, segment_end)
    assert np.isclose(distance, np.sqrt(2))
    assert np.allclose(vector, np.array([-1.0, -1.0]))


def test_point_to_segment_distance_3d():
    """Test the point_to_segment_distance_3d function."""
    # Test case 1: Point is closest to the middle of the segment
    point = np.array([1.0, 1.0, 1.0])
    segment_start = np.array([0.0, 0.0, 1.0])
    segment_end = np.array([2.0, 0.0, 1.0])
    distance, vector = point_to_segment_distance_3d(point, segment_start, segment_end)
    assert np.isclose(distance, 1.0)
    assert np.allclose(vector, np.array([0.0, -1.0, 0.0]))
    
    # Test case 2: Point is closest to the start of the segment
    point = np.array([-1.0, 1.0, 1.0])
    segment_start = np.array([0.0, 0.0, 1.0])
    segment_end = np.array([2.0, 0.0, 1.0])
    distance, vector = point_to_segment_distance_3d(point, segment_start, segment_end)
    assert np.isclose(distance, np.sqrt(2))
    assert np.allclose(vector, np.array([1.0, -1.0, 0.0]))
    
    # Test case 3: Point is closest to the end of the segment
    point = np.array([3.0, 1.0, 1.0])
    segment_start = np.array([0.0, 0.0, 1.0])
    segment_end = np.array([2.0, 0.0, 1.0])
    distance, vector = point_to_segment_distance_3d(point, segment_start, segment_end)
    assert np.isclose(distance, np.sqrt(2))
    assert np.allclose(vector, np.array([-1.0, -1.0, 0.0]))
    
    # Test case 4: Zero-length segment
    point = np.array([1.0, 1.0, 1.0])
    segment_start = np.array([0.0, 0.0, 1.0])
    segment_end = np.array([0.0, 0.0, 1.0])
    distance, vector = point_to_segment_distance_3d(point, segment_start, segment_end)
    assert np.isclose(distance, np.sqrt(2))
    assert np.allclose(vector, np.array([-1.0, -1.0, 0.0]))


def test_place_initial_triangle():
    """Test the place_initial_triangle function."""
    # Create a simple equilateral triangle in 3D
    side_length = 2.0
    height = side_length * np.sqrt(3) / 2
    vertices_3d = np.array([
        [0.0, 0.0, 0.0],
        [side_length, 0.0, 0.0],
        [side_length/2, height, 0.0]
    ])
    face_indices = np.array([0, 1, 2])
    
    # Place it in 2D
    vertices_2d = place_initial_triangle(vertices_3d, face_indices)
    
    # Check that it preserves edge lengths
    # Edge 0-1
    length_3d_01 = np.linalg.norm(vertices_3d[1] - vertices_3d[0])
    length_2d_01 = np.linalg.norm(vertices_2d[1] - vertices_2d[0])
    assert np.isclose(length_2d_01, length_3d_01)
    
    # Edge 0-2
    length_3d_02 = np.linalg.norm(vertices_3d[2] - vertices_3d[0])
    length_2d_02 = np.linalg.norm(vertices_2d[2] - vertices_2d[0])
    assert np.isclose(length_2d_02, length_3d_02)
    
    # Edge 1-2
    length_3d_12 = np.linalg.norm(vertices_3d[2] - vertices_3d[1])
    length_2d_12 = np.linalg.norm(vertices_2d[2] - vertices_2d[1])
    assert np.isclose(length_2d_12, length_3d_12)
    
    # Check that first vertex is at origin
    assert np.allclose(vertices_2d[0], np.array([0.0, 0.0]))
    
    # Check that second vertex is on x-axis
    assert np.isclose(vertices_2d[1][1], 0.0)
    assert np.isclose(vertices_2d[1][0], side_length)


def test_build_face_adjacency():
    """Test the build_face_adjacency function."""
    # Simple cube-like mesh with 8 vertices and 12 triangular faces
    faces = np.array([
        [0, 1, 2],  # Face 0
        [0, 2, 3],  # Face 1 - shares edge with face 0
        [4, 5, 6],  # Face 2
        [4, 6, 7],  # Face 3 - shares edge with face 2
        [0, 1, 4],  # Face 4 - shares edge with face 0
        [1, 4, 5],  # Face 5 - shares edge with face 4 and face 2
    ])
    
    adjacency = build_face_adjacency(faces)
    
    # Check adjacency lists
    assert set(adjacency[0]) == {1, 4}, f"Expected {0} to be adjacent to {1, 4}, got {adjacency[0]}"
    assert set(adjacency[1]) == {0}, f"Expected {1} to be adjacent to {0}, got {adjacency[1]}"
    assert set(adjacency[2]) == {3, 5}, f"Expected {2} to be adjacent to {3, 5}, got {adjacency[2]}"
    assert set(adjacency[3]) == {2}, f"Expected {3} to be adjacent to {2}, got {adjacency[3]}"
    assert set(adjacency[4]) == {0, 5}, f"Expected {4} to be adjacent to {0, 5}, got {adjacency[4]}"
    assert set(adjacency[5]) == {2, 4}, f"Expected {5} to be adjacent to {2, 4}, got {adjacency[5]}"


def test_calculate_vertex_areas():
    """Test the calculate_vertex_areas function."""
    # Define a simple mesh with 4 vertices and 2 triangular faces
    # The mesh is a square divided into two triangles
    vertices = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [1.0, 1.0, 0.0],  # Vertex 2
        [0.0, 1.0, 0.0],  # Vertex 3
    ])
    
    faces = np.array([
        [0, 1, 2],  # Face 0 (Triangle 0-1-2)
        [0, 2, 3],  # Face 1 (Triangle 0-2-3)
    ])
    
    # Calculate vertex areas
    vertex_areas = calculate_vertex_areas(vertices, faces)
    
    # Each face has area 0.5, so total mesh area is 1.0
    # Each vertex participates in different numbers of faces:
    # Vertex 0: 2 faces
    # Vertex 1: 1 face
    # Vertex 2: 2 faces
    # Vertex 3: 1 face
    
    # Each vertex gets 1/3 of the area of each face it belongs to
    # Expected areas:
    # Vertex 0: (0.5 + 0.5) / 3 = 1/3
    # Vertex 1: 0.5 / 3 = 1/6
    # Vertex 2: (0.5 + 0.5) / 3 = 1/3
    # Vertex 3: 0.5 / 3 = 1/6
    
    expected_areas = np.array([1/3, 1/6, 1/3, 1/6])
    assert np.allclose(vertex_areas, expected_areas)


if __name__ == "__main__":
    # Run tests directly
    test_point_to_segment_distance_2d()
    test_point_to_segment_distance_3d()
    test_place_initial_triangle()
    test_build_face_adjacency()
    test_calculate_vertex_areas()
    print("All tests passed!")