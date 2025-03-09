import numpy as np
import unittest
from dataclasses import dataclass
from numpy.typing import NDArray

from data_types import Mesh3d
from support.identification import identify_join_edges


class TestIdentifyJoinEdges(unittest.TestCase):
    def setUp(self):
        # Create a simple cube mesh (8 vertices, 12 edges, 12 triangular faces)
        self.vertices = np.array([
            [0, 0, 0],  # 0: front bottom left
            [1, 0, 0],  # 1: front bottom right
            [1, 1, 0],  # 2: front top right
            [0, 1, 0],  # 3: front top left
            [0, 0, 1],  # 4: back bottom left
            [1, 0, 1],  # 5: back bottom right
            [1, 1, 1],  # 6: back top right
            [0, 1, 1],  # 7: back top left
        ], dtype=np.float64)
        
        # Define edges (12 edges of a cube)
        self.edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # front face edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # back face edges
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting edges
        ], dtype=np.int64)
        
        # Define triangular faces (2 triangles per cube face = 12 triangles)
        self.faces = np.array([
            # Front face (2 triangles)
            [0, 1, 2], [0, 2, 3],
            # Back face (2 triangles)
            [4, 5, 6], [4, 6, 7],
            # Right face (2 triangles)
            [1, 5, 6], [1, 6, 2],
            # Left face (2 triangles)
            [0, 4, 7], [0, 7, 3],
            # Top face (2 triangles)
            [3, 2, 6], [3, 6, 7],
            # Bottom face (2 triangles)
            [0, 1, 5], [0, 5, 4],
        ], dtype=np.int64)
        
        self.mesh = Mesh3d(
            vertices=self.vertices,
            edges=self.edges,
            faces=self.faces
        )

    def test_identify_join_edges_adjacent_regions(self):
        """Test identifying join edges between two adjacent regions."""
        # Region 1: Front face (faces 0, 1)
        region1 = {0, 1}
        
        # Region 2: Right face (faces 4, 5)
        region2 = {4, 5}
        
        # The join edge should be edge 1: [1, 2]
        expected_join_edges = np.array([1], dtype=np.int64)
        
        join_edges = identify_join_edges(self.mesh, region1, region2)
        
        np.testing.assert_array_equal(join_edges, expected_join_edges)
        self.assertEqual(len(join_edges), 1, "Should find exactly one join edge")

    def test_identify_join_edges_non_adjacent_regions(self):
        """Test identifying join edges between non-adjacent regions."""
        # Region 1: Front face (faces 0, 1)
        region1 = {0, 1}
        
        # Region 2: Back face (faces 2, 3)
        region2 = {2, 3}
        
        # No join edges expected
        expected_join_edges = np.array([], dtype=np.int64)
        
        join_edges = identify_join_edges(self.mesh, region1, region2)
        
        np.testing.assert_array_equal(join_edges, expected_join_edges)
        self.assertEqual(len(join_edges), 0, "Should find no join edges")

    def test_identify_join_edges_multiple_joins(self):
        """Test identifying multiple join edges between two regions."""
        # Region 1: Front and right faces (faces 0, 1, 4, 5)
        region1 = {0, 1, 4, 5}
        
        # Region 2: Top face (faces 8, 9)
        region2 = {8, 9}
        
        # Expected join edges: edge 2 [2, 3] and edge 10 [2, 6]
        # Note: The actual expected indices might differ depending on how edges are defined
        expected_join_edges = np.array([2, 10], dtype=np.int64)
        
        join_edges = identify_join_edges(self.mesh, region1, region2)
        
        # Sort arrays for comparison since order doesn't matter
        np.testing.assert_array_equal(np.sort(join_edges), np.sort(expected_join_edges))
        self.assertEqual(len(join_edges), 2, "Should find exactly two join edges")

    def test_identify_join_edges_empty_region(self):
        """Test identifying join edges when one region is empty."""
        # Region 1: Front face (faces 0, 1)
        region1 = {0, 1}
        
        # Region 2: Empty
        region2 = set[int]()
        
        # No join edges expected
        expected_join_edges = np.array([], dtype=np.int64)
        
        join_edges = identify_join_edges(self.mesh, region1, region2)
        
        np.testing.assert_array_equal(join_edges, expected_join_edges)
        self.assertEqual(len(join_edges), 0, "Should find no join edges with empty region")


if __name__ == "__main__":
    unittest.main()