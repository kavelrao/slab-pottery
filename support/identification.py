import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import trimesh
import matplotlib.pyplot as plt
import itertools

from data_types import Mesh3d
from segmenting import segment_mesh_face_normals
from plotting import plot_mesh_regions, plot_mesh_with_highlighted_edges
from deduplication import extract_mesh_regions


def identify_join_edges(mesh: Mesh3d, region1: set[int], region2: set[int]) -> list[int]:
    """
    Identifies edges that form joins between two regions of faces in a 3D mesh.
    
    This function finds all edges that serve as boundaries between two distinct regions
    of a mesh. These edges are critical for identifying areas that may require support
    tabs in applications like slab pottery construction.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
        Vertices is a V x 3 array of vertex coordinates.
        Edges is an E x 2 array of vertex indices representing edge endpoints.
        Faces is an F x 3 array of vertex indices representing face corners.
    
    region1 : set[int]
        A set of face indices that belong to the first region.
    
    region2 : set[int]
        A set of face indices that belong to the second region.
    
    Returns
    -------
    list[int]
        A list of edge indices that form joins between region1 and region2.
        Each index corresponds to a row in mesh.edges.
    
    Notes
    -----
    - Edges are considered join edges if they connect faces from different regions.
    - The function first constructs a mapping from edges to the faces they belong to,
      then identifies edges that have faces in both regions.
    - This is useful for structural analysis in pottery construction to determine
      where support tabs might be needed.
    """
    assert len(region1.intersection(region2)) == 0, "Regions must be disjoint sets"

    # Create a dictionary to map each edge to the faces it belongs to
    edge_to_faces = {}
    
    # For each face, add it to the mapping for each of its edges
    for face_idx, face in enumerate(mesh.faces):
        # Get the three edges of the triangle
        # We need to ensure consistent edge ordering for lookup
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0]))
        ]
        
        # Add this face to each edge's list of faces
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
    
    # Find edges that connect faces from region1 and region2
    join_edges = []
    
    # For each edge in the mesh
    for edge_idx, edge in enumerate(mesh.edges):
        # Create a sorted tuple for consistent lookup
        edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        
        # Check if this edge exists in our mapping
        if edge_key in edge_to_faces:
            faces = edge_to_faces[edge_key]
            
            # Check if this edge has faces in both regions
            has_region1_face = any(face_idx in region1 for face_idx in faces)
            has_region2_face = any(face_idx in region2 for face_idx in faces)
            
            if has_region1_face and has_region2_face:
                join_edges.append(edge_idx)
    
    return join_edges


if __name__ == '__main__':
    mesh = trimesh.load(Path(__file__).parent.parent / "files" / "Mug_Thick_Handle_Selected.stl")
    regions = segment_mesh_face_normals(mesh, angle_threshold=30)
    join_edges = []
    assert len(regions) > 1
    for region1, region2 in itertools.combinations(regions, r=2):
        join_edges += identify_join_edges(mesh, set(region1), set(region2))
    
    fig, ax = plot_mesh_with_highlighted_edges(mesh, join_edges)
    plt.show()
