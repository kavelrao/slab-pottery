import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import itertools

from data_types import Mesh3d
from segmenting import segment_mesh_face_normals


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


def plot_mesh_with_highlighted_edges(mesh, highlighted_edge_indices, title="Mesh with Highlighted Edges", 
                                      figsize=(10, 8), highlight_color='red', highlight_width=3, 
                                      mesh_alpha=0.7, mesh_cmap='viridis'):
    """
    Plots a 3D mesh with specific edges highlighted in color.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
        Should have vertices (Vx3 array), edges (Ex2 array), and faces (Fx3 array).
    
    highlighted_edge_indices : NDArray[np.int64]
        Array of edge indices to highlight. Each index corresponds to a row in mesh.edges.
    
    title : str, optional
        Title for the plot. Default is "Mesh with Highlighted Edges".
    
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 8).
    
    highlight_color : str or color, optional
        Color for the highlighted edges. Default is 'red'.
    
    highlight_width : float, optional
        Line width for highlighted edges. Default is 3.
    
    mesh_alpha : float, optional
        Transparency of the mesh surface. Default is 0.7.
    
    mesh_cmap : str or colormap, optional
        Colormap for the mesh surface. Default is 'viridis'.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    
    ax : matplotlib.axes.Axes
        The 3D axes containing the plot.
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh surface
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                   triangles=mesh.faces, cmap=mesh_cmap, edgecolor='black', alpha=mesh_alpha)
    
    # Create line segments for the highlighted edges
    highlighted_lines = []
    for edge_idx in highlighted_edge_indices:
        # Get vertex indices for this edge
        v1_idx, v2_idx = mesh.edges[edge_idx]
        
        # Get the 3D coordinates of these vertices
        v1 = mesh.vertices[v1_idx]
        v2 = mesh.vertices[v2_idx]
        
        # Add line segment
        highlighted_lines.append([v1, v2])
    
    # Create a Line3DCollection for better performance with many lines
    if highlighted_lines:
        lc = Line3DCollection(highlighted_lines, colors=highlight_color, linewidths=highlight_width)
        ax.add_collection(lc)
    
    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Attempt to set equal aspect ratio for a more realistic view
    # This may not work perfectly in all matplotlib versions
    ax.set_box_aspect([1, 1, 1])
    
    # Auto-adjust limits to include all vertices
    x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    
    # Add a small buffer for better visualization
    buffer = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.05
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.set_zlim(z_min - buffer, z_max + buffer)
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    mesh = trimesh.load(Path(__file__).parent.parent / "files" / "Mug_Thick_Handle.stl")
    regions = segment_mesh_face_normals(mesh)
    join_edges = []
    assert len(regions) > 1
    for region1, region2 in itertools.combinations(regions, r=2):
        join_edges += identify_join_edges(mesh, set(region1), set(region2))
    
    fig, ax = plot_mesh_with_highlighted_edges(mesh, join_edges)
    plt.show()
