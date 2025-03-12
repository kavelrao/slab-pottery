from pathlib import Path
import numpy as np
import trimesh
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

from data_types import Mesh3d


def extract_mesh_regions_with_thickness_and_bevel_angles(
    mesh: trimesh.Trimesh,
    region_pairs: List[Tuple[int, int]],
    regions: List[List[int]]
) -> Tuple[Dict[int, trimesh.Trimesh], Dict[int, Dict[int, float]]]:
    """
    Creates new meshes for each of the specified outer regions while preserving
    thickness information from the corresponding inner regions and detecting beveled edges.
    Bevel angles are rounded to the nearest multiple of 5 degrees.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to extract regions from.
    region_pairs : List[Tuple[int, int]]
        List of tuples where each tuple contains (outer_region_idx, inner_region_idx).
    regions : List[List[int]]
        List of lists, where each inner list contains face indices for a region.
    
    Returns
    -------
    Tuple[Dict[int, trimesh.Trimesh], Dict[int, Dict[int, float]]]
        First item: Dictionary mapping region index to its corresponding mesh with thickness data.
        Second item: Dictionary mapping region index to a dictionary of edge indices and their bevel angles.
    """
    # Helper function to round to nearest multiple of 5
    def round_to_nearest_5(value):
        return round(value / 5) * 5
    
    # Extract the outer regions
    outer_region_indices = [pair[0] for pair in region_pairs]
    
    # Dictionary to store extracted region meshes and their vertex mappings
    region_meshes = {}
    region_vertex_maps = {}  # Maps from new vertex indices to original vertex indices
    region_face_maps = {}    # Maps from new face indices to original face indices
    
    # Process each outer region to extract meshes
    for region_idx, face_indices in enumerate(regions):
        # Skip if this region is not an outer region
        if region_idx not in outer_region_indices:
            continue
            
        # Convert to numpy array if it's not already
        face_indices = np.array(face_indices, dtype=int)
        
        # Skip if no faces in this region
        if len(face_indices) == 0:
            print(f"Warning: No faces in region {region_idx}")
            continue
            
        # Extract faces for this region
        faces = mesh.faces[face_indices]
        
        # Create a mapping from old vertex indices to new ones
        # This is to remove unused vertices and create a compact mesh
        unique_vertices = np.unique(faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        vertex_map_inverse = {new_idx: old_idx for old_idx, new_idx in vertex_map.items()}
        
        # Create the new faces array with remapped vertex indices
        new_faces = np.zeros_like(faces)
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                new_faces[i, j] = vertex_map[faces[i, j]]
        
        # Create the new vertices array with only the used vertices
        new_vertices = mesh.vertices[unique_vertices]
        
        # Create the new mesh for this region
        region_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        
        # Copy applicable attributes from the original mesh
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            region_mesh.visual.face_colors = mesh.visual.face_colors[face_indices]
        
        # Find corresponding inner region for thickness calculation
        inner_idx = None
        for outer, inner in region_pairs:
            if outer == region_idx:
                inner_idx = inner
                break
                
        if inner_idx is not None:
            # Get vertices for inner region
            inner_region_vertices = set()
            for face_idx in regions[inner_idx]:
                for vertex_idx in mesh.faces[face_idx]:
                    inner_region_vertices.add(vertex_idx)
            
            # Get coordinates for inner vertices and build KD-tree
            inner_vertices_coords = np.array([mesh.vertices[idx] for idx in inner_region_vertices])
            
            if len(inner_vertices_coords) == 0:
                print(f"Warning: No inner vertices found for region {inner_idx}")
            else:
                inner_kdtree = cKDTree(inner_vertices_coords)
                
                # Calculate thickness for each vertex in the new mesh
                # But we'll only store the average instead of per-vertex values
                total_thickness = 0.0
                valid_vertices = 0
                
                for new_vertex_idx in range(len(region_mesh.vertices)):
                    # Get the original vertex index
                    orig_vertex_idx = vertex_map_inverse[new_vertex_idx]
                    vertex_coord = mesh.vertices[orig_vertex_idx]
                    
                    # Use KD-tree for efficient minimum distance calculation
                    distance, _ = inner_kdtree.query(vertex_coord, k=1)
                    total_thickness += distance
                    valid_vertices += 1
                
                # Calculate average thickness and round to nearest 10th
                if valid_vertices > 0:
                    average_thickness = round(total_thickness / valid_vertices, 1)
                else:
                    average_thickness = 0.0
                    
                # Add average thickness data to the mesh
                if not hasattr(region_mesh, 'vertex_attributes'):
                    region_mesh.vertex_attributes = {}
                    
                # Store a single average thickness value instead of per-vertex values
                region_mesh.vertex_attributes['thickness'] = average_thickness
        
        # Store the mesh, vertex map, and a mapping from new face indices to original face indices
        region_meshes[region_idx] = region_mesh
        region_vertex_maps[region_idx] = vertex_map_inverse
        region_face_maps[region_idx] = {i: face_indices[i] for i in range(len(face_indices))}
    
    # Now perform join analysis on the extracted meshes
    region_bevel_angles = {}
    
    # Initialize bevel angles for each region
    for region_idx in region_meshes:
        region_bevel_angles[region_idx] = {}
    
    # For each pair of outer region meshes, analyze joins between them
    outer_region_indices = list(region_meshes.keys())
    for i, region_idx1 in enumerate(outer_region_indices):
        for region_idx2 in outer_region_indices[i+1:]:
            region_mesh1 = region_meshes[region_idx1]
            region_mesh2 = region_meshes[region_idx2]
            
            # Get original mesh vertices for each region's vertices
            orig_vertices1 = {region_vertex_maps[region_idx1][v]: v for v in range(len(region_mesh1.vertices))}
            orig_vertices2 = {region_vertex_maps[region_idx2][v]: v for v in range(len(region_mesh2.vertices))}
            
            # Find shared original vertices
            shared_orig_vertices = set(orig_vertices1.keys()) & set(orig_vertices2.keys())
            
            # If no shared vertices, these regions don't join
            if not shared_orig_vertices:
                continue
            
            # Find edges in both meshes that use shared vertices
            edges1 = []
            for edge_idx, (v1, v2) in enumerate(region_mesh1.edges):
                orig_v1 = region_vertex_maps[region_idx1][v1]
                orig_v2 = region_vertex_maps[region_idx1][v2]
                if orig_v1 in shared_orig_vertices or orig_v2 in shared_orig_vertices:
                    edges1.append(edge_idx)
            
            edges2 = []
            for edge_idx, (v1, v2) in enumerate(region_mesh2.edges):
                orig_v1 = region_vertex_maps[region_idx2][v1]
                orig_v2 = region_vertex_maps[region_idx2][v2]
                if orig_v1 in shared_orig_vertices or orig_v2 in shared_orig_vertices:
                    edges2.append(edge_idx)
            
            # If no potential join edges in either mesh, continue
            if not edges1 or not edges2:
                continue
            
            # Get face normals for both regions
            if not hasattr(region_mesh1, 'face_normals'):
                region_mesh1.compute_face_normals()
            if not hasattr(region_mesh2, 'face_normals'):
                region_mesh2.compute_face_normals()
            
            # For each potential join edge in region1, find corresponding edge in region2
            for edge1_idx in edges1:
                v1, v2 = region_mesh1.edges[edge1_idx]
                orig_v1 = region_vertex_maps[region_idx1][v1]
                orig_v2 = region_vertex_maps[region_idx1][v2]
                
                # Find matching edge in region2
                for edge2_idx in edges2:
                    v3, v4 = region_mesh2.edges[edge2_idx]
                    orig_v3 = region_vertex_maps[region_idx2][v3]
                    orig_v4 = region_vertex_maps[region_idx2][v4]
                    
                    # Check if these edges share both vertices
                    if (orig_v1 == orig_v3 and orig_v2 == orig_v4) or (orig_v1 == orig_v4 and orig_v2 == orig_v3):
                        # These edges are the same in the original mesh - we've found a join!
                        
                        # Find faces connected to these edges
                        faces1_for_edge = []
                        for face_idx, face in enumerate(region_mesh1.faces):
                            if v1 in face and v2 in face:
                                faces1_for_edge.append(face_idx)
                        
                        faces2_for_edge = []
                        for face_idx, face in enumerate(region_mesh2.faces):
                            if v3 in face and v4 in face:
                                faces2_for_edge.append(face_idx)
                        
                        # If both edges have connected faces, calculate angle
                        if faces1_for_edge and faces2_for_edge:
                            # Use first face from each edge for angle calculation
                            face1_normal = region_mesh1.face_normals[faces1_for_edge[0]]
                            face2_normal = region_mesh2.face_normals[faces2_for_edge[0]]
                            
                            # Calculate angle between face normals
                            dot_product = np.clip(np.dot(face1_normal, face2_normal), -1.0, 1.0)
                            angle_rad = np.arccos(dot_product)
                            angle_deg = np.degrees(angle_rad)
                            
                            # Calculate smaller angle (either angle or 180-angle)
                            join_angle = min(angle_deg, 180 - angle_deg)
                            
                            # Calculate bevel angle (half of join angle) and round to nearest multiple of 5
                            bevel_angle = round_to_nearest_5(join_angle / 2.0)
                            
                            # Store bevel angles for both regions' edges
                            region_bevel_angles[region_idx1][edge1_idx] = bevel_angle
                            region_bevel_angles[region_idx2][edge2_idx] = bevel_angle
    
    return region_meshes, region_bevel_angles


def identify_join_edges(mesh, region1_set, region2_set):
    """
    Identifies edges that form joins between two regions of faces in a 3D mesh.
    
    Parameters
    ----------
    mesh : Object with vertices, edges, and faces attributes
        A mesh object.
    
    region1_set : set
        A set of face indices that belong to the first region.
    
    region2_set : set
        A set of face indices that belong to the second region.
    
    Returns
    -------
    list
        A list of edge indices that form joins between region1 and region2.
    """
    assert len(region1_set.intersection(region2_set)) == 0, "Regions must be disjoint sets"

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
            has_region1_face = any(face_idx in region1_set for face_idx in faces)
            has_region2_face = any(face_idx in region2_set for face_idx in faces)
            
            if has_region1_face and has_region2_face:
                join_edges.append(edge_idx)
    
    return join_edges


def visualize_mesh_thickness(
    mesh: Mesh3d, 
    ax=None, 
    alpha=1.0, 
    show_edges=False, 
    cmap='viridis',
    title="Mesh Surface with Thickness Visualization"
):
    """
    Creates a visualization of the mesh with surface colored by a uniform thickness value.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object with thickness data.
    ax : matplotlib.axes.Axes, optional
        A 3D axes object to plot on.
    alpha : float, optional
        Transparency level (0.0 to 1.0).
    show_edges : bool, optional
        Whether to show mesh edges.
    cmap : str, optional
        Colormap name for thickness visualization.
    title : str, optional
        Plot title.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The 3D axes containing the plot.
    """
    # Create figure and 3D axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Handle case with thickness data
    if hasattr(mesh, 'vertex_attributes') and 'thickness' in mesh.vertex_attributes:
        # Get the average thickness (now a single value instead of an array)
        thickness = mesh.vertex_attributes['thickness']
        
        # Plot mesh with uniform color based on the average thickness
        colormap = get_cmap(cmap)
        
        # For visualization purposes, create a color range
        color_val = 0.5  # Middle of the colormap
        mesh_color = colormap(color_val)
        
        # Plot the mesh with a uniform color
        ax.plot_trisurf(
            mesh.vertices[:, 0], 
            mesh.vertices[:, 1], 
            mesh.vertices[:, 2],
            triangles=mesh.faces, 
            color=mesh_color,
            edgecolor='black' if show_edges else None,
            linewidth=0.2 if show_edges else 0,
            alpha=alpha
        )
        
        # Add text annotation for the average thickness
        ax.text2D(
            0.05, 0.95, 
            f"Average Thickness: {thickness:.1f} inches",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    else:
        # Plot regular mesh if no thickness data
        ax.plot_trisurf(
            mesh.vertices[:, 0], 
            mesh.vertices[:, 1], 
            mesh.vertices[:, 2],
            triangles=mesh.faces, 
            color='lightgray', 
            edgecolor='black' if show_edges else None,
            linewidth=0.2 if show_edges else 0
        )
    
    # Set labels and properties
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set equal aspect ratio and adjust limits
    ax.set_box_aspect([1, 1, 1])
    
    x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    
    buffer = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.05
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.set_zlim(z_min - buffer, z_max + buffer)
    
    plt.tight_layout()
    return fig, ax


def print_thickness_statistics(mesh: Mesh3d):
    """
    Print statistics about mesh thickness data.
    
    Parameters
    ----------
    mesh : Mesh3d
        A mesh with thickness data in vertex_attributes.
    """
    if not hasattr(mesh, 'vertex_attributes') or 'thickness' not in mesh.vertex_attributes:
        print("No thickness data available")
        return
    
    thickness = mesh.vertex_attributes['thickness']
    
    print(f"Mesh thickness statistics:")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"  Average thickness: {thickness:.1f} inches")
