from pathlib import Path
import numpy as np
import trimesh
from typing import List, Tuple, Set, Dict, Optional, Union
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

from data_types import Mesh3d
from join_angle_identification import segment_mesh_face_normals


def extract_mesh_regions_with_thickness(
    mesh: trimesh.Trimesh,
    region_pairs: List[Tuple[int, int]],
    regions: Optional[List[Set[int]]] = None
) -> Mesh3d:
    """
    Creates a new mesh that includes only the outer regions while preserving
    thickness information from the corresponding inner regions.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to extract regions from.
    region_pairs : List[Tuple[int, int]]
        List of tuples where each tuple contains (outer_region_idx, inner_region_idx).
    regions : Optional[List[Set[int]]], default=None
        Pre-calculated regions as sets of face indices.
    
    Returns
    -------
    Mesh3d
        A new mesh containing only the outer regions with thickness data.
    """
    # If regions aren't provided, compute them
    if regions is None:
        regions = segment_mesh_face_normals(mesh)
    
    # Extract just the outer regions
    outer_region_indices = [pair[0] for pair in region_pairs]
    outer_mesh = extract_mesh_regions(mesh, region_indices=outer_region_indices, regions=regions)
    
    # Set up thickness calculation
    thickness_data = np.zeros(len(outer_mesh.vertices))
    vertex_mapping = {}
    
    # Create mapping from original mesh vertices to new mesh vertices
    for new_idx, vertex in enumerate(outer_mesh.vertices):
        dists = np.linalg.norm(mesh.vertices - vertex, axis=1)
        orig_idx = np.argmin(dists)
        vertex_mapping[orig_idx] = new_idx
    
    # Process each region pair
    for outer_idx, inner_idx in region_pairs:
        # Get vertices for outer region
        outer_region_vertices = set()
        for face_idx in regions[outer_idx]:
            for vertex_idx in mesh.faces[face_idx]:
                outer_region_vertices.add(vertex_idx)
        
        # Get vertices for inner region
        inner_region_vertices = set()
        for face_idx in regions[inner_idx]:
            for vertex_idx in mesh.faces[face_idx]:
                inner_region_vertices.add(vertex_idx)
        
        # Get coordinates for inner vertices and build KD-tree
        inner_vertices_coords = np.array([mesh.vertices[idx] for idx in inner_region_vertices])
        
        if len(inner_vertices_coords) == 0:
            print(f"Warning: No inner vertices found for region {inner_idx}")
            continue
            
        inner_kdtree = cKDTree(inner_vertices_coords)
        
        # Calculate thickness for each outer vertex
        for orig_vertex_idx in outer_region_vertices:
            if orig_vertex_idx not in vertex_mapping:
                continue
                
            new_vertex_idx = vertex_mapping[orig_vertex_idx]
            vertex_coord = mesh.vertices[orig_vertex_idx]
            
            # Use KD-tree for efficient minimum distance calculation
            distance, _ = inner_kdtree.query(vertex_coord, k=1)
            thickness_data[new_vertex_idx] = distance
    
    # Add thickness data to the mesh
    if not hasattr(outer_mesh, 'vertex_attributes'):
        outer_mesh.vertex_attributes = {}
    outer_mesh.vertex_attributes['thickness'] = thickness_data
    
    return outer_mesh


def extract_mesh_regions(
    mesh: trimesh.Trimesh, 
    region_indices: List[int], 
    regions: Optional[List[Set[int]]] = None
) -> Mesh3d:
    """
    Creates a new mesh from specified regions in the original mesh.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to extract regions from.
    region_indices : List[int]
        List of region indices to extract.
    regions : Optional[List[Set[int]]], default=None
        Pre-calculated regions as sets of face indices.
    
    Returns
    -------
    Mesh3d
        A new mesh containing only the specified regions.
    """
    # If regions aren't provided, compute them
    if regions is None:
        regions = segment_mesh_face_normals(mesh)
    
    # Get all faces from the specified regions
    selected_faces = set()
    for idx in region_indices:
        selected_faces.update(regions[idx])
    
    # Create a submesh with only those faces
    sub_mesh = mesh.submesh([list(selected_faces)], append=True)
    
    # Create a Mesh3d object
    result = Mesh3d(
        vertices=sub_mesh.vertices,
        edges=sub_mesh.edges_unique,
        faces=sub_mesh.faces
    )
    
    # Copy face normals if available
    if hasattr(sub_mesh, 'face_normals'):
        result.face_normals = sub_mesh.face_normals
    
    # Initialize metadata
    result.metadata = {}
    
    return result


def visualize_mesh_thickness(
    mesh: Mesh3d, 
    ax=None, 
    alpha=1.0, 
    show_edges=False, 
    cmap='viridis',
    title="Mesh Surface with Thickness Visualization"
):
    """
    Creates a visualization of the mesh with surface colored by thickness.
    
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
        thickness = mesh.vertex_attributes['thickness']
        
        # Create colormap and map thickness to vertex colors
        colormap = get_cmap(cmap)
        norm = Normalize(vmin=thickness.min(), vmax=thickness.max())
        vertex_colors = colormap(norm(thickness))
        
        # Calculate face colors by averaging vertex colors
        face_colors = np.zeros((len(mesh.faces), 4))
        for i, face in enumerate(mesh.faces):
            face_colors[i, :] = np.mean(vertex_colors[face], axis=0)
        
        # Try primary visualization method
        try:
            ax.plot_trisurf(
                mesh.vertices[:, 0], 
                mesh.vertices[:, 1], 
                mesh.vertices[:, 2],
                triangles=mesh.faces, 
                facecolors=face_colors,
                edgecolor='black' if show_edges else None,
                linewidth=0.2 if show_edges else 0,
                alpha=alpha
            )
        except Exception as e:
            print(f"Primary visualization failed: {e}")
            # Try fallback method
            try:
                ax.plot_trisurf(
                    mesh.vertices[:, 0], 
                    mesh.vertices[:, 1], 
                    mesh.vertices[:, 2],
                    triangles=mesh.faces, 
                    cmap=colormap,
                    array=thickness,
                    edgecolor='black' if show_edges else None,
                    linewidth=0.2 if show_edges else 0,
                    alpha=alpha
                )
            except Exception as e2:
                print(f"Fallback visualization failed: {e2}")
                # Basic visualization as last resort
                ax.plot_trisurf(
                    mesh.vertices[:, 0], 
                    mesh.vertices[:, 1], 
                    mesh.vertices[:, 2],
                    triangles=mesh.faces, 
                    color='lightgray',
                    edgecolor='black',
                    linewidth=0.2,
                    alpha=alpha
                )
        
        # Add colorbar
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="Thickness (distance units)")
        
        # Add thickness range label
        cbar.ax.text(
            0.5, -0.1, 
            f"Range: {thickness.min():.3f} to {thickness.max():.3f} units",
            transform=cbar.ax.transAxes,
            ha='center', va='top'
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
    
    thickness_data = mesh.vertex_attributes['thickness']
    
    print(f"Mesh thickness statistics:")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"  Min thickness: {thickness_data.min():.4f}")
    print(f"  Max thickness: {thickness_data.max():.4f}")
    print(f"  Mean thickness: {thickness_data.mean():.4f}")
    print(f"  Median thickness: {np.median(thickness_data):.4f}")
    
    # Calculate histogram
    hist, bins = np.histogram(thickness_data, bins=10)
    print(f"\nThickness distribution:")
    for i in range(len(hist)):
        print(f"  {bins[i]:.4f} to {bins[i+1]:.4f}: {hist[i]} vertices")


if __name__ == '__main__':
    """Main function to process and visualize mesh thickness."""
    # Load mesh
    mesh_file = Path(__file__).parent.parent / "files" / "Mug_w_Thickness.stl"
    mesh = trimesh.load(mesh_file)
    
    # Segment mesh into regions
    regions = segment_mesh_face_normals(mesh)
    
    # Define outer-inner region pairs
    region_pairs = [(1, 0), (4, 3)]
    
    # Extract mesh with thickness information
    thickness_mesh = extract_mesh_regions_with_thickness(mesh, region_pairs, regions)
    
    # Display statistics
    print_thickness_statistics(thickness_mesh)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Main view
    ax = fig.add_subplot(111, projection='3d')
    visualize_mesh_thickness(
        thickness_mesh, 
        ax=ax, 
        alpha=1.0, 
        show_edges=True,
        cmap='viridis',
        title="Mesh Surface Thickness Visualization"
    )
    
    plt.tight_layout()
    plt.show()
