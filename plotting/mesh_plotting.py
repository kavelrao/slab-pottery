import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
import random

from data_types import Mesh3d


def plot_mesh(mesh: Mesh3d, title="3D Mesh", figsize=(12, 10), ax=None, alpha=0.7):
    # Create figure and 3D axis if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, cmap='viridis', edgecolor='black', alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_mesh_regions(mesh: Mesh3d, regions: list[NDArray[np.int64]], title="Mesh Regions", figsize=(12, 10),
                      region_colors=None, edge_color='black', edge_width=0.3,
                      alpha=0.7, with_edges=True, ax=None, region_labels=None):
    """
    Plots a 3D mesh with different regions colored distinctly.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
        Should have vertices (Vx3 array) and faces (Fx3 array).
    
    regions : list of NDArray[np.int64]
        List of arrays, where each array contains face indices for a region.
        Each face should belong to at most one region.
    
    title : str, optional
        Title for the plot. Default is "Mesh Regions".
    
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (12, 10).
    
    region_colors : list of colors, optional
        List of colors for each region. If None, colors are generated automatically.
        Must have at least as many colors as regions.
    
    edge_color : str or color, optional
        Color for the mesh edges. Default is 'black'.
    
    edge_width : float, optional
        Line width for mesh edges. Default is 0.3.
    
    alpha : float, optional
        Transparency of the mesh surfaces. Default is 0.8.
    
    with_edges : bool, optional
        Whether to display edges on the mesh. Default is True.
    
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes to plot on. If None, new figure and axes are created.

    region_labels : list of str, optional
        List of labels for each region. If None, "Region {i}" is used.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    
    ax : matplotlib.axes.Axes
        The 3D axes containing the plot.
    """
    # Create figure and 3D axis if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Generate colors if not provided
    if region_colors is None:
        # Create a colormap with distinct colors for each region
        # Avoid similar adjacent colors if possible
        cmap = plt.cm.get_cmap('tab20', max(20, len(regions)))
        region_colors = [cmap(i % 20) for i in range(len(regions))]
        
        # If more than 20 regions, add some random colors to ensure distinction
        if len(regions) > 20:
            for i in range(20, len(regions)):
                r, g, b = random.random(), random.random(), random.random()
                region_colors.append((r, g, b, 1.0))
    
    # Ensure enough colors are provided
    if len(region_colors) < len(regions):
        raise ValueError(f"Not enough colors provided. Need at least {len(regions)} colors.")

    if region_labels is None:
        region_labels = [f"Region {i}" for i in range(len(regions))]
    
    # Create a face color array initialized to a background color (for unassigned faces)
    # Use a light gray for unassigned faces
    background_color = (0.9, 0.9, 0.9, alpha)  # Light gray with alpha
    face_colors = np.full((len(mesh.faces), 4), background_color)
    
    # Assign colors to faces based on region
    for i, region in enumerate(regions):
        # For each face in this region, set its color
        for face_idx in region:
            # Make sure face_idx is valid
            if 0 <= face_idx < len(mesh.faces):
                # Set color with specified alpha
                color = region_colors[i]
                if len(color) == 3:  # RGB
                    face_colors[face_idx] = color + (alpha,)
                else:  # RGBA
                    face_colors[face_idx] = color[:3] + (alpha,)
    
    # Set edge parameters
    edgecolor = edge_color if with_edges else None
    linewidth = edge_width if with_edges else 0
    
    # Plot each region separately
    # First plot the base mesh with default coloring (this creates the structure)
    tri = ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                         triangles=mesh.faces, color='white', alpha=0.0)
    
    # Now add colored faces manually using Poly3DCollection
    mesh_triangles = mesh.vertices[mesh.faces]
    poly3d = Poly3DCollection(mesh_triangles, 
                             linewidths=linewidth,
                             edgecolors=edgecolor if with_edges else 'none')
    poly3d.set_facecolor(face_colors)
    ax.add_collection3d(poly3d)
    
    # Add a legend
    legend_elements = []
    for i, region_color in enumerate(region_colors[:len(regions)]):
        # Create a proxy artist for the legend
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=region_color, edgecolor=edge_color if with_edges else None,
                 label=region_labels[i]))
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, framealpha=0.7)
    
    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Attempt to set equal aspect ratio for a more realistic view
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


def plot_mesh_with_highlighted_edges(mesh, highlighted_edge_indices, title="Mesh with Highlighted Edges", 
                                      figsize=(10, 8), highlight_color='red', highlight_width=2, 
                                      mesh_alpha=0.7, mesh_cmap='viridis', ax=None):
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
    # Create figure and 3D axis if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot the mesh surface
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, cmap=mesh_cmap, edgecolor='black', alpha=mesh_alpha)
    
    # For few edges, use individual line plotting which can be more visible
    if len(highlighted_edge_indices) < 100:  # Threshold for switching methods
        # Plot each edge individually for better visibility
        for edge_idx in highlighted_edge_indices:
            # Get vertex indices for this edge
            v1_idx, v2_idx = mesh.edges[edge_idx]
            
            # Get the 3D coordinates of these vertices
            v1 = mesh.vertices[v1_idx]
            v2 = mesh.vertices[v2_idx]
            
            # Plot line with zorder to ensure it's on top
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                   color=highlight_color, linewidth=highlight_width, zorder=10)
    else:
        # For many edges, use Line3DCollection for better performance
        highlighted_lines = []
        for edge_idx in highlighted_edge_indices:
            # Get vertex indices for this edge
            v1_idx, v2_idx = mesh.edges[edge_idx]
            
            # Get the 3D coordinates of these vertices
            v1 = mesh.vertices[v1_idx]
            v2 = mesh.vertices[v2_idx]
            
            # Add line segment
            highlighted_lines.append([v1, v2])
            
        # Create a Line3DCollection with zorder to place it above the mesh
        if highlighted_lines:
            lc = Line3DCollection(highlighted_lines, colors=highlight_color, 
                                 linewidths=highlight_width, zorder=10)
            ax.add_collection(lc)
    
    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Attempt to set equal aspect ratio for a more realistic view
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
    
    # Adjust view angle slightly to make line visibility more consistent
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    return fig, ax


def plot_flat_mesh_with_node_energies(mesh, flattened_vertices_2d_initial, all_sampled_positions, all_energies, title="Initial Surface Flattening with Energies", save_png=True, mesh_name=''):
    # PLOT 3D MESH WITH ENERGIES
  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111)
  ax.set_aspect("equal")  
  face_verts_2d_initial = flattened_vertices_2d_initial[mesh.faces]
  poly_collection2 = PolyCollection(face_verts_2d_initial, facecolors="none", edgecolors="black", linewidths=0.5)
  ax.add_collection(poly_collection2)

  # Scatter plot of finer points with energy-based coloring
  sc = plt.scatter(all_sampled_positions[:, 0], all_sampled_positions[:, 1], c=all_energies, 
                 cmap=plt.cm.jet, s=40, edgecolors='k', linewidth=1.5)
  

  # Set plot limits for initial 2D plot
  min_coords = min(flattened_vertices_2d_initial[:, 0]), min(flattened_vertices_2d_initial[:, 1])
  max_coords = max(flattened_vertices_2d_initial[:, 0]), max(flattened_vertices_2d_initial[:, 1])
  range_x = max_coords[0] - min_coords[0]
  range_y = max_coords[1] - min_coords[1]
  padding_x = range_x * 0.1
  padding_y = range_y * 0.1
  ax.set_xlim(min_coords[0] - padding_x, max_coords[0] + padding_x)
  ax.set_ylim(min_coords[1] - padding_y, max_coords[1] + padding_y)
  ax.set_title(f)
  cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
  cbar.set_label('Energy Value')
  # Set axis labels and title
  ax.set_title(title)
  if save_png:
    plt.savefig(Path(__file__).parent / (mesh_name + "_energy_plot2d.npy"))
  else:
    plt.show()