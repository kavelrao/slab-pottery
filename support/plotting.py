import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Optional, Tuple, List, Dict


def plot_mesh_with_support_regions(mesh, support_regions, save_path=None):
    """
    Plot a 3D mesh with support regions highlighted in red and regular faces in blue.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The 3D mesh to visualize.
    support_regions : List[int]
        List of face indices for support regions.
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization.
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert support_regions to a set for faster lookup
    support_set = set(support_regions)
    
    # Create two collections: one for regular faces and one for support faces
    regular_faces = []
    support_faces = []
    
    # Sort faces into appropriate collections
    for i, face in enumerate(mesh.faces):
        face_vertices = mesh.vertices[face]
        if i in support_set:
            support_faces.append(face_vertices)
        else:
            regular_faces.append(face_vertices)
    
    # Add regular faces to the plot (blue)
    if regular_faces:
        regular_collection = Poly3DCollection(regular_faces, alpha=0.7)
        regular_collection.set_facecolor('blue')
        regular_collection.set_edgecolor('black')
        regular_collection.set_linewidth(0.2)
        ax.add_collection3d(regular_collection)
    
    # Add support faces to the plot (red)
    if support_faces:
        support_collection = Poly3DCollection(support_faces, alpha=0.8)
        support_collection.set_facecolor('red')
        support_collection.set_edgecolor('black')
        support_collection.set_linewidth(0.2)
        ax.add_collection3d(support_collection)
    
    # Set axis limits and labels
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a title that includes the number of support regions
    ax.set_title(f'Mesh Visualization with {len(support_regions)} Support Regions (Red)')
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Add a legend
    import matplotlib.patches as mpatches
    regular_patch = mpatches.Patch(color='blue', label='Regular Faces')
    support_patch = mpatches.Patch(color='red', label='Support Regions')
    ax.legend(handles=[regular_patch, support_patch], loc='upper right')
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_mesh_comparison(original_mesh: trimesh.Trimesh, relaxed_vertices: NDArray[np.float32]):
    """Plot the original mesh and the relaxed mesh side by side."""
    # Create a copy of the original mesh with the relaxed vertices
    relaxed_mesh = original_mesh.copy()
    relaxed_mesh.vertices = relaxed_vertices
    
    # Create the figure and subplots
    fig = plt.figure(figsize=(15, 7))
    
    # Plot original mesh
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Mesh')
    plot_mesh(original_mesh, ax1)
    
    # Plot relaxed mesh
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Relaxed Mesh')
    plot_mesh(relaxed_mesh, ax2)
    
    # Set the same scale for both plots
    x_limits = np.array([ax1.get_xlim(), ax2.get_xlim()])
    y_limits = np.array([ax1.get_ylim(), ax2.get_ylim()])
    z_limits = np.array([ax1.get_zlim(), ax2.get_zlim()])
    
    x_range = [np.min(x_limits), np.max(x_limits)]
    y_range = [np.min(y_limits), np.max(y_limits)]
    z_range = [np.min(z_limits), np.max(z_limits)]
    
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
    
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    ax2.set_zlim(z_range)
    
    plt.tight_layout()
    plt.show()


def plot_mesh(mesh: trimesh.Trimesh, ax):
    """Plot a mesh on the given axis."""
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Plot the mesh as a wireframe
    for face in faces:
        # Get the vertices of this face
        verts = vertices[face]
        
        # Connect the last vertex to the first one to close the face
        verts = np.vstack((verts, verts[0]))
        
        # Plot the face
        ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], 'k-', linewidth=0.5)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Make the plot more visually appealing
    ax.grid(True)


def plot_mesh_energy(vertices_3d: NDArray[np.float64], 
                    faces: NDArray[np.int64], 
                    vertex_energy: NDArray[np.float64],
                    edges: Optional[NDArray[np.int64]] = None,
                    title: str = "Mesh Energy Distribution",
                    cmap: str = "viridis",
                    show_colorbar: bool = True,
                    alpha: float = 0.7,
                    figsize: Tuple[int, int] = (10, 8),
                    view_angles: Tuple[float, float] = (30, 45),
                    annotate_vertices: bool = False,
                    energy_range: Optional[Tuple[float, float]] = None,
                    highlight_high_energy: bool = False,
                    high_energy_threshold: Optional[float] = None,
                    show_edges: bool = True,
                    edge_color: str = 'black',
                    edge_width: float = 0.5) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 3D mesh with vertices colored according to their energy values.
    
    Args:
        vertices_3d: 3D vertex positions (N x 3 array)
        faces: Face indices (M x 3 array for triangular mesh or M x 4 for quad mesh)
        vertex_energy: Energy value for each vertex (N array)
        edges: Optional edge indices (P x 2 array)
        title: Plot title
        cmap: Colormap name
        show_colorbar: Whether to display the colorbar
        alpha: Transparency of the mesh faces
        figsize: Figure size (width, height)
        view_angles: Initial view angles (elevation, azimuth)
        annotate_vertices: Whether to annotate vertex indices
        energy_range: Optional custom range for energy values (min, max)
        highlight_high_energy: Highlight vertices with high energy
        high_energy_threshold: Energy threshold for highlighting (if None, uses mean + std)
        show_edges: Whether to show mesh edges
        edge_color: Color for mesh edges
        edge_width: Width for mesh edges
        
    Returns:
        tuple: (figure, axes) for further customization
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set colormap normalization (either based on provided range or data)
    if energy_range is not None:
        norm = Normalize(energy_range[0], energy_range[1])
    else:
        norm = Normalize(vertex_energy.min(), vertex_energy.max())
    
    # Create the colormap
    cmap_obj = cm.get_cmap(cmap)
    
    # Plot faces with energy-based color
    for face in faces:
        # Get face vertices
        face_vertices = vertices_3d[face]
        
        # Get face vertex energies for average coloring
        face_energy = np.mean([vertex_energy[i] for i in face])
        
        # Create polygon and color it based on average face energy
        poly = Poly3DCollection([face_vertices], alpha=alpha)
        poly.set_facecolor(cmap_obj(norm(face_energy)))
        ax.add_collection3d(poly)
    
    # Plot edges if provided or requested
    if show_edges and edges is not None:
        for edge in edges:
            ax.plot(
                [vertices_3d[edge[0], 0], vertices_3d[edge[1], 0]],
                [vertices_3d[edge[0], 1], vertices_3d[edge[1], 1]],
                [vertices_3d[edge[0], 2], vertices_3d[edge[1], 2]],
                color=edge_color, linewidth=edge_width
            )
    
    # Highlight high-energy vertices if requested
    if highlight_high_energy:
        if high_energy_threshold is None:
            # Default threshold: mean + 1 standard deviation
            high_energy_threshold = np.mean(vertex_energy) + np.std(vertex_energy)
        
        # Find high energy vertices
        high_energy_indices = np.where(vertex_energy > high_energy_threshold)[0]
        
        if len(high_energy_indices) > 0:
            high_energy_vertices = vertices_3d[high_energy_indices]
            ax.scatter(
                high_energy_vertices[:, 0],
                high_energy_vertices[:, 1],
                high_energy_vertices[:, 2],
                color='red', s=50, marker='o', label=f'Energy > {high_energy_threshold:.2f}'
            )
            ax.legend()
    
    # Annotate vertices if requested
    if annotate_vertices:
        for i, (x, y, z) in enumerate(vertices_3d):
            ax.text(x, y, z, f"{i}", color='red', fontsize=8)
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, shrink=0.7)
        cbar.set_label('Energy')
    
    # Set plot properties
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set view angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust limits for a better view if needed
    x_lim = (vertices_3d[:, 0].min(), vertices_3d[:, 0].max())
    y_lim = (vertices_3d[:, 1].min(), vertices_3d[:, 1].max())
    z_lim = (vertices_3d[:, 2].min(), vertices_3d[:, 2].max())
    
    # Add a bit of padding
    padding = max(
        0.1 * (x_lim[1] - x_lim[0]),
        0.1 * (y_lim[1] - y_lim[0]),
        0.1 * (z_lim[1] - z_lim[0])
    )
    
    ax.set_xlim(x_lim[0] - padding, x_lim[1] + padding)
    ax.set_ylim(y_lim[0] - padding, y_lim[1] + padding)
    ax.set_zlim(z_lim[0] - padding, z_lim[1] + padding)
    
    plt.tight_layout()
    
    return fig, ax


def generate_energy_visualization_report(vertices_3d: NDArray[np.float64], 
                                       faces: NDArray[np.int64], 
                                       edges: NDArray[np.int64],
                                       vertex_energy: NDArray[np.float64],
                                       save_path: Optional[str] = None) -> None:
    """
    Generate a comprehensive report on mesh energy distribution.
    
    Args:
        vertices_3d: 3D vertex positions
        faces: Face indices
        edges: Edge indices
        masses: Masses of each vertex
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for energy calculation
        gravity: Acceleration due to gravity
        save_path: Optional path to save the figure
    """
    # Create multi-view figure
    fig = plt.figure(figsize=(18, 10))
    
    # Standard view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_standard(ax1, vertices_3d, faces, vertex_energy, edges)
    ax1.set_title("Energy Distribution - Standard View")
    ax1.view_init(elev=30, azim=45)
    
    # Alternative view
    ax2 = fig.add_subplot(122, projection='3d')
    plot_mesh_standard(ax2, vertices_3d, faces, vertex_energy, edges)
    ax2.set_title("Energy Distribution - Alternative View")
    ax2.view_init(elev=0, azim=90)
    
    # Add overall title
    plt.suptitle(f"Mesh Energy Analysis", fontsize=16)
    
    # Add energy statistics
    energy_stats = f"Energy Stats: Min={vertex_energy.min():.2f}, Max={vertex_energy.max():.2f}, Mean={np.mean(vertex_energy):.2f}, Std={np.std(vertex_energy):.2f}"
    fig.text(0.5, 0.01, energy_stats, ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_mesh_standard(ax, vertices, faces, energy, edges=None):
    """Helper function for the report plotting"""
    # Create the colormap normalization
    norm = Normalize(energy.min(), energy.max())
    cmap_obj = cm.get_cmap('viridis')
    
    # Plot faces
    for face in faces:
        face_vertices = vertices[face]
        face_energy = np.mean([energy[i] for i in face])
        poly = Poly3DCollection([face_vertices], alpha=0.7)
        poly.set_facecolor(cmap_obj(norm(face_energy)))
        ax.add_collection3d(poly)
    
    # Plot edges if provided
    if edges is not None:
        for edge in edges:
            ax.plot(
                [vertices[edge[0], 0], vertices[edge[1], 0]],
                [vertices[edge[0], 1], vertices[edge[1], 1]],
                [vertices[edge[0], 2], vertices[edge[1], 2]],
                color='black', linewidth=0.5
            )
    
    # Add colorbar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, shrink=0.7)
    cbar.set_label('Energy')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

