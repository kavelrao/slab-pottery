from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from data_types import Mesh3d
from segmenting import segment_mesh_face_normals
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
    """
    # Convert to sets if they aren't already
    region1_set = set(region1)
    region2_set = set(region2)
    
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


def calculate_region_join_angle(mesh: Mesh3d, region1: set[int], region2: set[int], join_edges: list[int]) -> float:
    """
    Calculates the angle between two regions of a mesh that share join edges.
    Handles edge cases where 90-degree joins might incorrectly calculate as 0 degrees.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
    
    region1 : set[int]
        A set of face indices that belong to the first region.
    
    region2 : set[int]
        A set of face indices that belong to the second region.
    
    join_edges : list[int]
        A list of edge indices that form joins between region1 and region2.
    
    Returns
    -------
    float
        The angle in degrees between the average normals of the two regions.
    """
    
    # Calculate face normals if not already available
    if not hasattr(mesh, 'face_normals'):
        # Compute face normals
        face_normals = np.zeros((len(mesh.faces), 3))
        
        for i, face in enumerate(mesh.faces):
            v0, v1, v2 = mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]
            # Calculate normal using cross product
            normal = np.cross(v1 - v0, v2 - v0)
            # Normalize the normal vector
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                normal = normal / normal_length
            face_normals[i] = normal
    else:
        face_normals = mesh.face_normals
    
    # First approach: Calculate angle using average normals of entire regions
    # Calculate average normal for region1
    region1_normals = np.array([face_normals[face_idx] for face_idx in region1])
    avg_normal1 = np.mean(region1_normals, axis=0)
    avg_normal1_length = np.linalg.norm(avg_normal1)
    if avg_normal1_length > 0:
        avg_normal1 = avg_normal1 / avg_normal1_length
    
    # Calculate average normal for region2
    region2_normals = np.array([face_normals[face_idx] for face_idx in region2])
    avg_normal2 = np.mean(region2_normals, axis=0)
    avg_normal2_length = np.linalg.norm(avg_normal2)
    if avg_normal2_length > 0:
        avg_normal2 = avg_normal2 / avg_normal2_length
    
    # Calculate the angle between the average normals (in radians)
    dot_product = np.clip(np.dot(avg_normal1, avg_normal2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    # Handle edge case where angle is very close to 0 or 180
    # This could indicate a genuine parallel surface OR an incorrect calculation
    if angle_deg < 1.0 or angle_deg > 179.0:
        # Second approach: Check angles along the join boundary
        # Create a mapping from edges to faces
        edge_to_faces = {}
        for face_idx, face in enumerate(mesh.faces):
            edges = [
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[2], face[0]), max(face[2], face[0]))
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Collect face pairs along the join
        boundary_angles = []
        for edge_idx in join_edges:
            # Convert edge index to edge key for dictionary lookup
            v1_idx, v2_idx = mesh.edges[edge_idx]
            edge_key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            
            # Get faces connected to this edge
            if edge_key in edge_to_faces and len(edge_to_faces[edge_key]) == 2:
                face1_idx, face2_idx = edge_to_faces[edge_key]
                
                # Make sure faces are from different regions
                if ((face1_idx in region1 and face2_idx in region2) or 
                    (face1_idx in region2 and face2_idx in region1)):
                    # Calculate angle between these specific faces
                    normal1 = face_normals[face1_idx]
                    normal2 = face_normals[face2_idx]
                    
                    face_dot = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                    face_angle = np.degrees(np.arccos(face_dot))
                    boundary_angles.append(min(face_angle, 180 - face_angle))
        
        # If we found boundary face pairs, use their average angle instead
        if boundary_angles:
            boundary_angle = np.mean(boundary_angles)
            
            # If boundary angle calculation suggests non-parallel surfaces,
            # use this more accurate measurement
            if boundary_angle > 10.0:  # Threshold to detect meaningful angles
                return boundary_angle
    
    # Return the smallest angle (either the angle or 180-angle)
    return min(angle_deg, 180 - angle_deg)


def analyze_mesh_joins(mesh: Mesh3d, regions: list) -> dict:
    """
    Analyzes joins between all pairs of regions in a mesh.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
    
    regions : list
        A list of collections (lists, arrays, or sets), where each contains face indices belonging to a region.
    
    Returns
    -------
    dict
        A dictionary where keys are region pair tuples (r1, r2) and values are dictionaries
        containing 'join_edges' (list of edge indices) and 'angle' (float, in degrees).
    """
    join_analysis = {}
    
    # Analyze each pair of regions
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i >= j:  # Skip duplicate pairs and self-comparisons
                continue
                
            # Find join edges between these regions
            join_edges = identify_join_edges(mesh, region1, region2)
            
            # If there are join edges, calculate the angle
            if join_edges:
                angle = calculate_region_join_angle(mesh, region1, region2, join_edges)
                
                # Store the results
                join_analysis[(i, j)] = {
                    'join_edges': join_edges,
                    'angle': angle,
                    'region1': region1,
                    'region2': region2
                }
    
    return join_analysis


def visualize_all_region_joins(mesh: Mesh3d, join_analysis: dict, colors=None):
    """
    Creates a visualization of all region joins in the mesh, each with a different color.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
    
    join_analysis : dict
        A dictionary of join analysis as returned by analyze_mesh_joins().
    
    colors : list, optional
        A list of colors to use for different joins. If None, uses a default color cycle.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    
    ax : matplotlib.axes.Axes
        The 3D axes containing the plot.
    """
    import matplotlib.cm as cm
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh surface with transparency
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                   triangles=mesh.faces, color='lightgray', edgecolor=None, alpha=0.3)
    
    # If no colors provided, create a color cycle
    if colors is None:
        colormap = cm.get_cmap('tab10')
        colors = [colormap(i % 10) for i in range(len(join_analysis))]
    
    # Create a legend handle list
    legend_handles = []
    
    # For each join, create a line collection with a unique color
    for i, ((r1_idx, r2_idx), data) in enumerate(join_analysis.items()):
        color = colors[i % len(colors)]
        
        # Create line segments for the join edges
        join_lines = []
        for edge_idx in data['join_edges']:
            v1_idx, v2_idx = mesh.edges[edge_idx]
            v1 = mesh.vertices[v1_idx]
            v2 = mesh.vertices[v2_idx]
            join_lines.append([v1, v2])
        
        # Create a Line3DCollection
        if join_lines:
            lc = Line3DCollection(join_lines, colors=color, linewidths=2)
            ax.add_collection(lc)
            
            # Create a dummy plot element for the legend
            legend_line = plt.Line2D([0], [0], color=color, linewidth=2, 
                                     label=f"Join {r1_idx}-{r2_idx}: {data['angle']:.1f}°")
            legend_handles.append(legend_line)
    
    # Add legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', title="Join Angles")
    
    # Set axis labels and title
    ax.set_title("Mesh with All Region Joins")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Auto-adjust limits
    x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    
    buffer = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.05
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.set_zlim(z_min - buffer, z_max + buffer)
    
    plt.tight_layout()
    return fig, ax


def calculate_bevel_angles(join_analysis: dict) -> dict:
    """
    Calculates the ideal bevel angle for each edge in the join analysis.
    
    The ideal bevel angle is half of the join angle, as this would allow the two
    beveled edges to fit perfectly together at the specified join angle.
    
    Parameters
    ----------
    join_analysis : dict
        A dictionary of join analysis as returned by analyze_mesh_joins().
        Keys are region pair tuples (r1, r2) and values are dictionaries
        containing 'join_edges' (list of edge indices) and 'angle' (float, in degrees).
    
    Returns
    -------
    dict
        A dictionary where keys are edge indices and values are the ideal bevel angles in degrees.
        Each edge will have a bevel angle that is half of the join angle of its regions.
    """
    # Initialize dictionary to store the bevel angle for each edge
    edge_bevel_angles = {}
    
    # Process each region join
    for _, data in join_analysis.items():
        # Calculate the ideal bevel angle (half of the join angle)
        bevel_angle = data['angle'] / 2.0
        
        # Assign this bevel angle to each edge in the join
        for edge_idx in data['join_edges']:
            edge_bevel_angles[edge_idx] = bevel_angle
    
    return edge_bevel_angles


def visualize_bevel_angles(mesh, edge_bevel_angles, ax=None):
    """
    Creates a visualization of the mesh with edges colored according to their bevel angles.
    
    Parameters
    ----------
    mesh : Mesh3d
        A 3D mesh object containing vertices, edges, and faces.
    
    edge_bevel_angles : dict
        A dictionary where keys are edge indices and values are bevel angles in degrees.
    
    ax : matplotlib.axes.Axes, optional
        A 3D axes object to plot on. If None, creates a new figure and axes.
    
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
    
    # Plot the mesh surface with transparency
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                   triangles=mesh.faces, color='lightgray', edgecolor=None, alpha=0.3)
    
    # Get all bevel angles for color mapping
    bevel_angles = list(edge_bevel_angles.values())
    
    if bevel_angles:
        # Create a colormap for the bevel angles
        cmap = plt.cm.viridis
        norm = Normalize(vmin=min(bevel_angles), vmax=max(bevel_angles))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Create line segments and colors for all join edges
        join_lines = []
        line_colors = []
        
        for edge_idx, bevel_angle in edge_bevel_angles.items():
            v1_idx, v2_idx = mesh.edges[edge_idx]
            v1 = mesh.vertices[v1_idx]
            v2 = mesh.vertices[v2_idx]
            join_lines.append([v1, v2])
            line_colors.append(cmap(norm(bevel_angle)))
        
        # Create a Line3DCollection
        if join_lines:
            lc = Line3DCollection(join_lines, colors=line_colors, linewidths=2)
            ax.add_collection(lc)
            
            # Add colorbar
            fig.colorbar(sm, ax=ax, label="Bevel Angle (degrees)")
    
    # Set axis labels and title
    ax.set_title("Mesh with Ideal Bevel Angles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Auto-adjust limits
    x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    
    buffer = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.05
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    ax.set_zlim(z_min - buffer, z_max + buffer)
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    mesh = trimesh.load(Path(__file__).parent.parent / "files" / "Trapezoid.stl")
    regions = segment_mesh_face_normals(mesh)
    
    # Extract specific regions for analysis
    region_indices = [1, 4]
    new_mesh = extract_mesh_regions(mesh, region_indices=region_indices, regions=regions)
    new_regions = segment_mesh_face_normals(new_mesh)
    
    # Analyze all region joins
    join_analysis = analyze_mesh_joins(new_mesh, new_regions)
    
    # Calculate ideal bevel angles for each edge
    edge_bevel_angles = calculate_bevel_angles(join_analysis)
    
    # Display results
    print(f"Found {len(join_analysis)} region joins:")
    for (r1_idx, r2_idx), data in join_analysis.items():
        print(f"Join between region {r1_idx} and region {r2_idx}:")
        print(f"  Join angle: {data['angle']:.2f} degrees")
        print(f"  Ideal bevel angle: {data['angle']/2:.2f} degrees")
        print(f"  Number of join edges: {len(data['join_edges'])}")
    
    # Create visualization
    fig, ax = visualize_bevel_angles(new_mesh, edge_bevel_angles)
    plt.show()
