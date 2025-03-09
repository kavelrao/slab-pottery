import numpy as np
from numpy.typing import NDArray
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

from support.plotting import plot_mesh_with_support_regions


def face_angle_support_detection(mesh: trimesh.Trimesh, angle_threshold=45, area_threshold=4.0):
    """
    Detects support regions in a 3D mesh based on face angles and Z-axis orientation.

    This function identifies regions of a 3D mesh that may require support
    structures based on two criteria:
    1. The angles between adjacent faces
    2. Faces with a Z-axis angle over the specified threshold (overhang detection)

    The function performs BFS to connect adjacent faces that satisfy the angle constraint,
    and only includes regions whose total face area exceeds the area threshold.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A 3D mesh object containing vertices, edges, and faces.
        Vertices is a V x 3 array of vertex coordinates.
        Edges is an E x 2 array of vertex indices representing edge endpoints.
        Faces is an F x 3 array of vertex indices representing face corners.
    angle_threshold : float, optional
        The angle threshold in degrees. Faces with Z-axis angle less than this
        threshold will be considered for support. Default is 45 degrees.
    area_threshold : float, optional
        The minimum area threshold. Only regions with total face area exceeding
        this threshold will be included in the results. Default is 1.0.

    Returns
    -------
    List[List[np.int64]]
        A list of face index lists, where each inner list represents a connected
        region requiring support.
    """
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Z-axis vector [0, 0, 1]
    z_axis = np.array([0, 0, 1])
    
    # Calculate angle between face normals and Z-axis
    # Using dot product: cos(θ) = (a·b)/(|a|·|b|)
    # Since both vectors are normalized, |a|·|b| = 1
    z_angles = np.minimum(np.arccos(np.clip(np.dot(face_normals, z_axis), -1.0, 1.0)), 
                         np.arccos(np.clip(np.dot(-face_normals, z_axis), -1.0, 1.0)))

    # Convert angles to degrees and identify candidate faces
    z_angles_deg = np.degrees(np.abs(z_angles))
    candidate_faces = np.where(z_angles_deg < angle_threshold)[0]
    
    # Create face adjacency lookup
    face_adjacency = mesh.face_adjacency
    face_neighbors = {}
    for i, j in face_adjacency:
        if i not in face_neighbors:
            face_neighbors[i] = []
        if j not in face_neighbors:
            face_neighbors[j] = []
        face_neighbors[i].append(j)
        face_neighbors[j].append(i)
    
    # Compute face areas
    face_areas = mesh.area_faces
    
    # Keep track of visited faces
    visited = set()
    
    # List to store all support regions
    support_regions = []
    
    # Perform BFS for each candidate face
    for face_idx in candidate_faces:
        if face_idx in visited:
            continue
            
        # Initialize a new region
        region = []
        queue = [face_idx]
        region_visited = set()
        
        # BFS to find connected faces that satisfy the angle constraint
        while queue:
            current_face = queue.pop(0)
            
            if current_face in region_visited:
                continue
                
            region_visited.add(current_face)
            region.append(current_face)
            
            # Add neighbors that satisfy the angle constraint
            if current_face in face_neighbors:
                for neighbor in face_neighbors[current_face]:
                    if (neighbor not in region_visited and 
                        neighbor not in visited and 
                        z_angles_deg[neighbor] < angle_threshold):
                        queue.append(neighbor)
        
        # Calculate total area of the region
        region_area = sum(face_areas[idx] for idx in region)
        
        # Add region to support regions if area exceeds threshold
        if region_area > area_threshold:
            support_regions.append(region)
            
        # Mark all faces in this region as visited
        visited.update(region_visited)
    
    return support_regions


def main():
    mesh_path = Path(__file__).parent.parent / "files" / "Swept_Bowl_Shell.stl"
    mesh = trimesh.load(mesh_path)
    
    print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    support_regions = face_angle_support_detection(mesh)
    fig = plot_mesh_with_support_regions(mesh, list(itertools.chain.from_iterable(support_regions)))
    plt.show()


if __name__ == "__main__":
    main()
