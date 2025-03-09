import numpy as np
from numpy.typing import NDArray
import trimesh
from typing import List, Dict


def extract_mesh_regions(
    mesh: trimesh.Trimesh, 
    regions: List[List[int]]
) -> Dict[int, trimesh.Trimesh]:
    """
    Creates separate meshes for each of the specified regions.
    
    Parameters:
    ----------
    mesh : trimesh.Trimesh
        The original input mesh
    regions : List[List[int]]
        List of lists, where each inner list contains face indices for a region
        
    Returns:
    -------
    Dict[int, trimesh.Trimesh]
        Dictionary mapping region index to its corresponding mesh
    """
    region_meshes = {}
    
    for region_idx, face_indices in enumerate(regions):
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
        
        # Store the mesh with its region index
        region_meshes[region_idx] = region_mesh
        
    return region_meshes
