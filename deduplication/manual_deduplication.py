import numpy as np
from numpy.typing import NDArray
import trimesh
from typing import List, Union

from segmenting import segment_mesh_face_normals


def extract_mesh_regions(
    mesh: trimesh.Trimesh, 
    region_indices: Union[int, List[int]], 
    regions: List[NDArray[np.int64]]
) -> trimesh.Trimesh:
    """
    Creates a new mesh containing only the specified regions.
    
    Parameters:
    ----------
    mesh : trimesh.Trimesh
        The original input mesh
    region_indices : int or List[int]
        Indices of the regions to extract from the regions list
    regions : List[NDArray[np.int64]]
        List of regions as returned by segment_mesh_face_normals
        
    Returns:
    -------
    trimesh.Trimesh
        A new mesh containing only the faces from the selected regions
    """
    # Convert single region index to list for consistent processing
    if isinstance(region_indices, int):
        region_indices = [region_indices]
    
    # Get all face indices from the selected regions
    selected_faces = []
    for idx in region_indices:
        if idx < 0 or idx >= len(regions):
            raise ValueError(f"Region index {idx} out of bounds. Available regions: 0-{len(regions)-1}")
        selected_faces.extend(regions[idx])
    
    # Create a new mesh with only the selected faces
    faces = mesh.faces[selected_faces]
    vertices = mesh.vertices
    
    # Create a mapping from old vertex indices to new ones
    # This is to remove unused vertices
    unique_vertices = np.unique(faces.flatten())
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
    
    # Create the new faces array with remapped vertex indices
    new_faces = np.zeros_like(faces)
    for i in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            new_faces[i, j] = vertex_map[faces[i, j]]
    
    # Create the new vertices array with only the used vertices
    new_vertices = vertices[unique_vertices]
    
    # Create and return the new mesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    # Copy applicable attributes from the original mesh
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        new_mesh.visual.face_colors = mesh.visual.face_colors[selected_faces]
    
    return new_mesh, segment_mesh_face_normals(new_mesh)
