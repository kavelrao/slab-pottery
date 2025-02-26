import numpy as np
from collections import defaultdict
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray

from plotting import plot_mesh_regions


def segment_mesh_face_normals(mesh: trimesh.Trimesh, angle_threshold=15) -> list[list[int]]:
    """Segments a mesh into regions based on face normal similarity."""
    regions = []
    unvisited_faces = set(range(len(mesh.faces)))

    # Preprocess face_adjacency and face_adjacency_angles:
    neighbors = defaultdict(list)
    for (face_1, face_2), angle in zip(mesh.face_adjacency, mesh.face_adjacency_angles):
        neighbors[face_1].append((face_2, angle))
        neighbors[face_2].append((face_1, angle))

    while unvisited_faces:
        seed_face_index = unvisited_faces.pop()
        current_region = [seed_face_index]
        faces_to_visit = [seed_face_index]

        while faces_to_visit:
            face_index = faces_to_visit.pop(0)

            for neighbor_index, neighbor_angle in neighbors[face_index]:
                if neighbor_index in unvisited_faces:

                  if np.abs(np.degrees(neighbor_angle)) < angle_threshold:
                      current_region.append(neighbor_index)
                      faces_to_visit.append(neighbor_index)
                      unvisited_faces.remove(neighbor_index)

        regions.append(current_region)

    return regions


if __name__ == '__main__':
    mesh = trimesh.load(Path(__file__).parent.parent / "files" / "Mug_Thick_Handle.stl")
    regions = segment_mesh_face_normals(mesh, angle_threshold=30)
    fig, ax = plot_mesh_regions(mesh, regions, title="Pottery Slab Regions")
    plt.show()
