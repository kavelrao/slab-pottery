import argparse
import os
import sys
import trimesh
import matplotlib.pyplot as plt

from deduplication import interactive_deduplicate
from segmenting import segment_mesh_face_normals
from plotting import plot_mesh

def load_stl_file_cli():
    parser = argparse.ArgumentParser(description='Load an STL file into a trimesh object.')
    parser.add_argument('filename', type=str, help='Path to the STL file to load')
    
    args = parser.parse_args()
    filepath = args.filename
    
    if not filepath.lower().endswith('.stl'):
        print(f"Error: The file must have a .stl extension. Got: {filepath}")
        sys.exit(1)
    
    if not os.path.isfile(filepath):
        print(f"Error: The file {filepath} does not exist.")
        sys.exit(1)
    
    try:
        mesh = trimesh.load(filepath)
        print(f"Successfully loaded {filepath}")
        print(f"Mesh info: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
        
        return mesh
    
    except Exception as e:
        print(f"Error loading the STL file: {e}")
        sys.exit(1)

def main():
    mesh = load_stl_file_cli()

    dedup_mesh, dedup_regions = interactive_deduplicate(mesh, segment_fn=lambda mesh: segment_mesh_face_normals(mesh, angle_threshold=30))
    plot_mesh(mesh, title="Selected Mesh")
    plt.show()

    

if __name__ == "__main__":
    main()
