import trimesh
import numpy as np
import networkx as nx
from itertools import combinations

def get_connected_components(edges):
    """
    Helper function to get connected components of a graph.
    Returns the number of connected components and a list of 2-element lists of vertex indices in each edge in each connected component
    """

    graph = nx.Graph()
    graph.add_edges_from(edges)

    connected_components = list(nx.connected_components(graph))
    num_connected_components = len(connected_components)

    component_edges = []
    for component in connected_components:
        component_edges.append(list(graph.subgraph(component).edges()))

    # print(f"Number of connected components: {num_connected_components}")
    # for i, component in enumerate(component_edges):
    #     print(f"Component {i+1}: Edges {component}")

    return num_connected_components, component_edges


def count_boundary_loops(mesh: trimesh.Trimesh, face_in_region):
    """
    Given a mesh and a list of faces in some region of the mesh, return the boundary loops in the region.
    Returns the number of boundary loops and a NumPy array of size 2() of 2-element lists of vertex indices in each edge in each boundary loop.
    """
    # Use combinations of vertices for each face to get edges
    edge_in_region = []
    for face in face_in_region:
        edge_in_region += [sorted(comb) for comb in combinations(mesh.faces[face], 2)]

    edge_in_region = np.array(edge_in_region)

    # print(trimesh.grouping.group_rows(edge_in_region, require_count=1))

    boundary_edges = edge_in_region[trimesh.grouping.group_rows(edge_in_region, require_count=1)]

    num_comp, edges_in_comp = get_connected_components(boundary_edges)

    return num_comp, np.array(edges_in_comp)


def get_connected_components_by_edge_index(mesh, boundary_edge_indices):
    """
    Helper function to get connected components of boundary edges using edge indices.
    Returns the number of connected components and a list of edge indices in each connected component.
    """
    # Create graph using vertices from the boundary edges
    graph = nx.Graph()
    
    # Add edges to graph using vertex pairs (not edge indices)
    for edge_idx in boundary_edge_indices:
        v1, v2 = mesh.edges[edge_idx]
        graph.add_edge(v1, v2)
    
    # Find connected components of vertices
    connected_components = list(nx.connected_components(graph))
    num_connected_components = len(connected_components)
    
    # Map components back to edge indices
    component_edge_indices = []
    for component_vertices in connected_components:
        component_subgraph = graph.subgraph(component_vertices)
        # Find which boundary edges correspond to the edges in this component
        component_edges = []
        for edge_idx in boundary_edge_indices:
            v1, v2 = mesh.edges[edge_idx]
            if v1 in component_vertices and v2 in component_vertices and component_subgraph.has_edge(v1, v2):
                component_edges.append(edge_idx)
        component_edge_indices.append(component_edges)
    
    return num_connected_components, component_edge_indices


def count_boundary_loops_by_edge_index(mesh: trimesh.Trimesh):
    """
    Given a mesh, return the boundary loops using edge indices.
    Returns the number of boundary loops and a list of edge indices in each boundary loop.
    """
    # Find edges that appear only once (boundary edges)
    edge_to_face = {}
    for face_idx, face in enumerate(mesh.faces):
        # For each edge in the face
        for i in range(3):  # Assuming triangular mesh
            v1, v2 = face[i], face[(i+1)%3]
            edge = tuple(sorted([v1, v2]))
            
            if edge in edge_to_face:
                edge_to_face[edge].append(face_idx)
            else:
                edge_to_face[edge] = [face_idx]
    
    # Get boundary edges (edges with only one adjacent face)
    boundary_edges = [edge for edge, faces in edge_to_face.items() if len(faces) == 1]
    
    # Map boundary edges to edge indices in mesh.edges
    boundary_edge_indices = []
    for i, (v1, v2) in enumerate(mesh.edges):
        edge = tuple(sorted([v1, v2]))
        if edge in boundary_edges:
            boundary_edge_indices.append(i)
    
    # Get connected components
    num_comp, edges_in_comp = get_connected_components_by_edge_index(mesh, boundary_edge_indices)
    
    return num_comp, edges_in_comp