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