import trimesh
import numpy as np
import heapq
import igl

from .boundary_loops import count_boundary_loops

def adaptive_map_generation(mesh: trimesh.Trimesh, boundary_verts):
    """
    Implements adaptive map generation to generate a Boundary geodesic distance map and a Gaussian curvature map.
    Input: A graph G(v, e) of the given mesh surface M.
    Output: The updated weight number of every triangular node.
    """

    weights = np.full(len(mesh.vertices), np.inf)

    # aka Lv in the paper
    event_list = []

    # aka fp in the paper
    visited = np.zeros(len(mesh.vertices), dtype=bool)

    for vertex in boundary_verts:
        weights[vertex] = 0
        event_list.append(vertex)
        visited[vertex] = True

    # Calculate shortest edge distance in the mesh, save it to l_min
    l_min = np.min(np.linalg.norm(mesh.vertices[mesh.edges_unique[:, 0]] - mesh.vertices[mesh.edges_unique[:, 1]], axis=1))

    # lambda in the paper
    l_curr = l_min

    while len(event_list) > 0:

        event_list_curr = []

        for vertex in event_list:
            # iterate through every neighbor of the vertex
            for neighbor in mesh.vertex_neighbors[vertex]:
                curr_len = weights[vertex] + np.abs(np.linalg.norm(mesh.vertices[vertex] - mesh.vertices[neighbor]))
                if curr_len < weights[neighbor]:
                    weights[neighbor] = curr_len
                if (not visited[neighbor]) and (weights[neighbor] < l_curr):
                    event_list_curr.append(neighbor)
                    visited[neighbor] = True
            
        for vertex in event_list:
            # If visited of any neighbor is False, add the vertex to the event list

            # Check if any value in [visited[neighbor] for neighbor in mesh.vertex_neighbors[vertex]] is False
            if np.any([not visited[neighbor] for neighbor in mesh.vertex_neighbors[vertex]]):
                event_list_curr.append(vertex)
        
        event_list = event_list_curr
        l_curr = l_curr + l_min

    return weights

def descent_func(weights, mesh, vert_i, vert_j):
    """
    Implements the descent function to calculate the descent value of a given edge.
    We want to travel in the direction of steepest descent.
    """

    # Calculate the descent value of the edge (vi, vj)
    return (weights[vert_i]  - weights[vert_j]) / np.abs(np.linalg.norm(mesh.vertices[vert_i] - mesh.vertices[vert_j]))
    
def path_generation(mesh: trimesh.Trimesh, vert_start, dist_map):
    """
    Implements the path generation function to generate the cutting path from vert_start to a boundary vertex.
    vert_start must be given as a vertex index in the mesh.
    map should be an array of the same size as the number of vertices in the mesh, ideally acquired from the adaptive_map_generation function.
    """

    path = []
    while dist_map[vert_start] != 0:
        v_max = mesh.vertex_neighbors[vert_start][0]

        for neighbor in mesh.vertex_neighbors[vert_start]:
            if descent_func(dist_map, mesh, vert_start, neighbor) > descent_func(dist_map, mesh, vert_start, v_max):
                v_max = neighbor

        # Add edge from vert_start to v_max to the path
        path.append((vert_start, v_max))
        vert_start = v_max

    return path

def find_cutting_path(mesh: trimesh.Trimesh):

    num_loops, boundary_loops = count_boundary_loops(mesh, np.arange(len(mesh.faces)))

    print(num_loops)

    # If there is more than 1 boundary loop, select the largest one

    loop_verts = []
    if num_loops > 1:
        print(num_loops)
        loop_sizes = [len(loop) for loop in boundary_loops]
        print(np.argmax(loop_sizes))
        largest_loop = np.argmax(loop_sizes)

        if largest_loop > 1:
            largest_loop = largest_loop[0]

        # Get the vertices in the largest loop
        boundary_verts = np.unique(np.array(largest_loop).flatten())

        for i in range(len(loop_sizes)):
            print(largest_loop)
            if i != largest_loop:
                loop_i_verts = np.unique(np.array(boundary_loops[i]).flatten())
                # append loop_i_verts to loop_verts
                loop_verts.extend(loop_i_verts)

    else:
        boundary_verts = np.unique(np.array(boundary_loops[0]).flatten())
        loop_verts = []

    p_star = []
    # Generate the adaptive map
    weights = adaptive_map_generation(mesh, boundary_verts)
   
    # Use Kobbelt approximation of gaussian curvature to calculate gaussian curvature of non-boundary vertices
    kurve = np.zeros(len(mesh.vertices))

    for i in range(len(mesh.vertices)):
        if i in loop_verts:
            kurve[i] = np.inf
        elif i not in boundary_verts:
            kurve[i] = mesh.vertex_defects[i] / ((1/3) * np.sum(mesh.area_faces[mesh.vertex_faces[i]]))

    # find the max curvature value where the curvature is not infinity
    epsilon = 0
    for curv in kurve:
        if curv < np.inf and curv > epsilon:
            epsilon = curv

    # Lower-threshold for selecting points which we should cut through
    epsilon = epsilon * 0.8

    # Create a min_heap that stores each vertex index with its weight

    min_heap = []

    # Iterate over indices in kurve
    for i in range(len(kurve)):
        # Find the index of the vertex with the maximum curvature
        if kurve[i] > epsilon:
            heapq.heappush(min_heap, (weights[i], i))

    # iterations = 0
    # iter_max = 10000
    
    while len(min_heap) > 0:
        # Pop the vertex with the maximum curvature from the heap
        _, vert_start = heapq.heappop(min_heap)
        path = path_generation(mesh, vert_start, weights)
        p_star.extend(path) # TODO is extend the right thing to use here?

        # iterations += 1
        # if iterations > iter_max:
        #     break

        # Expand our boundary
        boundary_verts = np.append(boundary_verts, np.array(path).flatten())

        # Update the weights
        weights = adaptive_map_generation(mesh, boundary_verts)

        # Update the heap
        min_heap_new = []

        while len(min_heap) > 0:
            _, vert = heapq.heappop(min_heap)
            heapq.heappush(min_heap_new, (weights[vert], vert))
        
        min_heap = min_heap_new
    
    return np.array(p_star)

def make_cut(mesh: trimesh.Trimesh, cutting_path):
    """
    Implements the make_cut function to cut the mesh along the cutting path.
    """

    # preprocess cutting path into a map of vertex to set of neighbors
    vertex_neighbors = {}
    for edge in cutting_path:
        if edge[0] not in vertex_neighbors:
            vertex_neighbors[edge[0]] = set()
        if edge[1] not in vertex_neighbors:
            vertex_neighbors[edge[1]] = set()
        vertex_neighbors[edge[0]].add(edge[1])
        vertex_neighbors[edge[1]].add(edge[0])

    cut_data = np.zeros(mesh.faces.shape, dtype=int)

    # preprocess edges to a map from edge to num faces that the edge belongs to
    # This will hopefully eliminate boundary edges from the cut
    edge_faces = {}
    for i, face in enumerate(mesh.faces):
        for j in range(3):
            # Sort face[j] and face[(j + 1) % 3] to ensure that the edge is always in the same order
            edge = sorted([face[j], face[(j + 1) % 3]])
            edge = (edge[0], edge[1])
            if edge not in edge_faces:
                edge_faces[edge] = 0
            edge_faces[edge] += 1

    final_cuts = []

    # iterate over faces
    for i, face in enumerate(mesh.faces):
        # iterate over edges
        for j in range(3):
            edge = sorted([face[j], face[(j + 1) % 3]])
            edge = (edge[0], edge[1])
            if edge[0] in vertex_neighbors:
                if edge[1] in vertex_neighbors[edge[0]]:
                    if edge_faces[edge] != 1:
                        final_cuts.append([edge[0], edge[1]])
                        cut_data[i, j] = True

    # Cut the mesh. For each pair in edge_faces, if the edge is in the cutting path, duplicate the vertex and add it to the mesh.
    # change the face to use the new vertex
                        
    vcut, fcut = igl.cut_mesh(mesh.vertices, mesh.faces, cut_data.astype(np.int64))

    # breakpoint()
                        
    # vcut = mesh.vertices.copy()

    # verts_to_add = []
    # fcut = mesh.faces.copy()

    # for j, face in enumerate(cut_data):
    #     for i, cut in enumerate(face):
    #         if cut:
    #             verts_to_add.append(vcut[i])
    #             fcut[j, i] = vcut.shape[0] + len(verts_to_add) - 1
    #             verts_to_add.append(vcut[(i + 1) % 3])
    #             fcut[j, (i + 1) % 3] = vcut.shape[0] + len(verts_to_add) - 1
    
    # vcut = np.vstack([vcut, np.array(verts_to_add)])

    

    # breakpoint()

    return trimesh.Trimesh(vcut, fcut, process=False, validate=False), final_cuts



if __name__ == "__main__":
    mesh = trimesh.load_mesh("files/Mug_Shell.stl")
    cutting_path = find_cutting_path(mesh)
    mesh_cut = make_cut(mesh, cutting_path)

    print(cutting_path)
