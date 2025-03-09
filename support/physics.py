import numpy as np
from numpy.typing import NDArray


def calculate_gravity_forces_vectorized(vertices_3d: NDArray[np.float64], masses: NDArray[np.float64], gravity: float = 9.8) -> NDArray[np.float64]:
    """
    Calculate gravity forces for each vertex in a 3D mesh.
    
    Args:
        vertices_3d: 3D vertex positions
        masses: Masses of each vertex
        gravity: Acceleration due to gravity
        
    Returns:
        numpy.ndarray: Force vectors for each vertex
    """
    # Calculate gravity forces using mass and gravity
    gravity_forces = np.zeros_like(vertices_3d)
    gravity_forces[:, 2] = -masses * gravity
    # gravity_forces = np.where(np.isclose(vertices_3d[:, 2], vertices_3d[:, 2].min())[:, np.newaxis], 0, gravity_forces)
    return gravity_forces


def calculate_spring_forces_vectorized(vertices_3d: NDArray[np.float64], edges: NDArray[np.int64], 
                             rest_lengths: dict, spring_constant: float) -> NDArray[np.float64]:
    """
    Vectorized implementation of force calculation for better performance.
    
    Args:
        vertices_2d: 2D vertex positions
        edges: Edge indices
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for force calculation
        
    Returns:
        numpy.ndarray: Force vectors for each vertex
    """
    num_vertices = len(vertices_3d)
    forces = np.zeros((num_vertices, 3), dtype=np.float64)
    
    # Extract edge endpoints
    v1_indices = edges[:, 0]
    v2_indices = edges[:, 1]
    
    # Get positions of edge endpoints
    p1_positions = vertices_3d[v1_indices]
    p2_positions = vertices_3d[v2_indices]
    
    # Calculate edge vectors and current lengths
    edge_vectors = p2_positions - p1_positions
    current_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Check for near-zero lengths
    mask = current_lengths > 1e-9
    if not np.all(mask):
        raise ValueError("Some edges have near-zero lengths in vectorized force calculation")
    
    # Get rest lengths for edges (convert tuple keys to indices)
    rest_length_values = np.array([rest_lengths[(v1, v2)] for v1, v2 in edges])
    
    # Calculate force magnitudes
    force_magnitudes = spring_constant * (current_lengths - rest_length_values)
    
    # Normalize edge vectors to get directions
    directions = edge_vectors / current_lengths[:, np.newaxis]
    
    # Calculate force vectors
    force_vectors = force_magnitudes[:, np.newaxis] * directions
    
    # Apply forces to vertices using numpy's add.at for accumulation
    np.add.at(forces, v1_indices, force_vectors)
    np.add.at(forces, v2_indices, -force_vectors)
    
    return forces

def calculate_energy_per_vertex(vertices_3d: NDArray[np.float64], edges: NDArray[np.int64], 
                               masses: NDArray[np.float64], rest_lengths: dict, 
                               spring_constant: float, gravity: float = 9.8) -> NDArray[np.float64]:
    """
    Calculate the energy at each vertex, including spring potential energy and gravitational potential energy.
    
    Args:
        vertices_3d: 3D vertex positions
        edges: Edge indices
        masses: Masses of each vertex
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for energy calculation
        gravity: Acceleration due to gravity
        
    Returns:
        numpy.ndarray: Energy values for each vertex
    """
    num_vertices = len(vertices_3d)
    
    # Initialize energy per vertex array
    vertex_energy = np.zeros(num_vertices, dtype=np.float64)
    
    # Calculate gravitational potential energy for each vertex
    # Energy = m * g * h where h is the height (z-coordinate)
    # Reference height (z=0) can be adjusted as needed
    gravitational_energy = masses * gravity * vertices_3d[:, 2]
    
    # Add gravitational energy to each vertex
    vertex_energy += gravitational_energy
    
    # Calculate spring potential energy
    # For each edge, calculate E = 0.5 * k * (l - l0)^2
    # and distribute equally to both vertices
    for i, (v1, v2) in enumerate(edges):
        # Get positions
        p1 = vertices_3d[v1]
        p2 = vertices_3d[v2]
        
        # Calculate current length
        current_length = np.linalg.norm(p2 - p1)
        
        # Get rest length
        rest_length = rest_lengths[(v1, v2)]
        
        # Calculate spring energy
        spring_energy = 0.5 * spring_constant * (current_length - rest_length)**2
        
        # Distribute energy equally to both vertices
        vertex_energy[v1] += spring_energy / 2
        vertex_energy[v2] += spring_energy / 2
    
    return vertex_energy

def calculate_total_energy(vertices_3d: NDArray[np.float64], edges: NDArray[np.int64], 
                          masses: NDArray[np.float64], rest_lengths: dict, 
                          spring_constant: float, gravity: float = 9.8) -> float:
    """
    Calculate the total energy of the system.
    
    Args:
        vertices_3d: 3D vertex positions
        edges: Edge indices
        masses: Masses of each vertex
        rest_lengths: Dictionary of rest lengths for each edge
        spring_constant: Spring constant for energy calculation
        gravity: Acceleration due to gravity
        
    Returns:
        float: Total energy of the system
    """
    # Get energy per vertex
    vertex_energy = calculate_energy_per_vertex(
        vertices_3d, edges, masses, rest_lengths, spring_constant, gravity
    )
    
    # Sum all vertex energies
    total_energy = np.sum(vertex_energy)
    
    return total_energy
