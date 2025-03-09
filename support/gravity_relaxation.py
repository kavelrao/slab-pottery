import numpy as np
from numpy.typing import NDArray
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from flattening.physics import calculate_rest_lengths, calculate_masses
from support.physics import calculate_gravity_forces_vectorized, calculate_spring_forces_vectorized, calculate_energy_per_vertex
from support.plotting import plot_mesh_comparison


"""
Doesn't really work for support detection; all the vertices end up going to the floor.
"""


def spring_mass_relaxation(
        mesh: trimesh.Trimesh,
        spring_constant: float = 0.5,
        gravity: float = 9.8,
        area_density: float = 1.0,
        dt: float = 0.001,
        rest_lengths: dict = None,
        num_iterations: int = 1000,
) -> NDArray[np.float32]:
    """
    Perform spring-mass relaxation on a 3D mesh.

    Args:
        mesh: The input 3D mesh
        spring_constant: The spring constant for the edges
        area_density: The density of the material
        dt: The time step for the simulation
        rest_lengths: The rest lengths of the edges
    Returns:
        new vertex positions
    """
    vertices_3d = mesh.vertices.copy()
    ground = vertices_3d[:, 2].min()
    # Calculate rest lengths if not provided
    if rest_lengths is None:
        rest_lengths = calculate_rest_lengths(vertices_3d, mesh.edges)

    # Calculate masses
    masses = calculate_masses(vertices_3d, mesh.faces, area_density)

    # Initialize velocities and accelerations
    velocities = np.zeros_like(mesh.vertices)
    accelerations = np.zeros_like(mesh.vertices)

    # Perform spring-mass relaxation
    for _ in range(num_iterations):
        # Calculate forces
        spring_forces = calculate_spring_forces_vectorized(vertices_3d, mesh.edges, rest_lengths, spring_constant)
        # Calculate gravity forces
        gravity_forces = calculate_gravity_forces_vectorized(mesh.vertices, masses, gravity)

        forces = spring_forces + gravity_forces

        # Calculate accelerations
        accelerations = forces / masses[:, np.newaxis]  # a = F/m

        # Update velocities and positions
        velocities += accelerations * dt
        displacements = velocities * dt + 0.5 * accelerations * dt**2
        vertices_3d += displacements
        # make sure the vertices don't go below the ground
        vertices_3d[:, 2] = np.maximum(vertices_3d[:, 2], ground)

    return vertices_3d


def main():
    """
    Main function to load a mesh, perform spring-mass relaxation, and plot the results.
    """
    # Load a mesh from a file
    mesh_path = Path(__file__).parent.parent / "files" / "Mug_Thick_Handle.stl"
    mesh = trimesh.load(mesh_path)
    
    print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    print("Performing spring-mass relaxation...")
    
    # Perform the spring-mass relaxation
    relaxed_vertices = spring_mass_relaxation(
        mesh=mesh,
        num_iterations=1000,
        gravity=1.0,
        spring_constant=5.0
    )
    
    print("Relaxation complete. Plotting results...")
    
    # Plot the results
    plot_mesh_comparison(mesh, relaxed_vertices)


if __name__ == "__main__":
    main()
