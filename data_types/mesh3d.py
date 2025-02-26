from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Mesh3d:
    vertices: NDArray[np.float64]  # V x 3 array of vertex coordinates
    edges: NDArray[np.int64]  # E x 2 array of vertex *indices* which are edge endpoints
    faces: NDArray[np.int64]  # F x 3 array of vertex *indices* which are face corners
