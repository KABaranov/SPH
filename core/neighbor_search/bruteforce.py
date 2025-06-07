import numpy as np
from typing import List, Callable, Sequence

from SPH.core.particle.particle_dataclass import Particle


def build_neigh_1d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[float, float, int], float],
    box: float | None = None,
    qmax: float = 2.0,
):
    """Brute-force neighbor search in 1D with optional periodic box."""
    r_cut = qmax * h
    r_cut2 = r_cut**2
    # extract positions
    x = np.array([p.x[0] for p in particles])        # shape (N,)
    # box handling (scalar only)
    if box is None:
        box_arr = None
    else:
        box_arr = float(box)

    for i, pi in enumerate(particles):
        dx = x - pi.x[0]  # (N,)
        # periodic boundary
        if box_arr is not None:
            L = box_arr
            dx -= L * np.round(dx / L)
        r2 = dx**2
        mask = r2 <= r_cut2
        idx = np.nonzero(mask)[0]
        r = np.sqrt(r2[mask])
        w = [kernel(ri, h, 1) for ri in r]
        pi.neigh = idx.tolist()
        pi.neigh_w = w


def build_neigh_2d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[float, float, int], float],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    """Brute-force neighbor search in 2D with optional periodic box."""
    r_cut = qmax * h
    r_cut2 = r_cut**2
    # extract positions
    x = np.array([p.x for p in particles])  # shape (N,2)
    # box handling
    if box is None:
        box_arr = None
    else:
        box_arr = np.asarray(box, dtype=float)
        if box_arr.size != 2:
            raise ValueError("len(box) must equal 2 for 2D")

    for i, pi in enumerate(particles):
        dx = x - pi.x  # (N,2)
        # periodic boundary
        if box_arr is not None:
            for d in range(2):
                L = box_arr[d]
                dx[:, d] -= L * np.round(dx[:, d] / L)
        r2 = np.einsum("ij,ij->i", dx, dx)
        mask = r2 <= r_cut2
        idx = np.nonzero(mask)[0]
        r = np.sqrt(r2[mask])
        w = [kernel(ri, h, 2) for ri in r]
        pi.neigh = idx.tolist()
        pi.neigh_w = w


def build_neigh_3d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[float, float, int], float],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    """Brute-force neighbor search in 3D with optional periodic box."""
    r_cut = qmax * h
    r_cut2 = r_cut**2
    # extract positions
    x = np.array([p.x for p in particles])  # shape (N,3)
    # box handling
    if box is None:
        box_arr = None
    else:
        box_arr = np.asarray(box, dtype=float)
        if box_arr.size != 3:
            raise ValueError("len(box) must equal 3 for 3D")

    for i, pi in enumerate(particles):
        dx = x - pi.x  # (N,3)
        # periodic boundary
        if box_arr is not None:
            for d in range(3):
                L = box_arr[d]
                dx[:, d] -= L * np.round(dx[:, d] / L)
        r2 = np.einsum("ij,ij->i", dx, dx)
        mask = r2 <= r_cut2
        idx = np.nonzero(mask)[0]
        r = np.sqrt(r2[mask])
        w = [kernel(ri, h, 3) for ri in r]
        pi.neigh = idx.tolist()
        pi.neigh_w = w


def bruteforce(dim: int):
    match dim:
        case 1: return build_neigh_1d
        case 2: return build_neigh_2d
        case 3: return build_neigh_3d
