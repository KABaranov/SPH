import numpy as np
from typing import List, Callable, Sequence
from SPH.core.particle.particle_dataclass import Particle


# --- Bruteforce neighbor search with grad_w for 1D, 2D, 3D ---

def build_neigh_1d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], float],
    grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
    box: float | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut**2
    x = np.array([p.x[0] for p in particles])  # (N,)
    box_len = float(box) if box is not None else None

    for pi in particles:
        dx = x - pi.x[0]  # (N,)
        if box_len is not None:
            dx -= box_len * np.round(dx / box_len)
        mask = dx**2 <= r_cut2
        idx = np.nonzero(mask)[0]

        neigh_w: List[float] = []
        grad_w: List[np.ndarray] = []
        for j in idx:
            # 3D vector [dx,0,0]
            r_vec = np.array([pi.x[0] - x[j], 0.0, 0.0], dtype=float)
            neigh_w.append(kernel(r_vec, h, 1))
            grad_w.append(grad_kernel(r_vec, h, 1))

        pi.neigh   = idx.tolist()
        pi.neigh_w = neigh_w
        pi.grad_w  = grad_w


def build_neigh_2d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], float],
    grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut**2
    xy = np.array([p.x[:2] for p in particles])  # (N,2)
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 2:
        raise ValueError("len(box) must equal 2 for 2D")

    for pi in particles:
        xi = pi.x[:2]
        dx = xy - xi  # (N,2)
        if box_arr is not None:
            for d in range(2):
                dx[:, d] -= box_arr[d] * np.round(dx[:, d] / box_arr[d])
        r2 = np.einsum("ij,ij->i", dx, dx)
        mask = r2 <= r_cut2
        idx = np.nonzero(mask)[0]

        neigh_w: List[float] = []
        grad_w: List[np.ndarray] = []
        for j in idx:
            r_vec = np.array([xi[0] - xy[j, 0], xi[1] - xy[j, 1], 0.0], dtype=float)
            neigh_w.append(kernel(r_vec, h, 2))
            grad_w.append(grad_kernel(r_vec, h, 2))

        pi.neigh   = idx.tolist()
        pi.neigh_w = neigh_w
        pi.grad_w  = grad_w


def build_neigh_3d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], float],
    grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut**2
    xyz = np.array([p.x for p in particles])  # (N,3)
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 3:
        raise ValueError("len(box) must equal 3 for 3D")

    for pi in particles:
        xi = pi.x
        dx = xyz - xi  # (N,3)
        if box_arr is not None:
            for d in range(3):
                dx[:, d] -= box_arr[d] * np.round(dx[:, d] / box_arr[d])
        r2 = np.einsum("ij,ij->i", dx, dx)
        mask = r2 <= r_cut2
        idx = np.nonzero(mask)[0]

        neigh_w: List[float] = []
        grad_w: List[np.ndarray] = []
        for j in idx:
            r_vec = dx[j]  # already length-3
            neigh_w.append(kernel(r_vec, h, 3))
            grad_w.append(grad_kernel(r_vec, h, 3))

        pi.neigh   = idx.tolist()
        pi.neigh_w = neigh_w
        pi.grad_w  = grad_w


def bruteforce(dim: int):
    return {
        1: build_neigh_1d,
        2: build_neigh_2d,
        3: build_neigh_3d,
    }[dim]
