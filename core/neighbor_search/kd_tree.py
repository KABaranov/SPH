import numpy as np
from typing import List, Callable, Sequence, Dict
from scipy.spatial import cKDTree
from SPH.core.particle.particle_dataclass import Particle


# --- KD-tree neighbor search with grad_w for 1D, 2D, 3D ---

def build_neigh_kdtree_1d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], float],
    grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
    box: float | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    x = np.array([p.x[0] for p in particles])  # shape (N,)
    box_arr = float(box) if box is not None else None
    coords = x.reshape(-1, 1)
    tree = cKDTree(coords, boxsize=box_arr) if box_arr is not None else cKDTree(coords)
    neighbors_idx = tree.query_ball_point(coords, r=r_cut)

    for i, pi in enumerate(particles):
        idx = neighbors_idx[i]
        neigh_w: List[float] = []
        grad_w: List[np.ndarray] = []
        # compute weights and grads
        for j in idx:
            dx = x[j] - x[i]
            if box_arr is not None:
                dx -= box_arr * np.round(dx/box_arr)
            # 3D vector
            r_vec = np.array([dx, 0.0, 0.0], dtype=float)
            neigh_w.append(kernel(r_vec, h, 1))
            grad_w.append(grad_kernel(r_vec, h, 1))
        pi.neigh = idx
        pi.neigh_w = neigh_w
        pi.grad_w = grad_w


def build_neigh_kdtree_2d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], float],
    grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    xy = np.vstack([p.x[:2] for p in particles])  # (N,2)
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 2:
        raise ValueError("len(box) must equal 2 for 2D")
    tree = cKDTree(xy, boxsize=box_arr) if box_arr is not None else cKDTree(xy)
    neighbors_idx = tree.query_ball_point(xy, r=r_cut)

    for i, pi in enumerate(particles):
        idx = neighbors_idx[i]
        neigh_w: List[float] = []
        grad_w: List[np.ndarray] = []
        xi = pi.x[:2]
        for j in idx:
            dx = xy[j] - xi
            if box_arr is not None:
                dx -= box_arr * np.round(dx/box_arr)
            # 3D vector
            r_vec = np.array([dx[0], dx[1], 0.0], dtype=float)
            neigh_w.append(kernel(r_vec, h, 2))
            grad_w.append(grad_kernel(r_vec, h, 2))
        pi.neigh = idx
        pi.neigh_w = neigh_w
        pi.grad_w = grad_w


def build_neigh_kdtree_3d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], float],
    grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    xyz = np.vstack([p.x for p in particles])  # (N,3)
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 3:
        raise ValueError("len(box) must equal 3 for 3D")
    tree = cKDTree(xyz, boxsize=box_arr) if box_arr is not None else cKDTree(xyz)
    neighbors_idx = tree.query_ball_point(xyz, r=r_cut)

    for i, pi in enumerate(particles):
        idx = neighbors_idx[i]
        neigh_w: List[float] = []
        grad_w: List[np.ndarray] = []
        for j in idx:
            dx = xyz[j] - xyz[i]
            if box_arr is not None:
                dx -= box_arr * np.round(dx/box_arr)
            neigh_w.append(kernel(dx, h, 3))
            grad_w.append(grad_kernel(dx, h, 3))
        pi.neigh = idx
        pi.neigh_w = neigh_w
        pi.grad_w = grad_w


def kd_tree(dim: int):
    return {
        1: build_neigh_kdtree_1d,
        2: build_neigh_kdtree_2d,
        3: build_neigh_kdtree_3d,
    }[dim]
