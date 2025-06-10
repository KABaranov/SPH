import numpy as np
from typing import List, Callable, Sequence
from scipy.spatial import cKDTree

from SPH.core.particle.particle_dataclass import Particle


def build_neigh_kdtree_1d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[float, float, int], float],
    box: float | None = None,
    qmax: float = 2.0,
):
    """KD-tree neighbor search in 1D with optional periodic box."""
    r_cut = qmax * h
    # 1D: массив координат shape (N,)
    x = np.array([p.x[0] for p in particles])
    # подготовим boxsize для cKDTree
    box_arr = float(box) if box is not None else None

    # cKDTree в 1D ожидает coords shape (N,1)
    coords = x.reshape(-1, 1)
    tree = cKDTree(coords, boxsize=box_arr) if box_arr is not None else cKDTree(coords)

    # query_ball_point вернет для каждой точки список соседних индексов
    neighbors_idx = tree.query_ball_point(coords, r=r_cut)

    for i, p in enumerate(particles):
        idx = neighbors_idx[i]
        p.neigh = idx

        # заново считаем dx, чтобы корректно получить r и применить kernel
        dx = x[idx] - x[i]
        if box_arr is not None:
            dx -= box_arr * np.round(dx / box_arr)
        r = np.abs(dx)

        p.neigh_w = [kernel(ri, h, 1) for ri in r]


def build_neigh_kdtree_2d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[float, float, int], float],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    """KD-tree neighbor search in 2D with optional periodic box."""
    r_cut = qmax * h
    x = np.vstack([p.x for p in particles])  # (N,2)
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 2:
        raise ValueError("len(box) must equal 2 for 2D")

    tree = (
        cKDTree(x, boxsize=box_arr)
        if box_arr is not None
        else cKDTree(x)
    )
    neighbors_idx = tree.query_ball_point(x, r=r_cut)

    for i, p in enumerate(particles):
        idx = neighbors_idx[i]
        p.neigh = idx

        dx = x[idx] - x[i]  # (m,2)
        if box_arr is not None:
            # минимальный образ
            for d in range(2):
                L = box_arr[d]
                dx[:, d] -= L * np.round(dx[:, d] / L)
        r = np.linalg.norm(dx, axis=1)

        p.neigh_w = [kernel(ri, h, 2) for ri in r]


def build_neigh_kdtree_3d(
    particles: List[Particle],
    h: float,
    kernel: Callable[[float, float, int], float],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    """KD-tree neighbor search in 3D with optional periodic box."""
    r_cut = qmax * h
    x = np.vstack([p.x for p in particles])  # (N,3)
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 3:
        raise ValueError("len(box) must equal 3 for 3D")

    tree = (
        cKDTree(x, boxsize=box_arr)
        if box_arr is not None
        else cKDTree(x)
    )
    neighbors_idx = tree.query_ball_point(x, r=r_cut)

    for i, p in enumerate(particles):
        idx = neighbors_idx[i]
        p.neigh = idx

        dx = x[idx] - x[i]  # (m,3)
        if box_arr is not None:
            for d in range(3):
                L = box_arr[d]
                dx[:, d] -= L * np.round(dx[:, d] / L)
        r = np.linalg.norm(dx, axis=1)

        p.neigh_w = [kernel(ri, h, 3) for ri in r]


def kd_tree(dim: int):
    return {
        1: build_neigh_kdtree_1d,
        2: build_neigh_kdtree_2d,
        3: build_neigh_kdtree_3d,
    }[dim]