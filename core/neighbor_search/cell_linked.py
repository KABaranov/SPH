import numpy as np
from typing import List, Sequence, Callable, Dict, Tuple
from SPH.core.particle.particle_dataclass import Particle


# --- Cell-linked neighbor search with grad_w for 1D, 2D, 3D ---

def build_neigh_cl_1d(
        particles: List[Particle],
        h: float,
        kernel: Callable[[np.ndarray, float, int], float],
        grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
        box: float | None = None,
        qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut ** 2
    box_len = float(box) if box is not None else None
    ncells = int(np.floor(box_len / r_cut)) if box_len is not None else None

    # assign particles to cells
    cell_map: Dict[int, List[int]] = {}
    for idx, p in enumerate(particles):
        x = p.x[0] % box_len if box_len is not None else p.x[0]
        cell = (int(np.floor(x / r_cut)) % ncells) if box_len is not None else int(np.floor(x / r_cut))
        cell_map.setdefault(cell, []).append(idx)

    # search neighbors
    for cell, idxs in cell_map.items():
        for i in idxs:
            pi = particles[i]
            xi = pi.x[0] % box_len if box_len is not None else pi.x[0]
            neigh_idx: List[int] = []
            neigh_w: List[float] = []
            grad_w: List[np.ndarray] = []
            for offset in (-1, 0, 1):
                nb = (cell + offset) % ncells if box_len is not None else cell + offset
                for j in cell_map.get(nb, []):
                    xj = particles[j].x[0] % box_len if box_len is not None else particles[j].x[0]
                    dx = xi - xj
                    if box_len is not None:
                        dx -= box_len * np.round(dx / box_len)
                    if dx * dx <= r_cut2:
                        # build 3D vector
                        r_vec = np.array([dx, 0.0, 0.0], dtype=float)
                        neigh_idx.append(j)
                        neigh_w.append(kernel(r_vec, h, 1))
                        grad_w.append(grad_kernel(r_vec, h, 1))
            pi.neigh = neigh_idx
            pi.neigh_w = neigh_w
            pi.grad_w = grad_w


def build_neigh_cl_2d(
        particles: List[Particle],
        h: float,
        kernel: Callable[[np.ndarray, float, int], float],
        grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
        box: Sequence[float] | None = None,
        qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut ** 2
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 2:
        raise ValueError("len(box) must equal 2 for 2D")
    nx = int(np.floor(box_arr[0] / r_cut)) if box_arr is not None else None
    ny = int(np.floor(box_arr[1] / r_cut)) if box_arr is not None else None

    # assign particles to cells
    cell_map: Dict[Tuple[int, int], List[int]] = {}
    for idx, p in enumerate(particles):
        coords = p.x[:2].copy()
        if box_arr is not None:
            coords %= box_arr
        ix = (int(np.floor(coords[0] / r_cut)) % nx) if box_arr is not None else int(np.floor(coords[0] / r_cut))
        iy = (int(np.floor(coords[1] / r_cut)) % ny) if box_arr is not None else int(np.floor(coords[1] / r_cut))
        cell_map.setdefault((ix, iy), []).append(idx)

    # search neighbors
    for (ix, iy), idxs in cell_map.items():
        for i in idxs:
            pi = particles[i]
            coords_i = pi.x[:2].copy()
            if box_arr is not None:
                coords_i %= box_arr
            neigh_idx: List[int] = []
            neigh_w: List[float] = []
            grad_w: List[np.ndarray] = []
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    jx = (ix + dx_cell) % nx if box_arr is not None else ix + dx_cell
                    jy = (iy + dy_cell) % ny if box_arr is not None else iy + dy_cell
                    for j in cell_map.get((jx, jy), []):
                        coords_j = particles[j].x[:2].copy()
                        if box_arr is not None:
                            coords_j %= box_arr
                        delta = coords_i - coords_j
                        if box_arr is not None:
                            for d in (0, 1):
                                delta[d] -= box_arr[d] * np.round(delta[d] / box_arr[d])
                        dist2 = np.dot(delta, delta)
                        if dist2 <= r_cut2:
                            # build 3D vector
                            r_vec = np.array([delta[0], delta[1], 0.0], dtype=float)
                            neigh_idx.append(j)
                            neigh_w.append(kernel(r_vec, h, 2))
                            grad_w.append(grad_kernel(r_vec, h, 2))
            pi.neigh = neigh_idx
            pi.neigh_w = neigh_w
            pi.grad_w = grad_w


def build_neigh_cl_3d(
        particles: List[Particle],
        h: float,
        kernel: Callable[[np.ndarray, float, int], float],
        grad_kernel: Callable[[np.ndarray, float, int], np.ndarray],
        box: Sequence[float] | None = None,
        qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut ** 2
    box_arr = np.asarray(box, float) if box is not None else None
    if box_arr is not None and box_arr.size != 3:
        raise ValueError("len(box) must equal 3 for 3D")

    # assign particles to cells
    cell_map: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, p in enumerate(particles):
        coords = p.x.copy()
        if box_arr is not None:
            coords %= box_arr
        cell = tuple((coords // r_cut).astype(int))
        cell_map.setdefault(cell, []).append(idx)

    # search neighbors
    for (cx, cy, cz), idxs in cell_map.items():
        for i in idxs:
            pi = particles[i]
            coords_i = pi.x.copy()
            if box_arr is not None:
                coords_i %= box_arr
            neigh_idx: List[int] = []
            neigh_w: List[float] = []
            grad_w: List[np.ndarray] = []
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for dz_cell in (-1, 0, 1):
                        nb = (cx + dx_cell, cy + dy_cell, cz + dz_cell)
                        if box_arr is not None:
                            nb = (nb[0] % int(box_arr[0] // r_cut), nb[1] % int(box_arr[1] // r_cut),
                                  nb[2] % int(box_arr[2] // r_cut))
                        for j in cell_map.get(nb, []):
                            coords_j = particles[j].x.copy()
                            if box_arr is not None:
                                coords_j %= box_arr
                            delta = coords_i - coords_j
                            if box_arr is not None:
                                delta -= box_arr * np.round(delta / box_arr)
                            dist2 = np.dot(delta, delta)
                            if dist2 <= r_cut2:
                                neigh_idx.append(j)
                                neigh_w.append(kernel(delta, h, 3))
                                grad_w.append(grad_kernel(delta, h, 3))
            pi.neigh = neigh_idx
            pi.neigh_w = neigh_w
            pi.grad_w = grad_w


def cell_linked(dim: int):
    return {1: build_neigh_cl_1d, 2: build_neigh_cl_2d, 3: build_neigh_cl_3d}[dim]
