from typing import List, Sequence, Callable, Dict, Tuple
import numpy as np

from SPH.core.particle.particle_dataclass import Particle


def build_neigh_cl_1d(
    particles: List[Particle], h: float,
    kernel: Callable[[float, float, int], float],
    box: float | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut**2
    box_len = float(box) if box is not None else None
    # число ячеек
    ncells = int(np.floor(box_len / r_cut)) if box_len is not None else None
    # распределяем частицы по ячейкам
    cell_map: Dict[int, List[int]] = {}
    for idx, p in enumerate(particles):
        x = p.x[0] % box_len if box_len is not None else p.x[0]
        cell = int(np.floor(x / r_cut)) % ncells if box_len is not None else int(np.floor(x / r_cut))
        cell_map.setdefault(cell, []).append(idx)
    # поиск соседей
    for cell, idxs in cell_map.items():
        for i in idxs:
            pi = particles[i]
            xi = pi.x[0] % box_len if box_len is not None else pi.x[0]
            neigh_idx: List[int] = []
            neigh_w: List[float] = []
            for offset in (-1, 0, 1):
                nb = (cell + offset) % ncells if box_len is not None else cell + offset
                for j in cell_map.get(nb, []):
                    xj = particles[j].x[0] % box_len if box_len is not None else particles[j].x[0]
                    dx = xi - xj
                    if box_len is not None:
                        dx -= box_len * np.round(dx / box_len)
                    if dx*dx <= r_cut2:
                        neigh_idx.append(j)
                        neigh_w.append(kernel(abs(dx), h, 1))
            pi.neigh = neigh_idx
            pi.neigh_w = neigh_w


def build_neigh_cl_2d(
    particles: List[Particle], h: float,
    kernel: Callable[[float, float, int], float],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    r_cut2 = r_cut**2
    box_arr = np.asarray(box, float) if box is not None else None
    # число ячеек по осям
    if box_arr is not None:
        nx = int(np.floor(box_arr[0] / r_cut))
        ny = int(np.floor(box_arr[1] / r_cut))
    else:
        nx = ny = None
    # распределяем частицы по ячейкам
    cell_map: Dict[Tuple[int, int], List[int]] = {}
    for idx, p in enumerate(particles):
        coords = p.x[:2].copy()
        if box_arr is not None:
            coords %= box_arr
        ix = int(np.floor(coords[0] / r_cut)) % nx if box_arr is not None else int(np.floor(coords[0] / r_cut))
        iy = int(np.floor(coords[1] / r_cut)) % ny if box_arr is not None else int(np.floor(coords[1] / r_cut))
        cell_map.setdefault((ix, iy), []).append(idx)
    # поиск соседей
    for (ix, iy), idxs in cell_map.items():
        for i in idxs:
            pi = particles[i]
            coords_i = pi.x[:2].copy()
            if box_arr is not None:
                coords_i %= box_arr
            neigh_idx: List[int] = []
            neigh_w: List[float] = []
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
                        if np.dot(delta, delta) <= r_cut2:
                            neigh_idx.append(j)
                            neigh_w.append(kernel(np.linalg.norm(delta), h, 2))
            pi.neigh = neigh_idx
            pi.neigh_w = neigh_w


def build_neigh_cl_3d(
    particles: List[Particle], h: float,
    kernel: Callable[[float, float, int], float],
    box: Sequence[float] | None = None,
    qmax: float = 2.0,
):
    r_cut = qmax * h
    box_arr = np.asarray(box, float) if box is not None else None
    cell_map: Dict[Tuple[int, int, int], List[int]] = {}
    # assign particles to cells
    for idx, p in enumerate(particles):
        coords = p.x.copy()
        if box_arr is not None:
            coords %= box_arr
        cell = tuple((coords // r_cut).astype(int))
        cell_map.setdefault(cell, []).append(idx)
    # neighbor search
    for (cx, cy, cz), idxs in cell_map.items():
        for i in idxs:
            pi = particles[i]
            coords_i = pi.x.copy()
            if box_arr is not None:
                coords_i %= box_arr
            neigh_idx: List[int] = []
            neigh_w: List[float] = []
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for dz_cell in (-1, 0, 1):
                        for j in cell_map.get((cx + dx_cell, cy + dy_cell, cz + dz_cell), []):
                            pj = particles[j]
                            coords_j = pj.x.copy()
                            if box_arr is not None:
                                coords_j %= box_arr
                            delta = coords_i - coords_j
                            if box_arr is not None:
                                delta -= box_arr * np.round(delta / box_arr)
                            dist2 = np.dot(delta, delta)
                            if dist2 <= r_cut**2:
                                neigh_idx.append(j)
                                neigh_w.append(kernel(np.sqrt(dist2), h, 3))
            pi.neigh = neigh_idx
            pi.neigh_w = neigh_w


def cell_linked(dim: int):
    return {1: build_neigh_cl_1d, 2: build_neigh_cl_2d, 3: build_neigh_cl_3d}[dim]
