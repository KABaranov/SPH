import numpy as np
from typing import List, Callable, Sequence, Union
from SPH.core.particle.particle_dataclass import Particle


def build_neigh_1d(
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

    for i, pi in enumerate(particles):
        xi = pi.x[0]
        neigh_idx = []
        neigh_w = []
        grad_w = []

        for j, pj in enumerate(particles):
            if i == j:
                continue  # пропускаем самопересечения

            dx = xi - pj.x[0]

            # Применяем периодические ГУ
            if box_len is not None:
                dx -= box_len * round(dx / box_len)

            if dx ** 2 <= r_cut2:
                r_vec = np.array([dx, 0.0, 0.0])
                neigh_idx.append(j)
                neigh_w.append(kernel(r_vec, h, 1))
                grad_w.append(grad_kernel(r_vec, h, 1))

        pi.neigh = neigh_idx
        pi.neigh_w = neigh_w
        pi.grad_w = grad_w


def build_neigh_2d(
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

    for i, pi in enumerate(particles):
        xi, yi = pi.x[0], pi.x[1]
        neigh_idx = []
        neigh_w = []
        grad_w = []

        for j, pj in enumerate(particles):
            if i == j:
                continue

            dx = xi - pj.x[0]
            dy = yi - pj.x[1]

            # Применяем периодические ГУ
            if box_arr is not None:
                dx -= box_arr[0] * round(dx / box_arr[0])
                dy -= box_arr[1] * round(dy / box_arr[1])

            dist2 = dx ** 2 + dy ** 2
            if dist2 <= r_cut2:
                r_vec = np.array([dx, dy, 0.0])
                neigh_idx.append(j)
                neigh_w.append(kernel(r_vec, h, 2))
                grad_w.append(grad_kernel(r_vec, h, 2))

        pi.neigh = neigh_idx
        pi.neigh_w = neigh_w
        pi.grad_w = grad_w


def build_neigh_3d(
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

    for i, pi in enumerate(particles):
        xi, yi, zi = pi.x[0], pi.x[1], pi.x[2]
        neigh_idx = []
        neigh_w = []
        grad_w = []

        for j, pj in enumerate(particles):
            if i == j:
                continue

            dx = xi - pj.x[0]
            dy = yi - pj.x[1]
            dz = zi - pj.x[2]

            # Применяем периодические ГУ
            if box_arr is not None:
                dx -= box_arr[0] * round(dx / box_arr[0])
                dy -= box_arr[1] * round(dy / box_arr[1])
                dz -= box_arr[2] * round(dz / box_arr[2])

            dist2 = dx ** 2 + dy ** 2 + dz ** 2
            if dist2 <= r_cut2:
                r_vec = np.array([dx, dy, dz])
                neigh_idx.append(j)
                neigh_w.append(kernel(r_vec, h, 3))
                grad_w.append(grad_kernel(r_vec, h, 3))

        pi.neigh = neigh_idx
        pi.neigh_w = neigh_w
        pi.grad_w = grad_w


def bruteforce(dim: int):
    return {
        1: build_neigh_1d,
        2: build_neigh_2d,
        3: build_neigh_3d,
    }[dim]