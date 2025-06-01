from typing import List
from math import ceil
from itertools import product
import numpy as np

from SPH.core.particle.particle_dataclass import Particle
from typing import Sequence

from SPH.core.neighbor_search.helpers import minimal_image


def find_neigh_cell_list(particles: List[Particle],
                         r_cut: float,
                         L: Sequence[float],
                         dim: int = 1) -> None:
    """
    Поиск соседей через связанные ячейки (cell-linked list) с периодикой.

    Параметры
    ----------
    particles : List[Particle]
        Список частиц, у каждой должно быть поле .x (array_like длины dim)
        и поле .neigh (список для индексов соседей).
    r_cut : float
        Радиус поиска соседей.
    L : float
        Длина стороны периодического домена в каждом измерении.
    dim : int
        Размерность (1, 2 или 3).

    Результат
    --------
    Для каждой частицы i заполняется поле particles[i].neigh списком
    индексов всех j≠i таких, что расстояние до j по периодике ≤ r_cut.
    """
    # сбрасываем старые списки соседей
    for p in particles:
        p.neigh = []
    L = L
    dim = dim
    # размер ячейки и число ячеек в каждом направлении
    ncell = ceil(L / r_cut)
    cell_size = L / ncell

    # распределяем частицы по ячейкам
    cell_particles = {}
    for idx, p in enumerate(particles):
        pos = np.asarray(p.x, float)
        cell_idx = tuple(int(pos[k] // cell_size) for k in range(dim))
        cell_particles.setdefault(cell_idx, []).append(idx)

    # для каждой частицы проверяем соседей в соседних ячейках
    offsets = list(product([-1, 0, 1], repeat=dim))
    for i, pi in enumerate(particles):
        pos_i = np.asarray(pi.x, float)
        cell_i = tuple(int(pos_i[k] // cell_size) for k in range(dim))

        for off in offsets:
            neigh_cell = tuple((cell_i[k] + off[k]) % ncell for k in range(dim))
            for j in cell_particles.get(neigh_cell, []):
                # if j == i:
                #     continue
                pos_j = np.asarray(particles[j].x, float)
                dx = minimal_image(pos_j - pos_i, dim, L)
                if np.linalg.norm(dx) <= r_cut:
                    pi.neigh.append(j)
