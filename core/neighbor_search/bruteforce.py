from typing import List
import numpy as np

from SPH.core.particle.particle_dataclass import Particle


def find_neigh_bruteforce(particles: List[Particle], r_cut: float, is_periodic: bool = False) -> None:
    """
    Brute‐force поиск соседей: для каждой частицы заполняет поле. neigh
    (список индексов соседних частиц на расстоянии ≤ r_cut), ничего не возвращая.
    """
    # Простой O(N^2)-поиск
    for i, pi in enumerate(particles):
        # Сбрасываем старых соседей
        pi.neigh = []
        xi = np.asarray(pi.x, dtype=float)
        for j, pj in enumerate(particles):
            # if i == j:
            #     continue
            xj = np.asarray(pj.x, dtype=float)
            if np.linalg.norm(xi - xj) <= r_cut:
                pi.neigh.append(j)
