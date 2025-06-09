import numpy as np
from typing import List

from SPH.core.particle.particle_dataclass import Particle


def shepard_filter(particles: List[Particle], n_iter: int = 2) -> None:
    for _ in range(n_iter):
        rho_orig = np.array([p.rho for p in particles])
        for pi in particles:
            w_sum = rho_sum = 0.0
            for w, j in zip(pi.neigh_w, pi.neigh):
                pj = particles[j]
                w_sum += (pj.m / rho_orig[j]) * w
                rho_sum += pj.m * w
            pi.rho = rho_sum / w_sum


# Не работает из-за того, что надо сохранять rho_0, но если начнёт вылезать ошибка деления на 0,
# то можно взять за идею этот прототип
def shepard_filter_density_correction(particles: List[Particle], n_iter: int) -> None:
    """Применение фильтра Шепарда для коррекции плотности всех частиц."""
    for i in range(n_iter):
        for pi in particles:
            # Вычисляем нормирующий коэффициент C_i
            norm_sum = 0.0
            for j, Wij in zip(pi.neigh, pi.neigh_w):
                # pj - соседняя частица, Wij - значение ядра W_{ij}
                norm_sum += Wij * (particles[j].m / particles[j].rho)
            if norm_sum > 1e-8:
                Ci = 1.0 / norm_sum
            else:
                Ci = 1.0  # на случай, если соседей нет или сумма слишком мала
            # Обновляем плотность: rho_i_new = C_i * sum_j (m_j W_ij)
            rho_new = 0.0
            for j, Wij in zip(pi.neigh, pi.neigh_w):
                rho_new += particles[j].m * Wij
            pi.rho = Ci * rho_new
