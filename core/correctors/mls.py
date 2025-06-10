import numpy as np
from typing import List

from SPH.core.particle.particle_dataclass import Particle


def mls_correction(particles: List[Particle], n_iter: int = 2) -> None:
    """
    Коррекция плотности через локальную полиномиальную регрессию (1-й порядок).
    После вызова у каждого pi.rho ← a0 из взвешенного МНК.
    """
    for _ in range(n_iter):
        rho_orig = np.array([p.rho for p in particles])
        for pi in particles:
            d = pi.x.size               # размерность (1, 2 или 3)
            n = len(pi.neigh)           # число соседей
            if n < d+1:
                # недостаточно соседей для аппроксимации первой степени
                continue

            # Формируем матрицу M (n x (d+1)), вектор y (n), и веса W (n)
            M = np.zeros((n, d+1))
            y = np.zeros(n)
            w = np.zeros(n)
            for idx, (j, Wij) in enumerate(zip(pi.neigh, pi.neigh_w)):
                pj = particles[j]
                delta = pj.x - pi.x   # вектор разности координат
                # строка [1, Δx, Δy, Δz]
                M[idx, 0] = 1.0
                M[idx, 1:1+d] = delta
                y[idx] = rho_orig[j]
                w[idx] = Wij

            # WLS: (M^T W M) a = M^T W y
            W = np.diag(w)
            A = M.T @ W @ M
            b = M.T @ W @ y
            # решаем систему. Если A плохо обусловлена — fallback на средневзвешенную плотность
            try:
                a = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                a0 = np.sum(w * y) / (np.sum(w) + 1e-12)
                pi.rho = a0
                continue

            # скорректированная плотность — значение полинома в центре (f(0)=a0)
            pi.rho = a[0]
