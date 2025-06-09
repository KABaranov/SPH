import numpy as np
from typing import List

from SPH.core.particle.particle_dataclass import Particle


def mls_density_correction(particles: List[Particle], n_iter: int = 2):
    """Коррекция плотности методом MLS (движущихся наименьших квадратов)."""
    rho_orig = np.array([p.rho for p in particles])
    for pi in particles:
        # Определяем размерность d (1,2,3) по размеру координаты частицы:
        d = pi.x.shape[0]  
        # Инициализируем матрицу A (размерностью (d+1)x(d+1))
        A = np.zeros((d+1, d+1))
        # Формируем вектор правой части (1,0,0,...)
        b = np.zeros(d+1)
        b[0] = 1.0
        # Заполняем матрицу A суммированием по соседям
        for j, Wij in zip(pi.neigh, pi.neigh_w):
            pj = particles[j]
            # расчет базисных функций Phi = [1, Δx, Δy, Δz] (для соответствующей размерности)
            # Вычисляем разности координат
            delta = pi.x - pj.x  # np.array размера d
            # Формируем вектор [1, delta_x, delta_y, delta_z] соответствующей длины
            if d == 1:
                phi = np.array([1.0, delta[0]])
            elif d == 2:
                phi = np.array([1.0, delta[0], delta[1]])
            elif d == 3:
                phi = np.array([1.0, delta[0], delta[1], delta[2]])
            else:
                # На случай нестандартной размерности можно собрать phi автоматически
                phi = np.concatenate(([1.0], delta))
            # Обновляем матрицу моментов: A += (m_j/ρ_j * W_ij) * (phi^T phi)
            weight = (pj.m / rho_orig[j]) * Wij
            # Outer product phi на phi^T и масштабирование:
            A += weight * np.outer(phi, phi)
        # Решаем систему A * beta = b для нахождения beta
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Если система не решается (вырожденная матрица),
            # можноFallback: beta = [1,0,...,0] т.е. без коррекции (равносильно Shepard при beta0=1)
            beta = np.zeros(d+1)
            beta[0] = 1.0
        # Используем коэффициенты beta для обновления плотности
        rho_new = 0.0
        for j, Wij in zip(pi.neigh, pi.neigh_w):
            # Вычисляем (β0 + β1*Δx + β2*Δy + β3*Δz) для данного соседа j
            pj = particles[j]
            delta = pi.x - pj.x
            # Скалярое произведение beta на [1, delta]:
            # (заметим, beta и delta имеют разную длину: добавим 1 перед delta)
            # Можно вычислить напрямую:
            factor = beta[0]
            for comp in range(d):
                factor += beta[comp+1] * delta[comp]
            # Добавляем вклад соседа в плотность
            rho_new += pj.m * factor * Wij
        pi.rho = rho_new
