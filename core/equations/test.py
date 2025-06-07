from __future__ import annotations

from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from compute_densities import compute_densities
from eos import eos

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, List, Callable

from SPH.configs.get_config import get_config


def build_neigh(
    particles: List[Particle],
    h: float,
    kernel: Callable[[np.ndarray, float, int], np.ndarray],
    dim: int = 1,
    box: Sequence[float] | float | int | None = None,   # теперь допускает скаляр
    qmax: float = 2,
):
    """Формирует neigh/neigh_w с поддержкой скалярного `box` в 1-D."""
    r_cut = qmax * h
    r_cut2 = r_cut * r_cut
    x = np.asarray([p.x for p in particles])          # (N, dim)

    # --- нормализуем box ---
    if box is None:
        box_arr = None
    else:
        if np.isscalar(box):                         # int/float → длина в 1-D
            if dim != 1:
                raise ValueError("Scalar `box` допустим только при dim=1")
            box_arr = np.asarray([box], dtype=float)
        else:                                        # последовательность
            box_arr = np.asarray(box, dtype=float)
            if box_arr.size != dim:
                raise ValueError("len(box) должно равняться dim")

    # --- главный цикл ---
    for i, pi in enumerate(particles):
        dx = x - pi.x                                # (N, dim)

        # минимальное изображение, если box задан
        if box_arr is not None:
            for d in range(dim):
                L = box_arr[d]
                dx[:, d] -= L * np.round(dx[:, d] / L)

        r2 = np.einsum("ij,ij->i", dx, dx)
        mask = r2 <= r_cut2
        idx = np.nonzero(mask)[0]

        r = np.sqrt(r2[mask])
        w = [kernel(ri, h, dim) for ri in r]

        pi.neigh = idx.tolist()
        pi.neigh_w = w


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


# Для периодических границ потребуется найти адекватную длину (не 0.98L, а -0.02L)
def dx_periodic(dx, box_L):
    """вернуть кратчайший вектор со знаком в 1-D периодике"""
    if dx >  0.5 * box_L:
        dx -= box_L
    elif dx < -0.5 * box_L:
        dx += box_L
    return dx


def check_moments(parts: List[Particle], h: float, rho0: List[float],
                  L: float = -1) -> None:
    for p in parts[:5]:
        M0 = M1 = 0.0
        for w,j in zip(p.neigh_w, p.neigh):
            dx = parts[j].x[0] - p.x[0]
            if L != -1:
                dx = dx_periodic(dx, L)
            fac = parts[j].m / rho0[j]
            M0 += fac*w
            M1 += fac*w*dx
        print(f"|M0-1|={abs(M0-1):.2e},  |M1|/h={abs(M1)/h:.2e}")


def mls_density_full(particles: List[Particle], L: float = -1, dim: int = 1) -> None:
    """
    MLS-коррекция плотности первого порядка.
    После вызова у каждой частицы поле rho имеет Δx²-точность
    независимо от ядра (поддержка 2h, хвосты, усечённый гаусс и т.д.).
    """
    for pi in particles:
        # --- нулевой и первые моменты -------------------
        M0 = 0.0
        b  = np.zeros(dim)
        A  = np.zeros((dim, dim))

        for w, j in zip(pi.neigh_w, pi.neigh):
            pj   = particles[j]
            fac  = pj.m / pj.rho          # m_j / ρ_j
            dx   = pj.x - pi.x            # вектор сдвига
            if L != 1:
                dx = dx_periodic(dx, L)
            M0  += fac * w
            b   += fac * w * dx
            A   += fac * w * np.outer(dx, dx)

        # --- решение A a = -b ---------------------------
        # добавляем малый диагональный шум, если A плохо обусловлена
        a = np.linalg.solve(A + 1e-14*np.eye(dim), -b)

        # --- скорректированная плотность ----------------
        num = 0.0
        for w, j in zip(pi.neigh_w, pi.neigh):
            dx = particles[j].x - pi.x
            if L != -1:
                dx = dx_periodic(dx, L)
            num += particles[j].m * w * (1.0 + np.dot(a, dx))

        denom = M0 + np.dot(a, b)         # = Σ (m/ρ) W (1 + a·dx)
        pi.rho = num / denom


def mls_correction(particles: List[Particle], L: float = -1.0, n_iter: int = 2) -> None:
    """
    L - длина рассматриваемой зоны (для периодических границ
    Однопроходная MLS-коррекция плотности:
    ρ_i ← Σ m_j W_ij / ( a_0 + aᵀ·0 )   (нулевой и линейный порядки).
    Корректирует нулевой и первый моменты ⇒ устраняет O(Δx) остаток.
    Работает при любых массах.
    """
    for _ in range(n_iter):
        rho0 = [p.rho for p in particles]  # исходная плотность
        for i, pi in enumerate(particles):
            M0 = M1 = M2 = 0.0
            for w, j in zip(pi.neigh_w, pi.neigh):
                pj = particles[j]
                dx = pj.x[0] - pi.x[0]
                if L != -1:
                    dx = dx_periodic(dx, L)
                fac = pj.m / rho0[j]  # ОБЯЗАТЕЛЬНО ρ₀!
                M0 += fac * w
                M1 += fac * w * dx
                M2 += fac * w * dx * dx
            a = -M1 / M2  # коэффициент линейной поправки

            num = 0.0
            for w, j in zip(pi.neigh_w, pi.neigh):
                pj = particles[j]
                dx = pj.x[0] - pi.x[0]
                if L != -1:
                    dx = dx_periodic(dx, L)
                num += particles[j].m * w * (1.0 + a * dx)  # ← линейный множитель

            pi.rho = num / (M0 + a * M1)  # = Σ (m/ρ₀) W (1+a·dx)


# Тест 2: Синус-мода
def density_test2(cfg: Config, out_plot=False) -> None:
    print("Проверка расчёта плотности (Тест 2):")
    # Ns = np.array([100, 200, 400, 800, 1600])
    Ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # Ns = np.array([100])
    errs = []
    dim, L = 1, 1000.0
    eps = 0.1  # Амплитуда синуса
    k = 2 * np.pi / L  # одна волна на домен
    # При помощи параметра alpha и kappa0 мы контролируем рост соседей
    # Т.к. ошибка зависит от кол-ва соседей
    # Если h уменьшается, при постоянном n, то в один момент ошибка "застынет"
    # Оптимальные настройки в 1D
    # WendlandC2: kappa0 = 1.3; alpha = 1.0; beta = 0.5
    # Cubic Spline: kappa0 = 3.0; alpha = 1.0; beta = 0.0 + 2 mls_corrector
    # Gauss: kappa0 = 3.0; alpha = 1.0; beta = 0.0 + 1 shepard_corrector
    kappa0 = 3.0
    alpha = 1.0
    beta = 0.0
    rho0 = 1000.0  # целевая базовая плотность
    qmax = 10.0  # Усечение соседей для build_neigh
    kernel = cfg.kernel
    print(f"alpha={alpha}\tbeta: {beta}\tkappa0={kappa0}\tkernel={cfg.kernel_name}\n")
    if out_plot:
        N = max(Ns)
        dx = L / N
        xs = np.linspace(0, L - dx, N)
        rho_exact = rho0 * (1 + eps * np.sin(k * xs))
        plt.plot(xs, rho_exact)

    for N in Ns:
        dx = L / N
        kappa = kappa0 * (N / 128) ** beta
        h = kappa * dx**alpha

        xs = np.linspace(0, L - dx, N)
        m0 = rho0 * (dx ** dim)  # “средняя” масса
        m_i = m0 * (1 + eps * np.sin(k * xs))  # переменная масса → синус в ρ
        particles = []
        for x, m in zip(xs, m_i):
            p = Particle(
                id=len(particles), m=m, p=0, x=np.array([x, 0, 0]),
                drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=h,
                neigh=[], neigh_w=[], rho=rho0, v=np.array([0, 0, 0])
            )
            particles.append(p)
        build_neigh(particles, h, kernel=kernel, box=L, qmax=qmax)
        mean_n = np.mean([len(p.neigh) for p in particles])
        print(f"\tN: {N}\th: {h}\tmean_n: {mean_n}")

        compute_densities(particles)
        rho0_list = np.fromiter((p.rho for p in particles), float)
        shepard_filter(particles, n_iter=1)
        # mls_correction(particles, L=L, n_iter=2)
        check_moments(particles, h, rho0_list, L=L)
        rho_num = np.array([p.rho for p in particles])
        rho_exact = rho0 * (1 + eps * np.sin(k * xs))
        # Вывод для сравнения rho и периодических границ
        if out_plot:
            plt.plot(xs, rho_num)
        err_L2 = np.linalg.norm(rho_num - rho_exact) / np.linalg.norm(rho_exact)
        err_Linf = np.max(np.abs(rho_num - rho_exact)) / rho0
        print(f"\tN: {N} errL2: {err_L2} err_Linf: {err_Linf}\n")
        errs.append(err_L2)
    errs = np.array(errs)
    print(f"errs: {errs}")
    print(f"p_est: {np.log(errs[:-1]/errs[1:]) / np.log(2)}")
    if out_plot:
        leg = ["Аналитически"] + [f"N={N}" for N in Ns]
        plt.legend(leg)
        plt.title("Сравнение аналитического и численных решений")
        plt.show()
        plt.loglog(Ns, errs)
        plt.title("Убывание ошибки при увеличении числа узлов в сетке (log-log)")
        plt.show()


if __name__ == "__main__":
    config = get_config("common", print_param=True)
    test_eos_all(config)
    density_test2(cfg=config, out_plot=True)
