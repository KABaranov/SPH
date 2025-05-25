from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from compute_densities import compute_densities
from eos import eos

import matplotlib.pyplot as plt
import numpy as np

from SPH.configs.get_config import get_config


# ===========================================
# Проверка eos
# ===========================================

# Проверка нулевого давления rho=rho_0
# Для классической формы уравнения Тейта (см. функцию eos)
# при rho=rho_0 давление строго равно 0 (если p_floor <= 0)
def test_eos_at_rho0(cfg: Config) -> bool:
    return eos(cfg.rho0, cfg) == max(0.0, cfg.p_floor)


# Проверка монотонности (2 функции)
# Функция должна быть строго возрастающей
# Если rho_2 > rho_1, то p(rho_2) > p(rho_1) (при любом gamma > 1)
def test_eos_monotonic(cfg: Config, k1: float, k2: float) -> bool:
    p1 = eos(k1*cfg.rho0, cfg)
    p2 = eos(k2*cfg.rho0, cfg)
    return p2 > p1


# Функция построения графика монотонности
def plot_eos_monotonic(cfg: Config, a: float = 0.0, b: float = 2.0) -> None:
    rho = np.linspace(a * cfg.rho0, b * cfg.rho0, int(cfg.rho0 * 2.0))
    p = [eos(rho_i, cfg) for rho_i in rho]
    plt.plot(rho, p)
    plt.title("Монотонность функции p(rho)")
    plt.show()


# Правильная скорость звука
# Скорость звука для Тейта - это корень частной производной dp/drho
# Функция проверяет численную производную
# (Может не выдавать True из-за p_floor)
def sound_speed(rho: float, cfg: Config):
    """Аналитический c(ρ) без учёта p_floor."""
    return np.sqrt(cfg.gamma * cfg.B / cfg.rho0) * (rho/cfg.rho0)**((cfg.gamma-1)/2)


def test_sound_speed(cfg: Config, delta: float = 1e-6):
    rho = cfg.rho0 * 1.02            # на 2 % выше, чтобы точно > p_floor
    dpdrho_num = (eos(rho*(1+delta), cfg) - eos(rho*(1-delta), cfg)) / (2*rho*delta)
    c_num = np.sqrt(dpdrho_num)
    c_theory = sound_speed(rho, cfg)
    return abs(c_num - c_theory)/c_theory < 1e-4


# Функция проверки eos на математическую корректность
# запускает 3 теста
def test_eos_all(cfg: Config, k1: float = 0.95, k2: float = 1.05,
                 delta: float = 1e-6) -> None:
    print("Проверка eos на математическую корректность:")
    r1 = test_eos_at_rho0(cfg)
    print(f"\tПроверка нулевого давления{'' if r1 else ' не'} пройдена")
    r2 = test_eos_monotonic(cfg, k1, k2)
    print(f"\tПроверка монотонности{'' if r2 else ' не'} пройдена")
    r3 = test_sound_speed(cfg, delta)
    print(f"\tПроверка правильной скорости звука{'' if r3 else ' не'} пройдена")
    print("\t============================")
    result = r1 * r2 * r3
    print(f"\tПроверка{'' if result else ' не'} пройдена")


# =============================================
# Проверка расчёта плотности
# =============================================

# Тест 1: Однородная решётка 2D
def density_test1(cfg: Config, out_plot=False) -> None:
    print("Проверка расчёта плотности (Тест 1):")
    width, height, dx = cfg.width, cfg.height, cfg.dx
    kernel = cfg.kernel
    x, y = [i * dx for i in range(int(width // dx) + 2)], [i * dx for i in range(int(height // dx) + 2)]
    particles = []
    for xi in x:
        for yi in y:
            p = Particle(
                id=len(particles), m=cfg.rho0*(dx**cfg.dim), p=0, x=np.array([xi, yi, 0]),
                drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=cfg.h,
                neigh=[], neigh_w=[], rho=cfg.rho0, v=np.array([0, 0, 0])
            )
            particles.append(p)
    for pi in particles:
        for pj in particles:
            dr = pj.x - pi.x
            pi.neigh.append(pj.id)
            pi.neigh_w.append(kernel(r=dr, h=pi.h, dim=cfg.dim))

    compute_densities(particles)

    x_out, y_out, rho_out = [], [], []
    for pi in particles:
        x_out.append(pi.x[0])
        y_out.append(pi.x[1])
        rho_out.append(pi.rho)
    rho_min, rho_max = min(rho_out), max(rho_out)
    print(f"\tМинимальная плотность: {rho_min}\n\tМаксимальная плотность {rho_max}")
    if out_plot:
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(rho_min, rho_max)
        line_colors = cmap(norm(rho_out))
        plt.scatter(x_out, y_out, color=line_colors)
        plt.show()


# Тест 2: Синус-мода
def density_test2(cfg: Config, out_plot=False) -> None:
    print("Проверка расчёта плотности (Тест 2):")
    Ns = np.array([200, 400, 800, 1600])
    errs = []
    for N in Ns:
        dim, L = 1, 1
        dx = L / N
        eps = 0.1  # Амплитуда синуса
        k = 2 * np.pi / L  # одна волна на домен
        h = 1.3 * dx
        kernel = cfg.kernel

        xs = np.linspace(0, L - dx, N)
        rho0 = 1000.0  # целевая базовая плотность
        m0 = rho0 * dx ** dim  # “средняя” масса
        m_i = m0 * (1 + eps * np.sin(k * xs))  # переменная масса → синус в ρ
        particles = []
        for x, m in zip(xs, m_i):
            p = Particle(
                id=len(particles), m=m, p=0, x=np.array([x, 0, 0]),
                drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=h,
                neigh=[], neigh_w=[], rho=rho0, v=np.array([0, 0, 0])
            )
            particles.append(p)
        for pi in particles:
            for pj in particles:
                dr = pj.x - pi.x
                dr -= L * round(dx / L)  # периодические границы
                pi.neigh.append(pj.id)
                pi.neigh_w.append(kernel(r=dr, h=pi.h, dim=1))

        compute_densities(particles)

        rho_exact = rho0 * (1 + eps * np.sin(k * xs))
        rho_num = np.array([p.rho for p in particles])
        err_L2 = np.linalg.norm(rho_num - rho_exact) / np.linalg.norm(rho_exact)
        err_Linf = np.max(np.abs(rho_num - rho_exact)) / rho0
        print("\t", err_L2, err_Linf)
        errs.append(err_L2)
    print("\t", errs)


if __name__ == "__main__":
    config = get_config("common", print_param=True)
    test_eos_all(config)
    density_test2(config, out_plot=True)
