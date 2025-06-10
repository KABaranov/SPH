import numpy as np
from SPH.core.particle.particle_dataclass import Particle
from SPH.configs.config_class import Config


# ------------------------------------------------------------------
# Ламинарная (физическая) вязкость SPH
# ------------------------------------------------------------------
def laminar_viscosity(
    cfg: Config,
    pi: Particle,
    pj: Particle,
    rij: np.ndarray,
) -> float:
    """
    Расчёт величины пары SPH-коэффициента вязкости для ламинарной (ньютоновской) жидкости:
    Π_ij = (mu_i + mu_j) / (rho_i * rho_j) * (v_i - v_j)·rij / (|rij|^2 + epsilon*h^2).
    Возвращает Π_ij, далее используется в a_i += m_j * Π_ij * gradW.

    Требует в cfg.viscosity_param:
      - 'mu': динамическая вязкость жидкости,
      - 'epsilon': малый регуляризационный параметр.
    """
    # Проверка параметров
    for param in ('mu', 'epsilon'):
        if param not in cfg.viscosity_param:
            raise ValueError(f"Необходимо указать '{param}' в параметрах вязкости (viscosity/laminar)")

    mu = cfg.viscosity_param['mu']
    epsilon = cfg.viscosity_param['epsilon']
    h = cfg.h

    # относительная скорость и проекция на вектор rij
    vij = pi.v - pj.v
    rij_dot = np.dot(vij, rij)

    r2 = np.dot(rij, rij)
    # динамические вязкости для частиц
    mu_i = mu
    mu_j = mu
    # ГСЧ-коэффициент вязкости
    return (mu_i + mu_j) / (pi.rho * pj.rho) * rij_dot / (r2 + epsilon * h**2)
