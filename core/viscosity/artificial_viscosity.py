import numpy as np
from SPH.core.particle.particle_dataclass import Particle
from SPH.configs.config_class import Config


# ------------------------------------------------------------------
# Искусственная вязкость Монaгана
# ------------------------------------------------------------------
def artificial_viscosity(cfg: Config,
                         pi: Particle,
                         pj: Particle,
                         rij: np.ndarray,) -> float:
    for param in ["beta", "alpha", "epsilon"]:
        if param not in cfg.viscosity_param:
            raise ValueError(f"Необходимо указать {param} в конфигурации вязкости (viscosity/artificial)")
    alpha, beta, epsilon = cfg.viscosity_param["alpha"], cfg.viscosity_param["beta"], cfg.viscosity_param["epsilon"]
    c0, h = cfg.c0, cfg.h

    vij = pi.v - pj.v
    rij_dot = np.dot(vij, rij)

    if rij_dot >= 0:
        return 0.0  # расходятся — без вязкости

    mu_ij = h * rij_dot / (np.dot(rij, rij) + epsilon * h ** 2)
    c_bar = c0  # для простоты: const звук
    rho_bar = 0.5 * (pi.rho + pj.rho)
    return (-alpha * c_bar * mu_ij + beta * mu_ij ** 2) / rho_bar
