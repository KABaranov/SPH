import numpy as np
from ..particle.particle_dataclass import Particle
from SPH.configs.config_class import Config


# ------------------------------------------------------------------
# Искусственная вязкость Монaгана
# ------------------------------------------------------------------
def artificial_viscosity(cfg: Config,
                         pi: Particle,
                         pj: Particle,
                         rij: np.ndarray,) -> float:
    vij = pi.v - pj.v
    rij_dot = np.dot(vij, rij)
    if rij_dot >= 0:
        return 0.0  # расходятся — без вязкости

    h = cfg.kernel.h
    mu_ij = h * rij_dot / (np.dot(rij, rij) + cfg.epsilon * h ** 2)
    c_bar = cfg.c0  # для простоты: const звук
    rho_bar = 0.5 * (pi.rho + pj.rho)
    return (-cfg.alpha * c_bar * mu_ij + cfg.beta * mu_ij ** 2) / rho_bar