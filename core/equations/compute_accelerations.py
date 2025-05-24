import numpy as np
from typing import List, Optional
from ..particle.particle_dataclass import Particle
from SPH.configs.config_class import Config
from .calculate_viscosity import artificial_viscosity


# ------------------------------------------------------------------
# Уравнение движения
# ------------------------------------------------------------------
def compute_accelerations(cfg: Config,
                          particles: List[Particle],
                          external_force: Optional[np.ndarray] = None,) -> None:
    if external_force is None:
        external_force = np.zeros_like(particles[0].v)

    for i, pi in enumerate(particles):
        acc = np.zeros_like(pi.v)
        for j in pi.neigh:
            pj = particles[j]
            rij = pi.x - pj.x
            grad_w = cfg.grad(rij)

            # Давление (symmetric form)
            pij_term = pi.p / (pi.rho ** 2.0) + pj.p / (pj.rho ** 2.0)

            # Искусственная вязкость
            visc = artificial_viscosity(pi, pj, rij, cfg.kernel)

            acc -= pj.m * (pij_term + visc) * grad_w

        pi.dv_dt = acc + external_force
