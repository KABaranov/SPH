import numpy as np
from typing import List, Optional

from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from SPH.core.viscosity.get_viscosity import get_viscosity


# ------------------------------------------------------------------
# Уравнение движения
# ------------------------------------------------------------------
def compute_accelerations(cfg: Config,
                          particles: List[Particle],
                          external_force: Optional[np.ndarray] = None,) -> None:
    if external_force is None:
        external_force = np.zeros_like(particles[0].v, dtype=float)

    for i, pi in enumerate(particles):

        acc = np.zeros_like(pi.v, dtype=float)
        for jdx, j in enumerate(pi.neigh):
            pj = particles[j]
            rij = pi.x - pj.x
            grad_w = pi.grad_w[jdx]

            # Давление (symmetric form)
            pij_term = pi.p / (pi.rho ** 2.0) + pj.p / (pj.rho ** 2.0)
            # print(pi.p, pi.rho, pj.p, pj.rho)

            # Искусственная вязкость
            visc = get_viscosity(cfg.viscosity_name)(cfg, pi, pj, rij)
            acc -= grad_w * pj.m * (pij_term + visc)

        pi.dv_dt = acc + external_force
