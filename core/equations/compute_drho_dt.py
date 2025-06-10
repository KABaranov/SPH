import numpy as np
from typing import List
from SPH.core.particle.particle_dataclass import Particle
from SPH.configs.config_class import Config


# ------------------------------------------------------------------
# Континуитет: dρ/dt
# ------------------------------------------------------------------
def compute_drho_dt(cfg: Config, particles: List[Particle]) -> None:
    for i, pi in enumerate(particles):
        drho = 0.0
        for j in pi.neigh:
            pj = particles[j]
            vij = pi.v - pj.v
            grad = cfg.grad(pi.x - pj.x)
            drho += pj.m * np.dot(vij, grad)
        pi.drho_dt = drho