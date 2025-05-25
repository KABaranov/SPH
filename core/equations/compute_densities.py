from typing import List
from SPH.core.particle.particle_dataclass import Particle


# ------------------------------------------------------------------
# Расчёт плотности
# ------------------------------------------------------------------
def compute_densities(particles: List[Particle]) -> None:
    """ρᵢ = Σⱼ mⱼ W(rᵢⱼ)"""
    for pi in particles:
        rho = 0.0
        for w, j in zip(pi.neigh_w, pi.neigh):
            rho += particles[j].m * w
        pi.rho = rho
